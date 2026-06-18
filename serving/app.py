"""
==============================================================================
Phase 5: FastAPI Model Serving Microservice
==============================================================================
Production-ready REST API for mule detection scoring, explanations, and policy
controls with authentication, audit persistence, and request protection.
==============================================================================
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from serving.auth import auth_manager
from serving.config import settings
from serving.feature_pipeline import BANK_KEY_FEATURES, FeaturePipeline
from serving.stacking_bundle import StackingBundle
from serving.store import audit_store
from serving.typology_engine import TypologyEngine, score_to_suspicion, typology_score_boost


class JsonFormatter(logging.Formatter):
    """Emit compact JSON logs for easier ingestion by log pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "path", "method", "status_code", "latency_ms", "account_id", "role", "event"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return logging.getLogger("mule_detector")


logger = configure_logging()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FRONTEND_DIR = BASE_DIR / "frontend"

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
logger.info("Loading model artifacts...")
try:
    feature_pipeline = FeaturePipeline.load(MODELS_DIR / "feature_pipeline.pkl")
    best_model_path = MODELS_DIR / "best_model.pkl"
    best_model = joblib.load(best_model_path) if best_model_path.exists() else joblib.load(MODELS_DIR / "lgbm_final.pkl")
    iso_forest = joblib.load(MODELS_DIR / "isolation_forest.pkl")
    robust_scaler = joblib.load(MODELS_DIR / "robust_scaler_anomaly.pkl")
    feature_scaler = joblib.load(MODELS_DIR / "scaler.pkl") if (MODELS_DIR / "scaler.pkl").exists() else None
    typology_engine = TypologyEngine()
    with open(MODELS_DIR / "best_model_metadata.json", encoding="utf-8") as f:
        model_metadata = json.load(f)
    OPTIMAL_THRESHOLD = model_metadata.get(
        "precision_optimal_threshold",
        model_metadata.get("optimal_threshold", 0.50),
    )
    SCALED_MODEL_NAMES = set(model_metadata.get("scaled_model_names", []))
    logger.info(
        "Model loaded: %s  threshold=%.3f",
        model_metadata.get("best_model", "LightGBM"),
        OPTIMAL_THRESHOLD,
    )
except Exception as e:
    raise RuntimeError(
        "Missing trained model artifacts. Run notebooks/01_feature_engineering.py, "
        "notebooks/02_model_training.py, and notebooks/03_anomaly_detection.py "
        "before starting the API."
    ) from e

_shap_explainer = None
MAX_CACHE_SIZE = 1000
MAX_BATCH_SIZE = settings.max_batch_size
CURRENT_DECISION_MODE = "balanced"
SCORED_ACCOUNTS_CACHE: dict[str, dict[str, Any]] = {}
MANUAL_OVERRIDES: dict[str, dict[str, Any]] = {}
REQUEST_COUNTS: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMIT_HITS = 0
AUTH_FAILURES = 0
REQUEST_COUNTER = 0

DECISION_PRESETS = {
    "strict": {"BLOCK": 0.75, "CHALLENGE": 0.55, "REVIEW": 0.35},
    "balanced": {"BLOCK": 0.85, "CHALLENGE": 0.65, "REVIEW": 0.45},
    "loose": {"BLOCK": 0.95, "CHALLENGE": 0.80, "REVIEW": 0.60},
}


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    actor: str
    role: str
    authenticated: bool


class SlidingWindowRateLimiter:
    def __init__(self, window_seconds: int = 60) -> None:
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        now = time.time()
        bucket = self._buckets[key]
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            return False, 0
        bucket.append(now)
        return True, max(0, limit - len(bucket))


rate_limiter = SlidingWindowRateLimiter(settings.request_window_seconds)


def normalize_decision_mode(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    aliases = {
        "strict": "strict",
        "stricter": "strict",
        "balanced": "balanced",
        "default": "balanced",
        "loose": "loose",
        "looser": "loose",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported decision mode: {mode}")
    return aliases[normalized]


def current_thresholds() -> Dict[str, float]:
    return DECISION_PRESETS[CURRENT_DECISION_MODE]


def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        import shap

        if hasattr(best_model, "predict_proba") and not isinstance(best_model, StackingBundle):
            _shap_explainer = shap.TreeExplainer(best_model)
        elif (MODELS_DIR / "lgbm_final.pkl").exists():
            _shap_explainer = shap.TreeExplainer(joblib.load(MODELS_DIR / "lgbm_final.pkl"))
        else:
            _shap_explainer = shap.TreeExplainer(best_model)
    return _shap_explainer


def predict_supervised(feat_vec: np.ndarray) -> float:
    if isinstance(best_model, StackingBundle):
        return float(best_model.predict_proba(feat_vec)[0][1])
    model_name = model_metadata.get("best_model", "")
    X = feat_vec
    if model_name in SCALED_MODEL_NAMES and feature_scaler is not None:
        X = feature_scaler.transform(feat_vec)
    return float(best_model.predict_proba(X)[0][1])


def predict_supervised_batch(feat_matrix: np.ndarray) -> np.ndarray:
    if isinstance(best_model, StackingBundle):
        return best_model.predict_proba(feat_matrix)[:, 1].astype(float)
    model_name = model_metadata.get("best_model", "")
    X = feat_matrix
    if model_name in SCALED_MODEL_NAMES and feature_scaler is not None:
        X = feature_scaler.transform(feat_matrix)
    return best_model.predict_proba(X)[:, 1].astype(float)


feature_list_path = REPORTS_DIR / "features" / "selected_feature_list.csv"
if feature_list_path.exists():
    FEATURE_COLS = pd.read_csv(feature_list_path)["feature"].tolist()
else:
    FEATURE_COLS = []
    logger.warning("Feature list file not found. Ensure feature engineering is run first.")


def cache_score_record(account_id: str, record: dict[str, Any]) -> None:
    if len(SCORED_ACCOUNTS_CACHE) >= MAX_CACHE_SIZE:
        first_key = next(iter(SCORED_ACCOUNTS_CACHE))
        SCORED_ACCOUNTS_CACHE.pop(first_key, None)
    SCORED_ACCOUNTS_CACHE[account_id] = record


def get_current_context(request: Request) -> RequestContext:
    principal = getattr(request.state, "principal", None) or auth_manager.public_principal()
    request_id = getattr(request.state, "request_id", None) or auth_manager.request_id(request)
    actor = principal.subject if principal.authenticated else "anonymous"
    return RequestContext(
        request_id=request_id,
        actor=actor,
        role=principal.role,
        authenticated=principal.authenticated,
    )


def restore_runtime_state() -> None:
    global CURRENT_DECISION_MODE, MANUAL_OVERRIDES
    policy = audit_store.load_latest_policy()
    if policy:
        try:
            CURRENT_DECISION_MODE = normalize_decision_mode(policy["mode"])
        except ValueError:
            CURRENT_DECISION_MODE = "balanced"

    MANUAL_OVERRIDES = audit_store.load_overrides()

    for row in audit_store.load_recent_scores(MAX_CACHE_SIZE):
        try:
            features = json.loads(row["features_json"]) if row.get("features_json") else {}
        except Exception:
            features = {}
        try:
            flags = json.loads(row["typology_flags"]) if row.get("typology_flags") else []
        except Exception:
            flags = []

        cache_score_record(
            row["account_id"],
            {
                "features": features,
                "engineered_frame": None,
                "supervised_prob": float(row.get("risk_score") or 0.0),
                "anomaly_score": float(row.get("anomaly_score") or 0.0),
                "fused": float(row.get("fused_risk_score") or 0.0),
                "adjusted_fused": float(row.get("adjusted_fused_risk_score") or 0.0),
                "typology_flags": flags,
                "suspicion_level": int(row.get("suspicion_level") or 1),
                "suspicion_label": row.get("suspicion_label") or "LOW",
                "decision": row.get("decision") or "APPROVE",
                "risk_level": row.get("risk_level") or "LOW",
            },
        )


def build_route_rule(path: str, method: str) -> tuple[Optional[set[str]], Optional[int]]:
    if path == "/health" or path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/openapi.json") or path == "/" or path.startswith("/favicon"):
        return None, None
    if path == "/score/batch":
        return {"investigator", "admin"}, settings.batch_rate_limit_per_minute
    if path == "/score":
        return {"investigator", "admin"}, settings.read_rate_limit_per_minute
    if path.startswith("/explain/") or path == "/alerts/suspicious-list" or path == "/model/info":
        return {"investigator", "admin"}, settings.read_rate_limit_per_minute
    if path == "/decision/policy" or path == "/metrics":
        return {"admin"}, settings.admin_rate_limit_per_minute
    if path == "/decision/override" or path.startswith("/decision/override/"):
        return {"investigator", "admin"}, settings.admin_rate_limit_per_minute
    return None, None


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def make_decision(fused_score: float, thresholds: Optional[Dict[str, float]] = None) -> tuple[str, str]:
    thresholds = thresholds or current_thresholds()
    if fused_score >= thresholds["BLOCK"]:
        return "BLOCK", "CRITICAL"
    if fused_score >= thresholds["CHALLENGE"]:
        return "CHALLENGE", "HIGH"
    if fused_score >= thresholds["REVIEW"]:
        return "REVIEW", "MEDIUM"
    return "APPROVE", "LOW"


def apply_manual_override(account_id: str, decision: str, reason: Optional[str] = None, context: Optional[RequestContext] = None) -> dict:
    normalized_decision = (decision or "").strip().upper()
    allowed = {"BLOCK", "CHALLENGE", "REVIEW", "APPROVE"}
    if normalized_decision not in allowed:
        raise ValueError(f"Unsupported decision override: {decision}")

    override = {
        "decision": normalized_decision,
        "reason": reason,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "actor": context.actor if context else "system",
        "role": context.role if context else "system",
        "request_id": context.request_id if context else None,
    }
    MANUAL_OVERRIDES[account_id] = override
    audit_store.upsert_override({"account_id": account_id, **override})
    return override


def build_feature_vector(features: Dict[str, float]) -> pd.DataFrame:
    return feature_pipeline.transform_from_bank_keys(features)


def build_feature_matrix(feature_dicts: List[Dict[str, float]]) -> pd.DataFrame:
    raw_df = pd.DataFrame(feature_dicts)
    return feature_pipeline.transform(raw_df)


def align_to_model_features(feat_frame: pd.DataFrame) -> np.ndarray:
    if FEATURE_COLS:
        aligned = feat_frame.reindex(columns=FEATURE_COLS, fill_value=0.0)
    else:
        aligned = feat_frame
    return aligned.to_numpy(dtype=np.float32)


def enrich_override_fields(response: dict, override: Optional[dict]) -> tuple[str, str, int, str]:
    decision = response["decision"]
    risk_level = response["risk_level"]
    suspicion_level = response["suspicion_level"]
    suspicion_label = response["suspicion_label"]

    if not override:
        return decision, risk_level, suspicion_level, suspicion_label

    decision = override["decision"]
    if decision == "BLOCK":
        return decision, "CRITICAL", 4, "CRITICAL"
    if decision == "CHALLENGE":
        return decision, "HIGH", 3, "HIGH"
    if decision == "REVIEW":
        return decision, "MEDIUM", 2, "MEDIUM"
    return decision, "LOW", 1, "LOW"


def persist_score_event(response: "ScoreResponse", features: Dict[str, float], context: RequestContext) -> None:
    payload = {
        "account_id": response.account_id,
        "request_id": context.request_id,
        "actor": context.actor,
        "role": context.role,
        "model_version": response.model_version,
        "risk_score": response.risk_score,
        "anomaly_score": response.anomaly_score,
        "fused_risk_score": response.fused_risk_score,
        "adjusted_fused_risk_score": response.adjusted_fused_risk_score,
        "suspicion_level": response.suspicion_level,
        "suspicion_label": response.suspicion_label,
        "typology_flags": response.typology_flags,
        "decision": response.decision,
        "risk_level": response.risk_level,
        "original_decision": response.original_decision,
        "decision_overridden": response.decision_overridden,
        "override_reason": response.override_reason,
        "scored_at": response.scored_at,
        "latency_ms": response.latency_ms,
        "features": features,
    }
    try:
        audit_store.record_score(payload)
    except Exception:
        logger.exception("Failed to persist score event", extra={"event": "score_persist", "account_id": response.account_id})


def score_single(account_id: str, features: Dict[str, float], context: RequestContext) -> "ScoreResponse":
    t0 = time.perf_counter()

    clean_features = {key: float(value) for key, value in features.items()}
    feat_frame = build_feature_vector(clean_features)
    feat_vec = align_to_model_features(feat_frame)

    supervised_prob = predict_supervised(feat_vec)
    feat_scaled = robust_scaler.transform(feat_vec)
    iso_raw = iso_forest.decision_function(feat_scaled)[0]
    anomaly_score = float(np.clip(0.5 - iso_raw, 0.0, 1.0))

    fused = 0.70 * supervised_prob + 0.30 * anomaly_score
    flags = typology_engine.detect(clean_features)
    boost = typology_score_boost(flags)
    adjusted_fused = float(min(1.0, fused + boost))

    thresholds = current_thresholds()
    suspicion_level, suspicion_label = score_to_suspicion(adjusted_fused, thresholds)
    model_decision, model_risk_level = make_decision(adjusted_fused, thresholds)

    decision = model_decision
    risk_level = model_risk_level
    override = MANUAL_OVERRIDES.get(account_id)
    decision_overridden = False
    override_reason = None
    if override:
        decision = override["decision"]
        decision_overridden = True
        override_reason = override.get("reason")
        decision, risk_level, suspicion_level, suspicion_label = enrich_override_fields(
            {
                "decision": decision,
                "risk_level": risk_level,
                "suspicion_level": suspicion_level,
                "suspicion_label": suspicion_label,
            },
            override,
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Scored account",
        extra={
            "event": "score",
            "request_id": context.request_id,
            "account_id": account_id,
            "role": context.role,
            "latency_ms": round(latency_ms, 2),
        },
    )

    response = ScoreResponse(
        account_id=account_id,
        risk_score=round(supervised_prob, 6),
        anomaly_score=round(anomaly_score, 6),
        fused_risk_score=round(fused, 6),
        adjusted_fused_risk_score=round(adjusted_fused, 6),
        suspicion_level=suspicion_level,
        suspicion_label=suspicion_label,
        typology_flags=flags,
        decision=decision,
        risk_level=risk_level,
        original_decision=None if not decision_overridden else model_decision,
        decision_overridden=decision_overridden,
        override_reason=override_reason,
        latency_ms=round(latency_ms, 2),
        model_version=model_metadata.get("best_model", "LightGBM"),
        scored_at=datetime.utcnow().isoformat() + "Z",
    )

    cache_score_record(
        account_id,
        {
            "features": clean_features,
            "engineered_frame": feat_frame,
            "supervised_prob": response.risk_score,
            "anomaly_score": response.anomaly_score,
            "fused": response.fused_risk_score,
            "adjusted_fused": response.adjusted_fused_risk_score,
            "typology_flags": flags,
            "suspicion_level": response.suspicion_level,
            "suspicion_label": response.suspicion_label,
            "decision": response.decision,
            "risk_level": response.risk_level,
        },
    )
    persist_score_event(response, clean_features, context)
    return response


def load_cached_or_persisted_account(account_id: str) -> Optional[dict[str, Any]]:
    cached = SCORED_ACCOUNTS_CACHE.get(account_id)
    if cached:
        return cached

    row = audit_store.load_score(account_id)
    if not row:
        return None

    try:
        features = json.loads(row["features_json"]) if row.get("features_json") else {}
    except Exception:
        features = {}
    try:
        flags = json.loads(row["typology_flags"]) if row.get("typology_flags") else []
    except Exception:
        flags = []

    feat_frame = build_feature_vector(features)
    cached = {
        "features": features,
        "engineered_frame": feat_frame,
        "supervised_prob": float(row.get("risk_score") or 0.0),
        "anomaly_score": float(row.get("anomaly_score") or 0.0),
        "fused": float(row.get("fused_risk_score") or 0.0),
        "adjusted_fused": float(row.get("adjusted_fused_risk_score") or 0.0),
        "typology_flags": flags,
        "suspicion_level": int(row.get("suspicion_level") or 1),
        "suspicion_label": row.get("suspicion_label") or "LOW",
        "decision": row.get("decision") or "APPROVE",
        "risk_level": row.get("risk_level") or "LOW",
    }
    cache_score_record(account_id, cached)
    return cached


def require_role_for_path(path: str, method: str) -> tuple[Optional[set[str]], Optional[int]]:
    if path == "/health" or path == "/" or path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/openapi.json") or path.startswith("/favicon"):
        return None, None
    if path == "/score/batch":
        return {"investigator", "admin"}, settings.batch_rate_limit_per_minute
    if path == "/score":
        return {"investigator", "admin"}, settings.read_rate_limit_per_minute
    if path.startswith("/explain/") or path == "/alerts/suspicious-list" or path == "/model/info":
        return {"investigator", "admin"}, settings.read_rate_limit_per_minute
    if path == "/decision/policy" or path == "/metrics":
        return {"admin"}, settings.admin_rate_limit_per_minute
    if path == "/decision/override" or path.startswith("/decision/override/"):
        return {"investigator", "admin"}, settings.admin_rate_limit_per_minute
    return None, None


class TransactionFeatures(BaseModel):
    account_id: str = Field(..., description="Unique account identifier for tracing.")
    features: Dict[str, float] = Field(
        ...,
        description=(
            "Dictionary of feature_name -> value. Must include the 18 bank-specified features."
        ),
    )

    @validator("features")
    def validate_required_features(cls, features):
        missing = [f for f in BANK_KEY_FEATURES if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return features


class ScoreResponse(BaseModel):
    account_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    fused_risk_score: float = Field(..., ge=0.0, le=1.0)
    adjusted_fused_risk_score: float = Field(..., ge=0.0, le=1.0)
    suspicion_level: int = Field(..., ge=1, le=4)
    suspicion_label: str
    typology_flags: List[str] = Field(default_factory=list)
    decision: str
    risk_level: str
    original_decision: Optional[str] = None
    decision_overridden: bool = False
    override_reason: Optional[str] = None
    latency_ms: float
    model_version: str
    scored_at: str


class BatchScoreRequest(BaseModel):
    accounts: List[TransactionFeatures]


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    total: int
    blocked: int
    challenged: int
    reviewed: int
    approved: int
    batch_latency_ms: float


class DecisionPolicyRequest(BaseModel):
    mode: str = Field(..., description="Decision preset: strict, balanced, loose")


class DecisionPolicyResponse(BaseModel):
    mode: str
    thresholds: Dict[str, float]


class ManualOverrideRequest(BaseModel):
    account_id: str
    decision: str = Field(..., description="BLOCK | CHALLENGE | REVIEW | APPROVE")
    reason: Optional[str] = Field(default=None, description="Investigator notes")


class ManualOverrideResponse(BaseModel):
    account_id: str
    decision: str
    reason: Optional[str]
    updated_at: str


app = FastAPI(title="Mule Account Detection Platform", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*", "Authorization", "X-API-Key", "X-Request-ID"],
    allow_credentials=False,
)


@app.middleware("http")
async def security_and_observability_middleware(request: Request, call_next):
    global RATE_LIMIT_HITS, AUTH_FAILURES, REQUEST_COUNTER

    REQUEST_COUNTER += 1
    request_id = auth_manager.request_id(request)
    request.state.request_id = request_id
    path = request.url.path
    method = request.method.upper()
    required_roles, rate_limit = require_role_for_path(path, method)
    principal = auth_manager.public_principal()

    if request.headers.get("content-length"):
        try:
            if int(request.headers["content-length"]) > settings.max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large."},
                    headers={"X-Request-ID": request_id},
                )
        except ValueError:
            pass

    if required_roles is not None:
        try:
            principal = auth_manager.principal_from_request(request)
            auth_manager.ensure_role(principal, required_roles)
        except HTTPException as exc:
            AUTH_FAILURES += 1
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers={"X-Request-ID": request_id},
            )

    request.state.principal = principal

    if rate_limit is not None and path not in {"/health", "/", "/docs", "/redoc", "/openapi.json"}:
        limit_key = f"{client_ip(request)}:{method}:{path}"
        allowed, remaining = rate_limiter.allow(limit_key, rate_limit)
        if not allowed:
            RATE_LIMIT_HITS += 1
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded."},
                headers={
                    "X-Request-ID": request_id,
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Remaining": str(remaining),
                },
            )

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request error",
            extra={"event": "request_error", "request_id": request_id, "path": path, "method": method},
        )
        raise

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(latency_ms)
    logger.info(
        "Request completed",
        extra={
            "event": "http",
            "request_id": request_id,
            "path": path,
            "method": method,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "role": getattr(principal, "role", "anonymous"),
        },
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled error",
        extra={"event": "exception", "request_id": getattr(request.state, "request_id", None), "path": request.url.path},
    )
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/model/info", tags=["System"])
def model_info(request: Request, include_features: bool = False):
    principal = getattr(request.state, "principal", auth_manager.public_principal())
    feature_columns = FEATURE_COLS if include_features and principal.role == "admin" else []
    return {
        "model_name": model_metadata.get("best_model", "LightGBM"),
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "roc_auc": model_metadata.get("best_roc_auc"),
        "pr_auc": model_metadata.get("best_pr_auc"),
        "precision_optimal_threshold": model_metadata.get("precision_optimal_threshold"),
        "holdout_precision": model_metadata.get("holdout_precision_at_optimal"),
        "holdout_recall": model_metadata.get("holdout_recall_at_optimal"),
        "n_features": len(FEATURE_COLS),
        "feature_columns": feature_columns,
        "decision_policy": {"mode": CURRENT_DECISION_MODE, "thresholds": current_thresholds()},
    }


@app.get("/metrics", tags=["System"])
def metrics_view():
    return {
        **audit_store.load_metrics(),
        "cache_size": len(SCORED_ACCOUNTS_CACHE),
        "request_count": REQUEST_COUNTER,
        "auth_failures": AUTH_FAILURES,
        "rate_limit_hits": RATE_LIMIT_HITS,
        "current_policy": CURRENT_DECISION_MODE,
    }


@app.post("/decision/policy", tags=["System"], response_model=DecisionPolicyResponse)
def update_decision_policy(request: Request, payload: DecisionPolicyRequest):
    global CURRENT_DECISION_MODE
    try:
        CURRENT_DECISION_MODE = normalize_decision_mode(payload.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    context = get_current_context(request)
    try:
        audit_store.record_policy(
            {
                "mode": CURRENT_DECISION_MODE,
                "thresholds": current_thresholds(),
                "actor": context.actor,
                "role": context.role,
                "request_id": context.request_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        )
    except Exception:
        logger.exception("Failed to persist policy change", extra={"event": "policy_persist"})

    logger.info(
        "Decision policy updated",
        extra={"event": "policy", "request_id": context.request_id, "role": context.role, "path": "/decision/policy"},
    )
    return DecisionPolicyResponse(mode=CURRENT_DECISION_MODE, thresholds=current_thresholds())


@app.post("/decision/override", tags=["System"], response_model=ManualOverrideResponse)
def set_manual_override(request: Request, payload: ManualOverrideRequest):
    context = get_current_context(request)
    try:
        override = apply_manual_override(payload.account_id, payload.decision, payload.reason, context)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        "Manual override applied",
        extra={
            "event": "override",
            "request_id": context.request_id,
            "account_id": payload.account_id,
            "role": context.role,
        },
    )
    return ManualOverrideResponse(
        account_id=payload.account_id,
        decision=override["decision"],
        reason=override.get("reason"),
        updated_at=override["updated_at"],
    )


@app.get("/decision/override/{account_id}", tags=["System"])
def get_manual_override(account_id: str):
    override = MANUAL_OVERRIDES.get(account_id)
    if not override:
        raise HTTPException(status_code=404, detail="No manual override found for this account.")
    return {"account_id": account_id, **override}


@app.get("/alerts/suspicious-list", tags=["Alerts"])
def suspicious_list(min_level: int = 2, typology: Optional[str] = None, limit: int = 500):
    min_level = max(1, min(4, min_level))
    limit = max(1, min(limit, MAX_CACHE_SIZE))
    rows = []
    for account_id, cached in SCORED_ACCOUNTS_CACHE.items():
        level = cached.get("suspicion_level", 1)
        flags = cached.get("typology_flags", [])
        if level < min_level:
            continue
        if typology and typology not in flags:
            continue
        rows.append(
            {
                "account_id": account_id,
                "risk_score": cached.get("supervised_prob"),
                "anomaly_score": cached.get("anomaly_score"),
                "fused_risk_score": cached.get("fused"),
                "adjusted_fused_risk_score": cached.get("adjusted_fused"),
                "suspicion_level": level,
                "suspicion_label": cached.get("suspicion_label"),
                "decision": cached.get("decision"),
                "typology_flags": flags,
            }
        )
    rows.sort(key=lambda r: (r["suspicion_level"], r["adjusted_fused_risk_score"]), reverse=True)
    return {"total": len(rows[:limit]), "accounts": rows[:limit]}


@app.get("/explain/{account_id}", tags=["Explain"])
def explain(account_id: str):
    cached = load_cached_or_persisted_account(account_id)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail="Account not found. Score the account via POST /score first.",
        )

    fused = cached.get("adjusted_fused", cached.get("fused", 0.0))
    feat_frame = cached.get("engineered_frame")
    if feat_frame is None:
        feat_frame = build_feature_vector(cached["features"])

    explainer = get_shap_explainer()
    feat_for_shap = feat_frame.reindex(columns=FEATURE_COLS, fill_value=0.0) if FEATURE_COLS else feat_frame
    shap_output = explainer.shap_values(feat_for_shap)

    if isinstance(shap_output, list):
        shap_vals = shap_output[1][0]
        base_value = float(explainer.expected_value[1])
    else:
        shap_vals = shap_output[0]
        ev = explainer.expected_value
        base_value = float(ev[1] if hasattr(ev, "__len__") and len(ev) > 1 else ev)

    feature_names = list(feat_for_shap.columns)
    shap_explanations = [
        {
            "feature": name,
            "value": float(feat_for_shap.iloc[0][name]),
            "shap_value": round(float(shap_vals[i]), 5),
        }
        for i, name in enumerate(feature_names)
    ]
    shap_explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "account_id": account_id,
        "base_value": round(base_value, 5),
        "fused_risk_score": fused,
        "explanations": shap_explanations[:10],
    }


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_account(request: Request, payload: TransactionFeatures):
    try:
        context = get_current_context(request)
        return score_single(payload.account_id, payload.features, context)
    except Exception as e:
        logger.exception(
            "Scoring error",
            extra={"event": "score_error", "request_id": getattr(request.state, "request_id", None), "account_id": payload.account_id},
        )
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
def score_batch(request: Request, payload: BatchScoreRequest):
    if len(payload.accounts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size cannot exceed {MAX_BATCH_SIZE:,} accounts.")

    context = get_current_context(request)
    t0 = time.perf_counter()
    n = len(payload.accounts)

    if n <= 100:
        results = [score_single(acc.account_id, acc.features, context) for acc in payload.accounts]
        batch_latency_ms = (time.perf_counter() - t0) * 1000
        decisions = [r.decision for r in results]
        return BatchScoreResponse(
            results=results,
            total=len(results),
            blocked=decisions.count("BLOCK"),
            challenged=decisions.count("CHALLENGE"),
            reviewed=decisions.count("REVIEW"),
            approved=decisions.count("APPROVE"),
            batch_latency_ms=round(batch_latency_ms, 2),
        )

    feature_dicts = [acc.features for acc in payload.accounts]
    feat_frame = build_feature_matrix(feature_dicts)
    feat_matrix = align_to_model_features(feat_frame)
    supervised_probs = predict_supervised_batch(feat_matrix)
    feat_scaled = robust_scaler.transform(feat_matrix)
    iso_raws = iso_forest.decision_function(feat_scaled)
    anomaly_scores = np.clip(0.5 - iso_raws, 0.0, 1.0).astype(float)
    fused_scores = 0.70 * supervised_probs + 0.30 * anomaly_scores

    results: list[ScoreResponse] = []
    persist_rows: list[dict[str, Any]] = []
    blocked_count = challenged_count = reviewed_count = approved_count = 0
    scored_time = datetime.utcnow().isoformat() + "Z"
    model_version = model_metadata.get("best_model", "LightGBM")

    for i, acc in enumerate(payload.accounts):
        supervised = float(supervised_probs[i])
        anomaly = float(anomaly_scores[i])
        fused = float(fused_scores[i])
        clean_features = {key: float(value) for key, value in acc.features.items()}
        flags = typology_engine.detect(clean_features)
        boost = typology_score_boost(flags)
        adjusted_fused = float(min(1.0, fused + boost))
        thresholds = current_thresholds()
        suspicion_level, suspicion_label = score_to_suspicion(adjusted_fused, thresholds)
        model_decision, model_risk_level = make_decision(adjusted_fused, thresholds)
        decision = model_decision
        risk_level = model_risk_level
        override = MANUAL_OVERRIDES.get(acc.account_id)
        decision_overridden = False
        override_reason = None
        if override:
            decision = override["decision"]
            decision_overridden = True
            override_reason = override.get("reason")
            decision, risk_level, suspicion_level, suspicion_label = enrich_override_fields(
                {
                    "decision": decision,
                    "risk_level": risk_level,
                    "suspicion_level": suspicion_level,
                    "suspicion_label": suspicion_label,
                },
                override,
            )

        if decision == "BLOCK":
            blocked_count += 1
        elif decision == "CHALLENGE":
            challenged_count += 1
        elif decision == "REVIEW":
            reviewed_count += 1
        else:
            approved_count += 1

        response = ScoreResponse(
            account_id=acc.account_id,
            risk_score=round(supervised, 6),
            anomaly_score=round(anomaly, 6),
            fused_risk_score=round(fused, 6),
            adjusted_fused_risk_score=round(adjusted_fused, 6),
            suspicion_level=suspicion_level,
            suspicion_label=suspicion_label,
            typology_flags=flags,
            decision=decision,
            risk_level=risk_level,
            original_decision=None if not decision_overridden else model_decision,
            decision_overridden=decision_overridden,
            override_reason=override_reason,
            latency_ms=0.0,
            model_version=model_version,
            scored_at=scored_time,
        )
        results.append(response)
        cache_score_record(
            acc.account_id,
            {
                "features": clean_features,
                "engineered_frame": feat_frame.iloc[[i]],
                "supervised_prob": supervised,
                "anomaly_score": anomaly,
                "fused": fused,
                "adjusted_fused": adjusted_fused,
                "typology_flags": flags,
                "suspicion_level": suspicion_level,
                "suspicion_label": suspicion_label,
                "decision": decision,
                "risk_level": risk_level,
            },
        )
        persist_rows.append(
            {
                "account_id": acc.account_id,
                "request_id": context.request_id,
                "actor": context.actor,
                "role": context.role,
                "model_version": model_version,
                "risk_score": round(supervised, 6),
                "anomaly_score": round(anomaly, 6),
                "fused_risk_score": round(fused, 6),
                "adjusted_fused_risk_score": round(adjusted_fused, 6),
                "suspicion_level": suspicion_level,
                "suspicion_label": suspicion_label,
                "typology_flags": flags,
                "decision": decision,
                "risk_level": risk_level,
                "original_decision": None if not decision_overridden else model_decision,
                "decision_overridden": decision_overridden,
                "override_reason": override_reason,
                "scored_at": scored_time,
                "latency_ms": 0.0,
                "features": clean_features,
            }
        )

    try:
        audit_store.record_scores(persist_rows)
    except Exception:
        logger.exception("Failed to persist batch scores", extra={"event": "batch_persist", "request_id": context.request_id})

    if len(SCORED_ACCOUNTS_CACHE) > MAX_CACHE_SIZE:
        keys_to_remove = list(SCORED_ACCOUNTS_CACHE.keys())[:-MAX_CACHE_SIZE]
        for key in keys_to_remove:
            SCORED_ACCOUNTS_CACHE.pop(key, None)

    batch_latency_ms = (time.perf_counter() - t0) * 1000
    avg_latency = round(batch_latency_ms / n, 4) if n else 0.0
    for result in results:
        result.latency_ms = avg_latency

    return BatchScoreResponse(
        results=results,
        total=n,
        blocked=blocked_count,
        challenged=challenged_count,
        reviewed=reviewed_count,
        approved=approved_count,
        batch_latency_ms=round(batch_latency_ms, 2),
    )


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.on_event("startup")
def startup_event() -> None:
    restore_runtime_state()
    if auth_manager.demo_mode:
        logger.warning(
            "Demo API keys are active. Configure MULE_API_KEYS and MULE_ADMIN_API_KEYS for production.",
            extra={"event": "startup"},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.app:app", host="0.0.0.0", port=8000, workers=1)
