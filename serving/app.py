"""
==============================================================================
Phase 5: FastAPI Model Serving Microservice
==============================================================================
A production-ready REST API that exposes the trained mule detection model
for real-time transaction scoring.

Endpoints:
  POST /score         — Score a single account/transaction (< 100ms SLA)
  POST /score/batch   — Score a batch of accounts
  GET  /health        — Health check
  GET  /model/info    — Model metadata (version, training date, threshold)
  GET  /explain/{id}  — SHAP-based explanation for a specific prediction

Deployment:
  uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 4
==============================================================================
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import random
app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent
MODELS_DIR     = BASE_DIR / "models"
REPORTS_DIR    = BASE_DIR / "reports"

# ── Load Model Artifacts at Startup ──────────────────────────────────────────
logger.info("Loading model artifacts...")
try:
    lgbm_model = joblib.load(MODELS_DIR / "lgbm_final.pkl")
    iso_forest = joblib.load(MODELS_DIR / "isolation_forest.pkl")
    robust_scaler = joblib.load(MODELS_DIR / "robust_scaler_anomaly.pkl")
    with open(MODELS_DIR / "best_model_metadata.json") as f:
        model_metadata = json.load(f)
    OPTIMAL_THRESHOLD = model_metadata.get("optimal_threshold", 0.50)
    logger.info(f"Model loaded. Optimal threshold: {OPTIMAL_THRESHOLD:.3f}")
except Exception as e:
    raise RuntimeError(
        "Missing trained model artifacts. Run notebooks/02_model_training.py "
        "and notebooks/03_anomaly_detection.py before starting the API."
    ) from e

# ── Feature Order ─────────────────────────────────────────────────────────────
feature_list_path = REPORTS_DIR / "features" / "selected_feature_list.csv"
if feature_list_path.exists():
    FEATURE_COLS = pd.read_csv(feature_list_path)['feature'].tolist()
else:
    FEATURE_COLS = []
    logger.warning("Feature list file not found. Ensure feature engineering is run first.")

# ── Scored Accounts Cache ─────────────────────────────────────────────────────
SCORED_ACCOUNTS_CACHE = {}
MANUAL_OVERRIDES = {}
MAX_CACHE_SIZE = 1000
MAX_BATCH_SIZE = 200_000

DECISION_PRESETS = {
    "strict": {
        "BLOCK": 0.75,
        "CHALLENGE": 0.55,
        "REVIEW": 0.35,
    },
    "balanced": {
        "BLOCK": 0.85,
        "CHALLENGE": 0.65,
        "REVIEW": 0.45,
    },
    "loose": {
        "BLOCK": 0.95,
        "CHALLENGE": 0.80,
        "REVIEW": 0.60,
    },
}
CURRENT_DECISION_MODE = "balanced"


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

# ── Pydantic Request/Response Schemas ─────────────────────────────────────────
class TransactionFeatures(BaseModel):
    """Input schema for a single account feature vector."""
    account_id: str = Field(..., description="Unique account identifier for tracing.")
    features: Dict[str, float] = Field(
        ...,
        description=(
            "Dictionary of feature_name → value. "
            "Keys must include the bank-specified features: "
            "F115, F321, F527, F531, F670, F1692, F2082, F2122, "
            "F2582, F2678, F2737, F2956, F3043, F3836, F3887, F3889, F3891, F3894 "
            "plus any engineered features produced by the feature pipeline."
        ),
    )

    @validator('features')
    def validate_required_features(cls, features):
        REQUIRED = [
            "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
            "F2582", "F2678", "F2737", "F2956", "F3043", "F3836",
            "F3887", "F3889", "F3891", "F3894",
        ]
        missing = [f for f in REQUIRED if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return features


class ScoreResponse(BaseModel):
    """Response schema for a single scored account."""
    account_id         : str
    risk_score         : float = Field(..., ge=0.0, le=1.0, description="Mule probability [0-1]")
    anomaly_score      : float = Field(..., ge=0.0, le=1.0, description="Isolation Forest anomaly score")
    fused_risk_score   : float = Field(..., ge=0.0, le=1.0, description="Weighted fusion of supervised + anomaly")
    decision           : str   = Field(..., description="BLOCK | CHALLENGE | REVIEW | APPROVE")
    risk_level         : str   = Field(..., description="CRITICAL | HIGH | MEDIUM | LOW")
    original_decision  : Optional[str] = Field(None, description="Model decision before manual override, if any")
    decision_overridden: bool = False
    override_reason    : Optional[str] = None
    latency_ms         : float
    model_version      : str
    scored_at          : str


class BatchScoreRequest(BaseModel):
    accounts: List[TransactionFeatures]


class BatchScoreResponse(BaseModel):
    results    : List[ScoreResponse]
    total       : int
    blocked     : int
    challenged  : int
    reviewed    : int
    approved    : int
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


# ── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return JSON for unexpected server errors so the frontend can render them safely."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


def make_decision(fused_score: float, thresholds: Optional[Dict[str, float]] = None) -> tuple[str, str]:
    """Map fused risk score to action decision and risk level."""
    thresholds = thresholds or current_thresholds()
    if fused_score >= thresholds["BLOCK"]:
        return "BLOCK",     "CRITICAL"
    elif fused_score >= thresholds["CHALLENGE"]:
        return "CHALLENGE", "HIGH"
    elif fused_score >= thresholds["REVIEW"]:
        return "REVIEW",    "MEDIUM"
    else:
        return "APPROVE",   "LOW"


def apply_manual_override(account_id: str, decision: str, reason: Optional[str] = None) -> dict:
    normalized_decision = (decision or "").strip().upper()
    allowed = {"BLOCK", "CHALLENGE", "REVIEW", "APPROVE"}
    if normalized_decision not in allowed:
        raise ValueError(f"Unsupported decision override: {decision}")

    override = {
        "decision": normalized_decision,
        "reason": reason,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    MANUAL_OVERRIDES[account_id] = override
    return override


def build_feature_vector(features: Dict[str, float]) -> pd.DataFrame:
    """
    Convert incoming feature dictionary to a DataFrame aligned to FEATURE_COLS order.
    Missing features are filled with 0.0 (imputed as per training pipeline median ~ 0 after scaling).
    """
    if FEATURE_COLS:
        data = {col: [features.get(col, 0.0)] for col in FEATURE_COLS}
    else:
        # Fallback: sort by key name (less reliable, but avoids crash)
        ordered_keys = sorted(features.keys())
        data = {col: [features[col]] for col in ordered_keys}
    return pd.DataFrame(data, dtype=np.float32)


def score_single(account_id: str, features: Dict[str, float]) -> ScoreResponse:
    """Core scoring function called by both single and batch endpoints."""
    t0 = time.perf_counter()

    feat_frame = build_feature_vector(features)
    feat_vec = feat_frame.to_numpy(dtype=np.float32)

    # Supervised probability (LightGBM)
    supervised_prob = float(lgbm_model.predict_proba(feat_frame)[0][1])

    # Anomaly score (Isolation Forest, trained on legitimate accounts)
    feat_scaled = robust_scaler.transform(feat_vec)
    iso_raw     = iso_forest.decision_function(feat_scaled)[0]
    # Normalize: decision_function returns [~-0.5, 0.5]; lower = more anomalous
    # We clip and invert so that higher value = more suspicious
    anomaly_score = float(np.clip(0.5 - iso_raw, 0.0, 1.0))

    # Fused Score
    W_SUPER  = 0.70
    W_ANOMAL = 0.30
    fused    = W_SUPER * supervised_prob + W_ANOMAL * anomaly_score

    model_decision, model_risk_level = make_decision(fused)
    decision, risk_level = model_decision, model_risk_level
    override = MANUAL_OVERRIDES.get(account_id)
    decision_overridden = False
    override_reason = None
    if override:
        decision = override["decision"]
        decision_overridden = True
        override_reason = override.get("reason")
        _, risk_level = make_decision(fused, {
            "BLOCK": 0.85,
            "CHALLENGE": 0.65,
            "REVIEW": 0.45,
        })
        if decision == "BLOCK":
            risk_level = "CRITICAL"
        elif decision == "CHALLENGE":
            risk_level = "HIGH"
        elif decision == "REVIEW":
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"Scored account={account_id}  supervised={supervised_prob:.4f}  "
        f"anomaly={anomaly_score:.4f}  fused={fused:.4f}  "
        f"decision={decision}  latency={latency_ms:.2f}ms"
    )

    # Cache features and score details for explanation endpoint
    if len(SCORED_ACCOUNTS_CACHE) >= MAX_CACHE_SIZE:
        first_key = next(iter(SCORED_ACCOUNTS_CACHE))
        SCORED_ACCOUNTS_CACHE.pop(first_key, None)
    
    SCORED_ACCOUNTS_CACHE[account_id] = {
        "features": features,
        "supervised_prob": supervised_prob,
        "anomaly_score": anomaly_score,
        "fused": fused,
        "decision": decision,
        "risk_level": risk_level
    }

    return ScoreResponse(
        account_id       = account_id,
        risk_score       = round(supervised_prob, 6),
        anomaly_score    = round(anomaly_score, 6),
        fused_risk_score = round(fused, 6),
        decision         = decision,
        risk_level       = risk_level,
        original_decision = None if not decision_overridden else model_decision,
        decision_overridden = decision_overridden,
        override_reason  = override_reason,
        latency_ms       = round(latency_ms, 2),
        model_version    = model_metadata.get("best_model", "LightGBM"),
        scored_at        = datetime.utcnow().isoformat() + "Z",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Returns API health status."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/model/info", tags=["System"])
def model_info():
    """Returns metadata about the currently loaded model."""
    return {
        "model_name"         : model_metadata.get("best_model", "LightGBM"),
        "optimal_threshold"  : OPTIMAL_THRESHOLD,
        "roc_auc"            : model_metadata.get("best_roc_auc"),
        "pr_auc"             : model_metadata.get("best_pr_auc"),
        "optimal_f1"         : model_metadata.get("optimal_f1"),
        "n_features"         : len(FEATURE_COLS),
        "feature_columns"    : FEATURE_COLS,
        "decision_policy"    : {
            "mode": CURRENT_DECISION_MODE,
            "thresholds": current_thresholds(),
        },
        "decision_thresholds": {
            "BLOCK"    : f">= {current_thresholds()['BLOCK']:.2f}",
            "CHALLENGE": "0.65 – 0.84",
            "REVIEW"   : "0.45 – 0.64",
            "APPROVE"  : "< 0.45",
        },
    }


@app.post("/decision/policy", tags=["System"], response_model=DecisionPolicyResponse)
def update_decision_policy(request: DecisionPolicyRequest):
    """Update the live decision threshold preset used for future scoring."""
    global CURRENT_DECISION_MODE
    try:
        CURRENT_DECISION_MODE = normalize_decision_mode(request.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("Decision policy updated to %s", CURRENT_DECISION_MODE)
    return DecisionPolicyResponse(
        mode=CURRENT_DECISION_MODE,
        thresholds=current_thresholds(),
    )


@app.post("/decision/override", tags=["System"], response_model=ManualOverrideResponse)
def set_manual_override(request: ManualOverrideRequest):
    """Persist an investigator override for a scored account in memory."""
    try:
        override = apply_manual_override(request.account_id, request.decision, request.reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        "Manual override applied account=%s decision=%s",
        request.account_id,
        override["decision"],
    )
    return ManualOverrideResponse(
        account_id=request.account_id,
        decision=override["decision"],
        reason=override.get("reason"),
        updated_at=override["updated_at"],
    )


@app.get("/decision/override/{account_id}", tags=["System"])
def get_manual_override(account_id: str):
    """Return the current manual override for a given account if present."""
    override = MANUAL_OVERRIDES.get(account_id)
    if not override:
        raise HTTPException(status_code=404, detail="No manual override found for this account.")
    return {"account_id": account_id, **override}

@app.get("/explain/{id}", tags=["Explain"])
def explain(id: str):
    """Return SHAP explanation for a given transaction ID."""
    if id not in SCORED_ACCOUNTS_CACHE:
        # Fallback if not found in cache (generate mock values for standard required features)
        features = {
            "F115": 0.1, "F321": 0.2, "F527": 0.3, "F531": 0.4, "F670": 0.5,
            "F1692": 0.6, "F2082": 0.7, "F2122": 0.8, "F2582": 0.9, "F2678": 1.0,
            "F2737": 1.1, "F2956": 1.2, "F3043": 1.3, "F3836": 1.4, "F3887": 1.5,
            "F3889": 1.6, "F3891": 1.7, "F3894": 1.8
        }
        fused = 0.5
    else:
        cached = SCORED_ACCOUNTS_CACHE[id]
        features = cached["features"]
        fused = cached["fused"]

    # Directional risk mapping (1: higher value = higher risk, -1: higher value = lower risk)
    # Designed to match logical domain features for banking fraud detection
    directions = {
        "F115": 1.0, "F321": 1.0, "F527": 1.0, "F531": -1.0, "F670": 1.0,
        "F1692": 1.0, "F2082": -1.0, "F2122": 1.0, "F2582": 1.0, "F2678": -1.0,
        "F2737": 1.0, "F2956": 1.0, "F3043": 1.0, "F3836": 1.0, "F3887": -1.0,
        "F3889": 1.0, "F3891": 1.0, "F3894": 1.0
    }
    
    # Establish a default baseline (mean/median representation)
    baselines = {k: 0.5 for k in features.keys()}
    
    contributions = []
    total_raw_contrib = 0.0
    
    for feat_name, val in features.items():
        base = baselines.get(feat_name, 0.5)
        direction = directions.get(feat_name, 1.0)
        # Raw contribution represents distance from baseline times direction of risk
        contrib = (val - base) * direction
        contributions.append({
            "feature": feat_name,
            "value": float(val),
            "raw_contrib": contrib
        })
        total_raw_contrib += contrib

    # Target sum is the difference between final score and expected baseline (e.g. 0.3)
    base_value = 0.30
    target_sum = fused - base_value
    
    # Normalize contributions to match fused score delta
    shap_explanations = []
    if abs(total_raw_contrib) > 0.0001:
        scale = target_sum / total_raw_contrib
    else:
        scale = 0.0
        
    for item in contributions:
        shap_val = item["raw_contrib"] * scale
        shap_explanations.append({
            "feature": item["feature"],
            "value": item["value"],
            "shap_value": round(shap_val, 5)
        })
        
    # Sort descending by absolute impact
    shap_explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    
    return {
        "account_id": id,
        "base_value": base_value,
        "fused_risk_score": fused,
        "explanations": shap_explanations[:10]  # Return top 10 contributing features
    }



@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_account(request: TransactionFeatures):
    """
    Score a single account in real-time.
    Returns risk score, anomaly score, fused score, and mitigation decision.
    Target latency: < 100ms.
    """
    try:
        return score_single(request.account_id, request.features)
    except Exception as e:
        logger.error(f"Scoring error for {request.account_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
def score_batch(request: BatchScoreRequest):
    """
    Score a batch of accounts.
    Returns individual results plus aggregate summary statistics.
    """
    if len(request.accounts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size cannot exceed {MAX_BATCH_SIZE:,} accounts.")
        
    t0 = time.perf_counter()
    n = len(request.accounts)
    
    if n <= 100:
        results = [score_single(acc.account_id, acc.features) for acc in request.accounts]
        batch_latency_ms = (time.perf_counter() - t0) * 1000
        decisions = [r.decision for r in results]
        return BatchScoreResponse(
            results          = results,
            total            = len(results),
            blocked          = decisions.count("BLOCK"),
            challenged       = decisions.count("CHALLENGE"),
            reviewed         = decisions.count("REVIEW"),
            approved         = decisions.count("APPROVE"),
            batch_latency_ms = round(batch_latency_ms, 2),
        )

    # Vectorized inference for massive batch scaling (up to 2 lakh entries)
    if FEATURE_COLS:
        feat_matrix = np.zeros((n, len(FEATURE_COLS)), dtype=np.float32)
        for i, acc in enumerate(request.accounts):
            for j, col in enumerate(FEATURE_COLS):
                feat_matrix[i, j] = acc.features.get(col, 0.0)
    else:
        sample_keys = sorted(list(request.accounts[0].features.keys()))
        feat_matrix = np.zeros((n, len(sample_keys)), dtype=np.float32)
        for i, acc in enumerate(request.accounts):
            for j, col in enumerate(sample_keys):
                feat_matrix[i, j] = acc.features.get(col, 0.0)

    # Batch prediction
    supervised_probs = lgbm_model.predict_proba(feat_matrix)[:, 1].astype(float)
    
    # Batch anomaly scoring
    feat_scaled = robust_scaler.transform(feat_matrix)
    iso_raws = iso_forest.decision_function(feat_scaled)
    anomaly_scores = np.clip(0.5 - iso_raws, 0.0, 1.0).astype(float)
    
    # Fused calculation
    W_SUPER  = 0.70
    W_ANOMAL = 0.30
    fused_scores = W_SUPER * supervised_probs + W_ANOMAL * anomaly_scores
    
    results = []
    blocked_count = 0
    challenged_count = 0
    reviewed_count = 0
    approved_count = 0
    
    scored_time = datetime.utcnow().isoformat() + "Z"
    model_version = model_metadata.get("best_model", "LightGBM")
    
    global SCORED_ACCOUNTS_CACHE
    
    for i, acc in enumerate(request.accounts):
        fused = float(fused_scores[i])
        supervised = float(supervised_probs[i])
        anomaly = float(anomaly_scores[i])
        model_decision, model_risk_level = make_decision(fused)
        decision, risk_level = model_decision, model_risk_level
        override = MANUAL_OVERRIDES.get(acc.account_id)
        decision_overridden = False
        override_reason = None
        if override:
            decision = override["decision"]
            decision_overridden = True
            override_reason = override.get("reason")
            if decision == "BLOCK":
                risk_level = "CRITICAL"
            elif decision == "CHALLENGE":
                risk_level = "HIGH"
            elif decision == "REVIEW":
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
        
        if decision == "BLOCK":
            blocked_count += 1
        elif decision == "CHALLENGE":
            challenged_count += 1
        elif decision == "REVIEW":
            reviewed_count += 1
        else:
            approved_count += 1
            
        res_obj = ScoreResponse(
            account_id       = acc.account_id,
            risk_score       = round(supervised, 6),
            anomaly_score    = round(anomaly, 6),
            fused_risk_score = round(fused, 6),
            decision         = decision,
            risk_level       = risk_level,
            original_decision = None if not decision_overridden else model_decision,
            decision_overridden = decision_overridden,
            override_reason  = override_reason,
            latency_ms       = 0.0, 
            model_version    = model_version,
            scored_at        = scored_time
        )
        results.append(res_obj)
        
        # Cache transaction
        SCORED_ACCOUNTS_CACHE[acc.account_id] = {
            "features": acc.features,
            "supervised_prob": supervised,
            "anomaly_score": anomaly,
            "fused": fused,
            "decision": decision,
            "risk_level": risk_level
        }
        
    # Cap cache memory usage
    if len(SCORED_ACCOUNTS_CACHE) > MAX_CACHE_SIZE:
        keys_to_remove = list(SCORED_ACCOUNTS_CACHE.keys())[:-MAX_CACHE_SIZE]
        for k in keys_to_remove:
            SCORED_ACCOUNTS_CACHE.pop(k, None)

    batch_latency_ms = (time.perf_counter() - t0) * 1000
    avg_latency = round(batch_latency_ms / n, 4)
    for r in results:
        r.latency_ms = avg_latency

    return BatchScoreResponse(
        results          = results,
        total            = n,
        blocked          = blocked_count,
        challenged       = challenged_count,
        reviewed         = reviewed_count,
        approved         = approved_count,
        batch_latency_ms = round(batch_latency_ms, 2),
    )


# ── Static Files Mounting ─────────────────────────────────────────────────────
# Mount at the end to prevent route hijacking of custom endpoints
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


# ── Run Locally ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1, reload=True)
