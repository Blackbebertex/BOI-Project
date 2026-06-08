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

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent
MODELS_DIR     = BASE_DIR / "models"
REPORTS_DIR    = BASE_DIR / "reports"

# ── Load Model Artifacts at Startup ──────────────────────────────────────────
logger.info("Loading model artifacts...")
lgbm_model     = joblib.load(MODELS_DIR / "lgbm_final.pkl")
iso_forest     = joblib.load(MODELS_DIR / "isolation_forest.pkl")
robust_scaler  = joblib.load(MODELS_DIR / "robust_scaler_anomaly.pkl")

with open(MODELS_DIR / "best_model_metadata.json") as f:
    model_metadata = json.load(f)

OPTIMAL_THRESHOLD = model_metadata.get("optimal_threshold", 0.50)
logger.info(f"Model loaded. Optimal threshold: {OPTIMAL_THRESHOLD:.3f}")

# ── Feature Order ─────────────────────────────────────────────────────────────
feature_list_path = REPORTS_DIR / "features" / "selected_feature_list.csv"
if feature_list_path.exists():
    FEATURE_COLS = pd.read_csv(feature_list_path)['feature'].tolist()
else:
    FEATURE_COLS = []
    logger.warning("Feature list file not found. Ensure feature engineering is run first.")

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


# ── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "Mule Account Detection API",
    description = (
        "Real-time AI/ML scoring service for suspicious transaction "
        "and mule account detection. Powered by LightGBM + Isolation Forest ensemble."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)


def make_decision(fused_score: float) -> tuple[str, str]:
    """Map fused risk score to action decision and risk level."""
    if fused_score >= 0.85:
        return "BLOCK",     "CRITICAL"
    elif fused_score >= 0.65:
        return "CHALLENGE", "HIGH"
    elif fused_score >= 0.45:
        return "REVIEW",    "MEDIUM"
    else:
        return "APPROVE",   "LOW"


def build_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert incoming feature dictionary to a numpy vector aligned to FEATURE_COLS order.
    Missing features are filled with 0.0 (imputed as per training pipeline median ~ 0 after scaling).
    """
    if FEATURE_COLS:
        vec = np.array([features.get(col, 0.0) for col in FEATURE_COLS], dtype=np.float32)
    else:
        # Fallback: sort by key name (less reliable, but avoids crash)
        vec = np.array(list(features.values()), dtype=np.float32)
    return vec.reshape(1, -1)


def score_single(account_id: str, features: Dict[str, float]) -> ScoreResponse:
    """Core scoring function called by both single and batch endpoints."""
    t0 = time.perf_counter()

    feat_vec = build_feature_vector(features)

    # Supervised probability (LightGBM)
    supervised_prob = float(lgbm_model.predict_proba(feat_vec)[0][1])

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

    decision, risk_level = make_decision(fused)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"Scored account={account_id}  supervised={supervised_prob:.4f}  "
        f"anomaly={anomaly_score:.4f}  fused={fused:.4f}  "
        f"decision={decision}  latency={latency_ms:.2f}ms"
    )

    return ScoreResponse(
        account_id       = account_id,
        risk_score       = round(supervised_prob, 6),
        anomaly_score    = round(anomaly_score, 6),
        fused_risk_score = round(fused, 6),
        decision         = decision,
        risk_level       = risk_level,
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
        "decision_thresholds": {
            "BLOCK"    : ">= 0.85",
            "CHALLENGE": "0.65 – 0.84",
            "REVIEW"   : "0.45 – 0.64",
            "APPROVE"  : "< 0.45",
        },
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
    if len(request.accounts) > 5000:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 5000 accounts.")
    t0 = time.perf_counter()
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


# ── Run Locally ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1, reload=True)
