"""Integration tests for FastAPI serving (requires trained artifacts)."""
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
LGBM_PATH = MODELS_DIR / "lgbm_final.pkl"

BANK_KEY_SAMPLE = {
    "F115": 0.55, "F321": 1.1, "F527": 1.0, "F531": 1.2, "F670": 0.0,
    "F1692": 0.0, "F2082": 0.0, "F2122": 0.01, "F2582": 0.0,
    "F2678": -0.1, "F2737": -0.15, "F2956": 64.0, "F3043": 87.0,
    "F3836": 382529.0, "F3887": 94.0, "F3889": 99.0, "F3891": 95.0, "F3894": 34.0,
}

HAS_MODEL = PIPELINE_PATH.exists() and (BEST_MODEL_PATH.exists() or LGBM_PATH.exists())

pytestmark = pytest.mark.skipif(
    not HAS_MODEL,
    reason="Trained model artifacts not found; run the notebook pipeline first.",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from serving.app import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_score_with_bank_keys_only(client):
    payload = {"account_id": "pytest-single", "features": BANK_KEY_SAMPLE}
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "fused_risk_score" in data
    assert 0.0 <= data["fused_risk_score"] <= 1.0
    assert data["decision"] in {"BLOCK", "CHALLENGE", "REVIEW", "APPROVE"}
    assert "suspicion_level" in data
    assert 1 <= data["suspicion_level"] <= 4
    assert data["suspicion_label"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert isinstance(data["typology_flags"], list)
    assert "adjusted_fused_risk_score" in data
    assert 0.0 <= data["adjusted_fused_risk_score"] <= 1.0


def test_suspicious_list_after_score(client):
    account_id = "pytest-suspicious-list"
    score_resp = client.post(
        "/score",
        json={"account_id": account_id, "features": BANK_KEY_SAMPLE},
    )
    assert score_resp.status_code == 200

    list_resp = client.get("/alerts/suspicious-list", params={"min_level": 1, "limit": 50})
    assert list_resp.status_code == 200
    body = list_resp.json()
    assert "accounts" in body
    ids = [row["account_id"] for row in body["accounts"]]
    assert account_id in ids
    row = next(r for r in body["accounts"] if r["account_id"] == account_id)
    assert "suspicion_level" in row
    assert "typology_flags" in row


def test_explain_returns_real_shap(client):
    account_id = "pytest-explain"
    score_resp = client.post(
        "/score",
        json={"account_id": account_id, "features": BANK_KEY_SAMPLE},
    )
    assert score_resp.status_code == 200

    explain_resp = client.get(f"/explain/{account_id}")
    assert explain_resp.status_code == 200
    body = explain_resp.json()
    assert body["account_id"] == account_id
    assert "base_value" in body
    assert len(body["explanations"]) <= 10
    assert all("shap_value" in e for e in body["explanations"])


def test_explain_404_without_prior_score(client):
    resp = client.get("/explain/nonexistent-account-id")
    assert resp.status_code == 404


def test_score_rejects_missing_bank_keys(client):
    bad = dict(BANK_KEY_SAMPLE)
    del bad["F115"]
    resp = client.post("/score", json={"account_id": "bad", "features": bad})
    assert resp.status_code == 422
