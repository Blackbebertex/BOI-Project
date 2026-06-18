"""Smoke tests for model zoo artifacts after training."""
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

EXPECTED_ARTIFACTS = [
    "feature_pipeline.pkl",
    "best_model.pkl",
    "best_model_metadata.json",
    "typology_thresholds.json",
    "lgbm_final.pkl",
    "xgb_final.pkl",
    "rf_final.pkl",
    "extratrees_final.pkl",
    "decision_tree_final.pkl",
    "lr_final.pkl",
    "mlp_final.pkl",
    "stacking_meta_learner.pkl",
    "stacking_bundle.pkl",
    "isolation_forest.pkl",
]

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "best_model.pkl").exists(),
    reason="Model zoo not trained; run notebooks/01 and 02 first.",
)


@pytest.mark.parametrize("filename", EXPECTED_ARTIFACTS)
def test_model_zoo_artifact_exists(filename):
    path = MODELS_DIR / filename
    assert path.exists(), f"Missing expected artifact: {filename}"


def test_model_comparison_report_exists():
    report = BASE_DIR / "reports" / "models" / "model_comparison.csv"
    assert report.exists()
