"""Unit tests for label leakage audit."""
import numpy as np
import pandas as pd

from serving.feature_pipeline import LEAKAGE_CORR_THRESHOLD, audit_leakage


def test_audit_leakage_detects_perfect_correlation():
    rng = np.random.default_rng(42)
    n = 200
    y = pd.Series(rng.integers(0, 2, size=n))
    X = pd.DataFrame({
        "clean": rng.normal(size=n),
        "leaky": y.astype(float) + rng.normal(scale=0.01, size=n),
    })
    report = audit_leakage(X, y, threshold=0.5)
    assert "leaky" in report["feature"].values
    assert "clean" not in report["feature"].values


def test_audit_leakage_empty_when_no_leakage():
    rng = np.random.default_rng(0)
    n = 300
    y = pd.Series(rng.integers(0, 2, size=n))
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    report = audit_leakage(X, y, threshold=LEAKAGE_CORR_THRESHOLD)
    assert len(report) == 0


def test_f3912_would_be_flagged_on_synthetic_proxy():
    """Synthetic stand-in: feature mirroring target should exceed threshold."""
    y = pd.Series([0, 0, 0, 1, 1, 1])
    X = pd.DataFrame({"F3912": [0.0, 0.0, 0.01, 0.99, 1.0, 1.0]})
    report = audit_leakage(X, y, threshold=0.5)
    assert len(report) == 1
    assert report.iloc[0]["feature"] == "F3912"
