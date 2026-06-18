"""Unit tests for typology rule engine."""
import pandas as pd
import pytest

from serving.typology_engine import (
    TypologyEngine,
    fit_typology_thresholds,
    score_to_suspicion,
    typology_score_boost,
)


@pytest.fixture
def fitted_engine():
    rng = pd.Series(range(100))
    X = pd.DataFrame({
        "F115": rng / 100.0,
        "F527": (rng % 50) / 50.0,
        "F531": (rng % 30) / 30.0,
        "F2678": (rng - 50) / 100.0,
        "F2737": (rng - 40) / 100.0,
        "F2082": (rng % 20) / 20.0,
        "F2122": (rng % 10 + 1) / 20.0,
    })
    thresholds = fit_typology_thresholds(X)
    return TypologyEngine(thresholds)


def test_silent_account_low_activity_burst(fitted_engine):
    features = {
        "F115": 0.1,
        "F527": 0.99,
        "F531": 0.01,
        "F2678": 0.95,
        "F2737": 0.0,
        "F2082": 0.0,
        "F2122": 0.01,
    }
    flags = fitted_engine.detect(features)
    assert "silent_account" in flags


def test_large_amount_mover_high_flow(fitted_engine):
    features = {
        "F115": 0.5,
        "F527": 0.2,
        "F531": 0.5,
        "F2678": 50.0,
        "F2737": 50.0,
        "F2082": 0.1,
        "F2122": 0.1,
    }
    flags = fitted_engine.detect(features)
    assert "large_amount_mover" in flags


def test_instant_mule_out_in_ratio(fitted_engine):
    features = {
        "F115": 0.5,
        "F527": 0.95,
        "F531": 0.01,
        "F2678": 0.1,
        "F2737": 0.1,
        "F2082": 0.0,
        "F2122": 0.01,
    }
    flags = fitted_engine.detect(features)
    assert "instant_mule" in flags


def test_typology_score_boost_capped():
    assert typology_score_boost([]) == 0.0
    assert typology_score_boost(["silent_account"]) == 0.05
    assert typology_score_boost(["a", "b", "c"]) == 0.10


def test_score_to_suspicion_levels():
    thresholds = {"REVIEW": 0.45, "CHALLENGE": 0.65, "BLOCK": 0.85}
    assert score_to_suspicion(0.2, thresholds) == (1, "LOW")
    assert score_to_suspicion(0.5, thresholds) == (2, "MEDIUM")
    assert score_to_suspicion(0.7, thresholds) == (3, "HIGH")
    assert score_to_suspicion(0.9, thresholds) == (4, "CRITICAL")
