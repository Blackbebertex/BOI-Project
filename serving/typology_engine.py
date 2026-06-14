"""
Rule-based fraud typology detection on bank key features.

Thresholds are fitted on the training split and persisted to
models/typology_thresholds.json for identical behaviour at serving time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from serving.feature_pipeline import BANK_KEY_FEATURES, safe_ratio

TYPOLOGY_THRESHOLD_PATH = Path(__file__).resolve().parent.parent / "models" / "typology_thresholds.json"

PERCENTILE_KEYS = [
    "F115_p75", "F527_p75", "F531_p25", "F531_p90",
    "F2678_p90", "F2678_p95", "F2737_p95",
    "flow_sum_p95", "net_flow_abs_p95", "burst_ratio_p90",
]

BOOST_PER_FLAG = 0.05
MAX_TYPOLOGY_BOOST = 0.10


def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        if np.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _percentile(series: pd.Series, q: float) -> float:
    clean = series.dropna()
    if len(clean) == 0:
        return 0.0
    return float(np.percentile(clean, q))


def fit_typology_thresholds(X_train: pd.DataFrame) -> Dict[str, float]:
    """Compute percentile cutoffs from raw train features (bank keys minimum)."""
    cols = [c for c in BANK_KEY_FEATURES if c in X_train.columns]
    df = X_train[cols].apply(pd.to_numeric, errors="coerce")

    thresholds: Dict[str, float] = {}
    if "F115" in df.columns:
        thresholds["F115_p75"] = _percentile(df["F115"], 75)
    if "F527" in df.columns:
        thresholds["F527_p75"] = _percentile(df["F527"], 75)
    if "F531" in df.columns:
        thresholds["F531_p25"] = _percentile(df["F531"], 25)
        thresholds["F531_p90"] = _percentile(df["F531"], 90)
    if "F2678" in df.columns:
        thresholds["F2678_p90"] = _percentile(df["F2678"].abs(), 90)
        thresholds["F2678_p95"] = _percentile(df["F2678"].abs(), 95)
    if "F2737" in df.columns:
        thresholds["F2737_p95"] = _percentile(df["F2737"].abs(), 95)

    if "F2678" in df.columns and "F2737" in df.columns:
        flow_sum = df["F2678"].abs() + df["F2737"].abs()
        thresholds["flow_sum_p95"] = _percentile(flow_sum, 95)
        net_flow = (df["F2678"] - df["F2737"]).abs()
        thresholds["net_flow_abs_p95"] = _percentile(net_flow, 95)

    if "F2082" in df.columns and "F2122" in df.columns:
        burst = safe_ratio(df["F2082"], df["F2122"])
        thresholds["burst_ratio_p90"] = _percentile(burst, 90)

    return thresholds


def save_typology_thresholds(thresholds: Dict[str, float], path: Path | str | None = None) -> Path:
    path = Path(path or TYPOLOGY_THRESHOLD_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    return path


def load_typology_thresholds(path: Path | str | None = None) -> Dict[str, float]:
    path = Path(path or TYPOLOGY_THRESHOLD_PATH)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TypologyEngine:
    """Detect behavioural typology flags from bank key features."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or load_typology_thresholds()

    @classmethod
    def fit_from_train(cls, X_train: pd.DataFrame, save_path: Path | str | None = None) -> "TypologyEngine":
        thresholds = fit_typology_thresholds(X_train)
        save_typology_thresholds(thresholds, save_path)
        return cls(thresholds)

    def _t(self, key: str, default: float = 0.0) -> float:
        return float(self.thresholds.get(key, default))

    def _derived(self, features: dict) -> dict:
        f527 = _safe_float(features.get("F527"))
        f531 = _safe_float(features.get("F531"))
        f2678 = _safe_float(features.get("F2678"))
        f2737 = _safe_float(features.get("F2737"))
        f2082 = _safe_float(features.get("F2082"))
        f2122 = _safe_float(features.get("F2122"))

        out_in = f527 / (abs(f531) + 1e-9)
        burst = f2082 / (abs(f2122) + 1e-9)
        net_flow = f2678 - f2737
        flow_sum = abs(f2678) + abs(f2737)
        return {
            "out_in_ratio": out_in,
            "burst_ratio": burst,
            "net_flow": net_flow,
            "flow_sum": flow_sum,
            "net_flow_abs": abs(net_flow),
        }

    def detect(self, features: dict) -> List[str]:
        flags: List[str] = []
        d = self._derived(features)

        f115 = _safe_float(features.get("F115"))
        f527 = _safe_float(features.get("F527"))
        f531 = _safe_float(features.get("F531"))
        f2678 = _safe_float(features.get("F2678"))

        silent_a = f531 < self._t("F531_p25") and (
            f527 > self._t("F527_p75") or abs(f2678) > self._t("F2678_p90")
        )
        silent_b = f115 > self._t("F115_p75") and d["burst_ratio"] > self._t("burst_ratio_p90")
        if silent_a or silent_b:
            flags.append("silent_account")

        if d["flow_sum"] > self._t("flow_sum_p95") or d["net_flow_abs"] > self._t("net_flow_abs_p95"):
            flags.append("large_amount_mover")

        if d["out_in_ratio"] > 0.9:
            flags.append("instant_mule")

        if f531 > self._t("F531_p90") and d["out_in_ratio"] < 0.3:
            flags.append("aggregator_hub")

        return flags


def typology_score_boost(flags: List[str]) -> float:
    if not flags:
        return 0.0
    return min(MAX_TYPOLOGY_BOOST, len(flags) * BOOST_PER_FLAG)


def score_to_suspicion(
    fused_score: float,
    thresholds: Dict[str, float],
) -> tuple[int, str]:
    """Map fused score to suspicion level 1-4 and label."""
    block = thresholds.get("BLOCK", 0.85)
    challenge = thresholds.get("CHALLENGE", 0.65)
    review = thresholds.get("REVIEW", 0.45)

    if fused_score >= block:
        return 4, "CRITICAL"
    if fused_score >= challenge:
        return 3, "HIGH"
    if fused_score >= review:
        return 2, "MEDIUM"
    return 1, "LOW"
