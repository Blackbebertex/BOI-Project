"""
Reusable feature engineering pipeline for mule account detection.

Fitted on training data only; serialized to models/feature_pipeline.pkl for
identical transforms at serving time.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

BANK_KEY_FEATURES = [
    "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
    "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894",
]

LEAKAGE_CORR_THRESHOLD = 0.50


def safe_ratio(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    return a / (b.abs() + eps)


def safe_log(s: pd.Series, eps: float = 1e-9) -> pd.Series:
    return np.log1p(np.maximum(s, 0))


def audit_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = LEAKAGE_CORR_THRESHOLD,
) -> pd.DataFrame:
    """Return features whose absolute Pearson correlation with y exceeds threshold."""
    rows = []
    for col in X.columns:
        corr = X[col].corr(y)
        if corr is None or np.isnan(corr):
            continue
        if abs(corr) > threshold:
            rows.append({"feature": col, "correlation_with_target": corr, "excluded": True})
    return pd.DataFrame(rows)


class FeaturePipeline:
    """End-to-end feature engineering: impute, select, enrich, cluster."""

    def __init__(
        self,
        missing_thresh_pct: float = 50.0,
        variance_thresh: float = 1e-4,
        top_mi_features: int = 200,
        n_clusters: int = 5,
        leakage_corr_threshold: float = LEAKAGE_CORR_THRESHOLD,
        random_state: int = 42,
        bank_key_features: Optional[Sequence[str]] = None,
    ):
        self.missing_thresh_pct = missing_thresh_pct
        self.variance_thresh = variance_thresh
        self.top_mi_features = top_mi_features
        self.n_clusters = n_clusters
        self.leakage_corr_threshold = leakage_corr_threshold
        self.random_state = random_state
        self.bank_key_features = list(bank_key_features or BANK_KEY_FEATURES)

        self.imputer: Optional[SimpleImputer] = None
        self.variance_selector: Optional[VarianceThreshold] = None
        self.qt: Optional[QuantileTransformer] = None
        self.kmeans: Optional[KMeans] = None

        self.numeric_cols_: List[str] = []
        self.dropped_missing_: List[str] = []
        self.dropped_low_var_: List[str] = []
        self.selected_cols_: List[str] = []
        self.leakage_excluded_: List[str] = []
        self.qt_candidates_: List[str] = []
        self.cluster_input_cols_: List[str] = []
        self.mi_series_: Optional[pd.Series] = None
        self.feature_cols_: List[str] = []
        self.interaction_cols_: List[str] = []
        self.train_percentiles_: Dict[str, float] = {}
        self.is_fitted_: bool = False

    def _fit_train_percentiles(self, X_selected: pd.DataFrame) -> None:
        """Percentile cutoffs for typology interaction features (train only)."""
        pct: Dict[str, float] = {}
        pairs = [
            ("F115", 75), ("F527", 75), ("F531", 25), ("F531", 75),
        ]
        for col, q in pairs:
            if col in X_selected.columns:
                key = f"{col}_p{q}"
                clean = X_selected[col].dropna()
                pct[key] = float(np.percentile(clean, q)) if len(clean) else 0.0
        self.train_percentiles_ = pct

    def _drop_high_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        miss_pct = X.isnull().mean() * 100
        drop_missing = miss_pct[miss_pct > self.missing_thresh_pct].index.tolist()
        drop_missing = [c for c in drop_missing if c not in self.bank_key_features]
        self.dropped_missing_ = drop_missing
        return X.drop(columns=drop_missing)

    def _build_interactions(self, X_selected: pd.DataFrame) -> pd.DataFrame:
        available = set(X_selected.columns)
        new_features = {}

        if "F527" in available and "F531" in available:
            new_features["FEAT_out_in_count_ratio"] = safe_ratio(
                X_selected["F527"], X_selected["F531"]
            )
        if "F2082" in available and "F2122" in available:
            new_features["FEAT_short_long_amount_ratio"] = safe_ratio(
                X_selected["F2082"], X_selected["F2122"]
            )
        if "F2678" in available and "F2737" in available:
            new_features["FEAT_net_fund_flow"] = X_selected["F2678"] - X_selected["F2737"]
        if "F115" in available and "F321" in available:
            new_features["FEAT_age_x_freq"] = X_selected["F115"] * X_selected["F321"]
            new_features["FEAT_log_F115"] = safe_log(X_selected["F115"])
            new_features["FEAT_log_F321"] = safe_log(X_selected["F321"])
        if "F670" in available:
            new_features["FEAT_log_F670"] = safe_log(X_selected["F670"])
        if "F1692" in available and "F2582" in available:
            new_features["FEAT_amount_deviation"] = (
                X_selected["F1692"] - X_selected["F2582"]
            ).abs()
        if "F2956" in available and "F3043" in available:
            new_features["FEAT_multichannel_signal"] = (
                X_selected["F2956"] * X_selected["F3043"]
            )
        if "F3836" in available:
            new_features["FEAT_log_F3836"] = safe_log(X_selected["F3836"])
        if all(f in available for f in ("F3887", "F3889", "F3891", "F3894")):
            new_features["FEAT_terminal_aggregate"] = (
                X_selected["F3887"]
                + X_selected["F3889"]
                + X_selected["F3891"]
                + X_selected["F3894"]
            )
            new_features["FEAT_terminal_max"] = np.maximum.reduce([
                X_selected["F3887"],
                X_selected["F3889"],
                X_selected["F3891"],
                X_selected["F3894"],
            ])

        # Typology-oriented features (percentiles applied when fitted)
        pct = self.train_percentiles_ or {}
        p115_75 = pct.get("F115_p75", X_selected["F115"].quantile(0.75) if "F115" in available else 0)
        p527_75 = pct.get("F527_p75", X_selected["F527"].quantile(0.75) if "F527" in available else 0)
        p531_25 = pct.get("F531_p25", X_selected["F531"].quantile(0.25) if "F531" in available else 0)

        if "F115" in available and "F527" in available:
            new_features["FEAT_dormancy_activation"] = (
                (X_selected["F115"] > p115_75) & (X_selected["F527"] > p527_75)
            ).astype(float)
        if "F531" in available and "F527" in available:
            new_features["FEAT_silent_to_active"] = (
                (X_selected["F531"] < p531_25) & (X_selected["F527"] > p527_75)
            ).astype(float)
        if "F2678" in available and "F2737" in available:
            new_features["FEAT_large_amount_mover"] = safe_log(
                X_selected["F2678"].abs() + X_selected["F2737"].abs()
            )
            new_features["FEAT_flow_magnitude"] = (
                X_selected["F2678"] - X_selected["F2737"]
            ).abs()
        if "F2082" in available and "F2122" in available:
            new_features["FEAT_velocity_burst"] = safe_ratio(
                X_selected["F2082"], X_selected["F2122"]
            )

        skew_values = X_selected.skew()
        high_skew_cols = skew_values[skew_values.abs() > 5].index.tolist()
        high_skew_cols = [c for c in high_skew_cols if c not in self.bank_key_features]
        for col in high_skew_cols[:30]:
            new_features[f"LOG_{col}"] = safe_log(X_selected[col])

        if not new_features:
            return pd.DataFrame(index=X_selected.index)
        return pd.DataFrame(new_features, index=X_selected.index)

    def fit(self, X_raw: pd.DataFrame, y: pd.Series) -> "FeaturePipeline":
        X = self._drop_high_missing(X_raw)
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[self.numeric_cols_].copy()

        self.imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_num),
            columns=self.numeric_cols_,
            index=X_num.index,
        )

        self.variance_selector = VarianceThreshold(threshold=self.variance_thresh)
        self.variance_selector.fit(X_imputed)
        low_var_mask = self.variance_selector.get_support()
        self.dropped_low_var_ = [
            c for c, keep in zip(self.numeric_cols_, low_var_mask)
            if not keep and c not in self.bank_key_features
        ]
        X_var = X_imputed.drop(columns=self.dropped_low_var_)

        mi_scores = mutual_info_classif(
            X_var, y, discrete_features=False, random_state=self.random_state
        )
        self.mi_series_ = pd.Series(mi_scores, index=X_var.columns).sort_values(
            ascending=False
        )

        top_mi = self.mi_series_.head(self.top_mi_features).index.tolist()
        must_include = [f for f in self.bank_key_features if f in X_var.columns]
        selected_cols = list(dict.fromkeys(top_mi + must_include))

        leakage_df = audit_leakage(X_var[selected_cols], y, self.leakage_corr_threshold)
        self.leakage_excluded_ = leakage_df["feature"].tolist() if len(leakage_df) else []
        selected_cols = [c for c in selected_cols if c not in self.leakage_excluded_]
        self.selected_cols_ = selected_cols

        X_selected = X_var[selected_cols].copy()
        self._fit_train_percentiles(X_selected)
        feat_df = self._build_interactions(X_selected)
        self.interaction_cols_ = list(feat_df.columns)
        X_enriched = pd.concat([X_selected, feat_df], axis=1)

        skew_enriched = X_enriched.skew()
        self.qt_candidates_ = [
            c for c in skew_enriched[skew_enriched.abs() > 3].index.tolist()
            if c not in self.bank_key_features
        ]
        if self.qt_candidates_:
            self.qt = QuantileTransformer(
                output_distribution="normal", random_state=self.random_state
            )
            qt_input = X_enriched.reindex(columns=self.qt_candidates_, fill_value=0.0)
            X_enriched[self.qt_candidates_] = self.qt.fit_transform(qt_input)

        self.cluster_input_cols_ = [
            c for c in self.mi_series_.head(20).index if c in X_enriched.columns
        ]
        cluster_input = X_enriched[self.cluster_input_cols_].fillna(0)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(cluster_input)
        X_enriched["FEAT_kmeans_cluster"] = cluster_labels.astype(float)
        centroid_distances = self.kmeans.transform(cluster_input)
        for k in range(self.n_clusters):
            X_enriched[f"FEAT_dist_cluster_{k}"] = centroid_distances[:, k]

        self.feature_cols_ = list(X_enriched.columns)
        self.is_fitted_ = True
        return self

    def _align_raw(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Align incoming rows to training numeric columns (NaN for missing)."""
        data = {col: X_raw[col] if col in X_raw.columns else np.nan for col in self.numeric_cols_}
        return pd.DataFrame(data, index=X_raw.index)

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FeaturePipeline must be fitted before transform.")

        X = self._drop_high_missing(X_raw)
        X_num = self._align_raw(X)
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_num),
            columns=self.numeric_cols_,
            index=X_num.index,
        )
        X_var = X_imputed.drop(columns=self.dropped_low_var_, errors="ignore")

        for col in self.selected_cols_:
            if col not in X_var.columns:
                X_var[col] = 0.0
        X_selected = X_var[self.selected_cols_].copy()

        feat_df = self._build_interactions(X_selected)
        for col in self.interaction_cols_:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        X_enriched = pd.concat([X_selected, feat_df], axis=1)

        if self.qt is not None and self.qt_candidates_:
            qt_input = X_enriched.reindex(columns=self.qt_candidates_, fill_value=0.0)
            X_enriched[self.qt_candidates_] = self.qt.transform(qt_input)

        cluster_input = X_enriched.reindex(
            columns=self.cluster_input_cols_, fill_value=0.0
        ).fillna(0)
        cluster_labels = self.kmeans.predict(cluster_input)
        X_enriched["FEAT_kmeans_cluster"] = cluster_labels.astype(float)
        centroid_distances = self.kmeans.transform(cluster_input)
        for k in range(self.n_clusters):
            X_enriched[f"FEAT_dist_cluster_{k}"] = centroid_distances[:, k]

        for col in self.feature_cols_:
            if col not in X_enriched.columns:
                X_enriched[col] = 0.0
        return X_enriched[self.feature_cols_]

    def transform_from_bank_keys(self, features: dict) -> pd.DataFrame:
        """Transform a single account dict (18 bank keys minimum) to model input."""
        return self.transform(pd.DataFrame([features]))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str) -> "FeaturePipeline":
        return joblib.load(path)

    def leakage_report(self) -> pd.DataFrame:
        if not self.leakage_excluded_:
            return pd.DataFrame(columns=["feature", "correlation_with_target", "excluded"])
        return pd.DataFrame({
            "feature": self.leakage_excluded_,
            "excluded": [True] * len(self.leakage_excluded_),
        })
