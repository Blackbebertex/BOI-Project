"""Unit tests for FeaturePipeline fit/transform parity."""
import numpy as np
import pandas as pd

from serving.feature_pipeline import BANK_KEY_FEATURES, FeaturePipeline


def _synthetic_raw(n: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    data = {}
    for feat in BANK_KEY_FEATURES:
        data[feat] = rng.uniform(0.1, 2.0, size=n)
    for i in range(1, 31):
        col = f"F{i}"
        if col not in data:
            data[col] = rng.normal(size=n)
    # Leaky column — should be excluded
    y = rng.integers(0, 2, size=n)
    data["F3912"] = y.astype(float) * 0.95 + rng.normal(scale=0.05, size=n)
    X = pd.DataFrame(data)
    return X, pd.Series(y)


def test_pipeline_fit_transform_shape():
    X, y = _synthetic_raw()
    pipe = FeaturePipeline(top_mi_features=10, n_clusters=3)
    pipe.fit(X, y)
    out = pipe.transform(X.iloc[:5])
    assert out.shape[0] == 5
    assert out.shape[1] == len(pipe.feature_cols_)
    assert "F3912" not in pipe.selected_cols_


def test_pipeline_excludes_leakage_features():
    X, y = _synthetic_raw()
    pipe = FeaturePipeline(top_mi_features=15, leakage_corr_threshold=0.5)
    pipe.fit(X, y)
    assert "F3912" in pipe.leakage_excluded_


def test_transform_from_bank_keys_only():
    X, y = _synthetic_raw(n=200)
    pipe = FeaturePipeline(top_mi_features=10, n_clusters=3)
    pipe.fit(X, y)
    row = {f: float(X.iloc[0][f]) for f in BANK_KEY_FEATURES}
    out = pipe.transform_from_bank_keys(row)
    assert out.shape == (1, len(pipe.feature_cols_))


def test_pipeline_save_load(tmp_path):
    X, y = _synthetic_raw(n=150)
    pipe = FeaturePipeline(top_mi_features=8, n_clusters=2)
    pipe.fit(X, y)
    path = tmp_path / "pipeline.pkl"
    pipe.save(path)
    loaded = FeaturePipeline.load(path)
    pd.testing.assert_frame_equal(
        pipe.transform(X.iloc[:3]),
        loaded.transform(X.iloc[:3]),
        check_dtype=False,
        rtol=1e-5,
    )
