"""Stacking ensemble wrapper for production inference."""
from __future__ import annotations

import numpy as np
import joblib


class StackingBundle:
    """Meta-learner over base model probability outputs."""

    def __init__(self, base_models, meta_learner, scaled_indices=None, scaler=None):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.scaled_indices = scaled_indices or []
        self.scaler = scaler
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        preds = []
        for i, model in enumerate(self.base_models):
            Xi = X_scaled if i in self.scaled_indices else X
            preds.append(model.predict_proba(Xi)[:, 1])
        meta_X = np.column_stack(preds)
        return self.meta_learner.predict_proba(meta_X)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
