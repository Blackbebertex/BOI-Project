"""Shared stratified train/holdout split indices for reproducible ML pipeline."""

import numpy as np
from sklearn.model_selection import train_test_split

from shared_config import DATA_DIR

SPLITS_DIR = DATA_DIR / "splits"
TRAIN_INDICES_PATH = SPLITS_DIR / "train_indices.npy"
HOLDOUT_INDICES_PATH = SPLITS_DIR / "holdout_indices.npy"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_or_create_split(
    n_samples: int,
    y: np.ndarray,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (train_indices, holdout_indices) with stratification on y.
    Persists indices to data/splits/ for reproducibility across pipeline phases.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    if (
        not force
        and TRAIN_INDICES_PATH.exists()
        and HOLDOUT_INDICES_PATH.exists()
    ):
        train_idx = np.load(TRAIN_INDICES_PATH)
        holdout_idx = np.load(HOLDOUT_INDICES_PATH)
        return train_idx, holdout_idx

    indices = np.arange(n_samples)
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    np.save(TRAIN_INDICES_PATH, train_idx)
    np.save(HOLDOUT_INDICES_PATH, holdout_idx)
    return train_idx, holdout_idx
