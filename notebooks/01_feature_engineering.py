"""
==============================================================================
Phase 1: Feature Engineering Pipeline for Mule Account Detection
==============================================================================
Dataset: Financial Transactions with 3924+ features
Target:  F3924 (Binary: 1=Suspicious/Mule, 0=Legitimate)

This module implements:
  1. Stratified train/holdout split (80/20)
  2. FeaturePipeline fitted on TRAIN ONLY (no leakage)
  3. Label leakage audit (exclude |r| > 0.50 vs target)
  4. Serialized pipeline for serving (models/feature_pipeline.pkl)

NOTE: Every transformation is fitted ONLY on the training set, then
      applied to holdout, to avoid data leakage.
==============================================================================
"""
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# Allow imports from serving/ when run as a script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from serving.feature_pipeline import FeaturePipeline, audit_leakage, BANK_KEY_FEATURES
from serving.typology_engine import TypologyEngine
from shared_config import DATA_DIR, EDA_REPORTS_DIR, FEATURE_REPORTS_DIR, MODELS_DIR, RAW_DATA_PATH
from shared_splits import get_or_create_split

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH = RAW_DATA_PATH
OUTPUT_DIR = DATA_DIR / "engineered"
REPORTS_DIR = FEATURE_REPORTS_DIR
TARGET_COL = "F3924"
LEAKAGE_CORR_THRESHOLD = 0.50

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── STEP 1: Load Data ─────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading Raw Dataset")
print("=" * 70)
df = pd.read_csv(DATA_PATH, low_memory=False)
unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)
    print(f"  Dropped index-like columns: {unnamed_cols}")
print(f"  Shape: {df.shape}")

X_raw = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)
print(f"  Class 0 (Legitimate)  : {(y == 0).sum():,}")
print(f"  Class 1 (Mule/Suspect): {(y == 1).sum():,}")

# ── STEP 2: Train / Holdout Split ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Stratified Train/Holdout Split (80/20)")
print("=" * 70)
train_idx, holdout_idx = get_or_create_split(len(df), y.values)
X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
X_holdout_raw = X_raw.iloc[holdout_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
y_holdout = y.iloc[holdout_idx].reset_index(drop=True)
print(f"  Train rows   : {len(train_idx):,}  (positives: {y_train.sum()})")
print(f"  Holdout rows : {len(holdout_idx):,}  (positives: {y_holdout.sum()})")

# ── STEP 3: Leakage Audit on Training Data ────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Label Leakage Audit (train set only)")
print("=" * 70)
numeric_train = X_train_raw.select_dtypes(include="number")
leakage_df = audit_leakage(numeric_train, y_train, LEAKAGE_CORR_THRESHOLD)
leakage_df["abs_correlation"] = leakage_df["correlation_with_target"].abs()
leakage_df = leakage_df.sort_values("abs_correlation", ascending=False)
leakage_df.to_csv(f"{EDA_REPORTS_DIR}/leakage_audit.csv", index=False)
print(f"  Features excluded (|r| > {LEAKAGE_CORR_THRESHOLD}): {len(leakage_df)}")
if len(leakage_df) > 0:
    print(leakage_df.head(10).to_string(index=False))
print("  [Saved] reports/eda/leakage_audit.csv")

# ── STEP 4: Fit FeaturePipeline on Train ──────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Fitting FeaturePipeline (train only)")
print("=" * 70)
pipeline = FeaturePipeline(leakage_corr_threshold=LEAKAGE_CORR_THRESHOLD)
pipeline.fit(X_train_raw, y_train)
print(f"  Selected base features : {len(pipeline.selected_cols_)}")
print(f"  Leakage excluded       : {pipeline.leakage_excluded_}")
print(f"  Final feature count    : {len(pipeline.feature_cols_)}")

# ── STEP 5: Transform Train & Holdout ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Transforming Train and Holdout")
print("=" * 70)
X_train_eng = pipeline.transform(X_train_raw)
X_holdout_eng = pipeline.transform(X_holdout_raw)
X_train_eng[TARGET_COL] = y_train.values
X_holdout_eng[TARGET_COL] = y_holdout.values
print(f"  Train engineered shape   : {X_train_eng.shape}")
print(f"  Holdout engineered shape : {X_holdout_eng.shape}")

# ── STEP 6: Save Artifacts ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Saving Engineered Data and Pipeline")
print("=" * 70)

train_parquet = OUTPUT_DIR / "transactions_engineered.parquet"
train_csv = OUTPUT_DIR / "transactions_engineered.csv"
holdout_parquet = OUTPUT_DIR / "transactions_engineered_holdout.parquet"
holdout_csv = OUTPUT_DIR / "transactions_engineered_holdout.csv"

try:
    X_train_eng.to_parquet(train_parquet, index=False)
    X_holdout_eng.to_parquet(holdout_parquet, index=False)
    print(f"  Saved train   : {train_parquet}")
    print(f"  Saved holdout : {holdout_parquet}")
except Exception as e:
    X_train_eng.to_csv(train_csv, index=False)
    X_holdout_eng.to_csv(holdout_csv, index=False)
    print(f"  [WARN] Parquet unavailable ({type(e).__name__}); saved CSV fallback")

pipeline_path = MODELS_DIR / "feature_pipeline.pkl"
pipeline.save(pipeline_path)
print(f"  Saved pipeline: {pipeline_path}")

typology_engine = TypologyEngine.fit_from_train(
    X_train_raw[BANK_KEY_FEATURES],
    MODELS_DIR / "typology_thresholds.json",
)
print(f"  Saved typology thresholds: {MODELS_DIR / 'typology_thresholds.json'}")
print(f"  Typology keys: {list(typology_engine.thresholds.keys())[:6]}...")

feature_list = pipeline.feature_cols_
pd.DataFrame({"feature": feature_list}).to_csv(
    f"{REPORTS_DIR}/selected_feature_list.csv", index=False
)
print(f"  Feature list  : {REPORTS_DIR}/selected_feature_list.csv")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)
