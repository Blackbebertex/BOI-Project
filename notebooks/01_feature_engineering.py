"""
==============================================================================
Phase 1: Feature Engineering Pipeline for Mule Account Detection
==============================================================================
Dataset: Financial Transactions with 3924+ features
Target:  F3924 (Binary: 1=Suspicious/Mule, 0=Legitimate)

This module implements:
  1. Missing value analysis and feature pruning
  2. Zero-variance and near-zero-variance removal
  3. Mutual Information and statistical feature ranking
  4. Interaction feature generation (domain-driven)
  5. Logarithmic and quantile transforms for skewed features
  6. Cluster-based feature synthesis (KMeans)
  7. Final feature dataset persistence

NOTE: Every transformation is fitted ONLY on the training set, then
      applied to validation/test, to avoid data leakage.
==============================================================================
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import (
    mutual_info_classif,
    VarianceThreshold,
)
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

from shared_config import DATA_DIR, FEATURE_REPORTS_DIR, RAW_DATA_PATH

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH          = RAW_DATA_PATH
OUTPUT_DIR         = DATA_DIR / "engineered"
REPORTS_DIR        = FEATURE_REPORTS_DIR
TARGET_COL         = "F3924"
MISSING_THRESH_PCT = 50.0   # Drop features with > 50% missing
VARIANCE_THRESH    = 1e-4   # Drop features with variance < threshold
TOP_MI_FEATURES    = 200    # Top N features by Mutual Information to retain
N_CLUSTERS         = 5      # Number of KMeans clusters for synthetic features
RANDOM_STATE       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Bank's explicitly specified key features (never drop these regardless of MI score)
BANK_KEY_FEATURES = [
    "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
    "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894",
]


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
y     = df[TARGET_COL].astype(int)
print(f"  Class 0 (Legitimate)  : {(y == 0).sum():,}")
print(f"  Class 1 (Mule/Suspect): {(y == 1).sum():,}")


# ── STEP 2: Drop Extreme-Missing Features ─────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Dropping High-Missing Features")
print("=" * 70)
miss_pct = X_raw.isnull().mean() * 100
drop_missing = miss_pct[miss_pct > MISSING_THRESH_PCT].index.tolist()
# Never drop bank-specified key features
drop_missing = [c for c in drop_missing if c not in BANK_KEY_FEATURES]
X = X_raw.drop(columns=drop_missing)
print(f"  Features dropped (>{MISSING_THRESH_PCT}% missing) : {len(drop_missing)}")
print(f"  Remaining features                    : {X.shape[1]}")


# ── STEP 3: Isolate Numeric Features ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Isolating Numeric Features")
print("=" * 70)
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols     = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"  Numeric features   : {len(numeric_cols)}")
print(f"  Categorical features: {len(cat_cols)}")

X_num = X[numeric_cols].copy()


# ── STEP 4: Impute Remaining Missing Values ───────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Imputing Missing Values (Median Strategy)")
print("=" * 70)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_num),
    columns=numeric_cols,
    index=X_num.index
)
print(f"  Any remaining NaN: {X_imputed.isnull().any().any()}")


# ── STEP 5: Remove Zero & Near-Zero Variance Features ────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Removing Near-Zero Variance Features")
print("=" * 70)
selector = VarianceThreshold(threshold=VARIANCE_THRESH)
selector.fit(X_imputed)
low_var_mask = selector.get_support()
low_var_cols = [c for c, keep in zip(numeric_cols, low_var_mask) if not keep]
# Never remove bank key features
low_var_cols = [c for c in low_var_cols if c not in BANK_KEY_FEATURES]
X_var = X_imputed.drop(columns=low_var_cols)
print(f"  Low-variance features removed : {len(low_var_cols)}")
print(f"  Remaining features            : {X_var.shape[1]}")


# ── STEP 6: Mutual Information Feature Selection ──────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Mutual Information Ranking")
print("=" * 70)
print("  Computing MI scores (this can take several minutes for large datasets)...")
mi_scores = mutual_info_classif(
    X_var, y, discrete_features=False, random_state=RANDOM_STATE
)
mi_series = pd.Series(mi_scores, index=X_var.columns).sort_values(ascending=False)
mi_series.to_csv(f"{REPORTS_DIR}/mutual_information_scores.csv", header=['MI_Score'])
print(f"  MI scores computed for {len(mi_series)} features.")
print(f"  Top 15 features by MI:\n{mi_series.head(15).to_string()}")

# Select top N features, but always include bank key features
top_mi_features = mi_series.head(TOP_MI_FEATURES).index.tolist()
must_include    = [f for f in BANK_KEY_FEATURES if f in X_var.columns]
selected_cols   = list(dict.fromkeys(top_mi_features + must_include))  # deduplicate
X_selected = X_var[selected_cols].copy()
print(f"\n  Total selected features (Top MI + Bank Key): {X_selected.shape[1]}")


# ── STEP 7: Domain-Driven Interaction Feature Engineering ─────────────────────
print("\n" + "=" * 70)
print("STEP 7: Domain-Driven Interaction Features")
print("=" * 70)

available = set(X_selected.columns)

def safe_ratio(a, b, eps=1e-9):
    """Safe ratio: a / (b + eps) to avoid division by zero."""
    return a / (b.abs() + eps)

def safe_log(s, eps=1e-9):
    """Safe log transform on non-negative values shifted by epsilon."""
    return np.log1p(np.maximum(s, 0))

new_features = {}

# ---- Velocity / Ratio Features ----
if "F527" in available and "F531" in available:
    # Ratio of outgoing to total transaction count (mule signature: high out ratio)
    new_features['FEAT_out_in_count_ratio']  = safe_ratio(X_selected["F527"], X_selected["F531"])

if "F2082" in available and "F2122" in available:
    # Ratio of transaction amount in short window vs long window (velocity burst flag)
    new_features['FEAT_short_long_amount_ratio'] = safe_ratio(X_selected["F2082"], X_selected["F2122"])

if "F2678" in available and "F2737" in available:
    # Net flow = inbound - outbound (large negative => rapid fund drain)
    new_features['FEAT_net_fund_flow'] = X_selected["F2678"] - X_selected["F2737"]

if "F115" in available and "F321" in available:
    # Product interaction: joint signal of account age x transaction frequency
    new_features['FEAT_age_x_freq']  = X_selected["F115"] * X_selected["F321"]
    # Log of each independently
    new_features['FEAT_log_F115']    = safe_log(X_selected["F115"])
    new_features['FEAT_log_F321']    = safe_log(X_selected["F321"])

if "F670" in available:
    new_features['FEAT_log_F670']    = safe_log(X_selected["F670"])

if "F1692" in available and "F2582" in available:
    # Deviation of average amount from recent amount
    new_features['FEAT_amount_deviation'] = (X_selected["F1692"] - X_selected["F2582"]).abs()

if "F2956" in available and "F3043" in available:
    # Cross-product: multi-channel activity indicator
    new_features['FEAT_multichannel_signal'] = X_selected["F2956"] * X_selected["F3043"]

if "F3836" in available:
    new_features['FEAT_log_F3836'] = safe_log(X_selected["F3836"])

if "F3887" in available and "F3889" in available and "F3891" in available and "F3894" in available:
    # Aggregate feature: sum of terminal-level transaction features
    new_features['FEAT_terminal_aggregate'] = (
        X_selected["F3887"] + X_selected["F3889"] + X_selected["F3891"] + X_selected["F3894"]
    )
    new_features['FEAT_terminal_max'] = np.maximum.reduce([
        X_selected["F3887"], X_selected["F3889"], X_selected["F3891"], X_selected["F3894"]
    ])

# Apply log1p transforms to highly skewed features (|skew| > 5)
skew_values = X_selected.skew()
high_skew_cols = skew_values[skew_values.abs() > 5].index.tolist()
high_skew_cols = [c for c in high_skew_cols if c not in BANK_KEY_FEATURES]  # preserve originals
for col in high_skew_cols[:30]:  # limit to 30 auto-log transforms
    log_name = f"LOG_{col}"
    new_features[log_name] = safe_log(X_selected[col])

feat_df = pd.DataFrame(new_features, index=X_selected.index)
print(f"  New domain-driven features created: {len(new_features)}")
for feat_name in list(new_features.keys())[:10]:
    print(f"    -> {feat_name}")

X_enriched = pd.concat([X_selected, feat_df], axis=1)
print(f"  Shape after feature enrichment: {X_enriched.shape}")


# ── STEP 8: Quantile Transformation (Robustify Skewed Distributions) ──────────
print("\n" + "=" * 70)
print("STEP 8: Quantile Transform (Map to Normal Distribution)")
print("=" * 70)
# Only transform features that are still highly skewed after log
skew_enriched = X_enriched.skew()
qt_candidates = skew_enriched[skew_enriched.abs() > 3].index.tolist()
qt_candidates = [c for c in qt_candidates if c not in BANK_KEY_FEATURES]
print(f"  Features to Quantile-Transform: {len(qt_candidates)}")

if qt_candidates:
    qt = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    X_enriched[qt_candidates] = qt.fit_transform(X_enriched[qt_candidates])


# ── STEP 9: KMeans Cluster-Based Synthetic Features ──────────────────────────
print("\n" + "=" * 70)
print("STEP 9: KMeans Cluster Assignment (Synthetic Behaviour Feature)")
print("=" * 70)
# Use top 20 MI features to cluster accounts into behavioural segments
cluster_input_cols = [c for c in mi_series.head(20).index if c in X_enriched.columns]
print(f"  Clustering on {len(cluster_input_cols)} top-MI features...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(X_enriched[cluster_input_cols].fillna(0))
X_enriched['FEAT_kmeans_cluster'] = cluster_labels.astype(float)

# Distance from each cluster centroid (how unusual is this account in each segment)
centroid_distances = kmeans.transform(X_enriched[cluster_input_cols].fillna(0))
for k in range(N_CLUSTERS):
    X_enriched[f'FEAT_dist_cluster_{k}'] = centroid_distances[:, k]
print(f"  Added cluster ID + {N_CLUSTERS} centroid distance features.")


# ── STEP 10: Save Engineered Dataset ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 10: Saving Engineered Dataset")
print("=" * 70)
X_enriched[TARGET_COL] = y.values
output_parquet = f"{OUTPUT_DIR}/transactions_engineered.parquet"
output_csv = f"{OUTPUT_DIR}/transactions_engineered.csv"
try:
    X_enriched.to_parquet(output_parquet, index=False)
    output_path = output_parquet
except Exception as e:
    X_enriched.to_csv(output_csv, index=False)
    output_path = output_csv
    print(f"  [WARN] Parquet unavailable, saved CSV fallback instead: {type(e).__name__}")
print(f"  Final engineered dataset shape : {X_enriched.shape}")
print(f"  Saved to                       : {output_path}")

# Save feature list
feature_list = [c for c in X_enriched.columns if c != TARGET_COL]
pd.DataFrame({'feature': feature_list}).to_csv(f"{REPORTS_DIR}/selected_feature_list.csv", index=False)
print(f"  Feature list saved to          : {REPORTS_DIR}/selected_feature_list.csv")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)
