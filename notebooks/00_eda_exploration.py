"""
==============================================================================
Phase 0: Exploratory Data Analysis (EDA) for Mule Account Detection
==============================================================================
Dataset: Financial Transactions Dataset
Target:  F3924 (Binary: 1=Suspicious/Mule, 0=Legitimate)

Run this script first to understand your dataset before feature engineering.
==============================================================================
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

from shared_config import EDA_REPORTS_DIR, RAW_DATA_PATH

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH      = RAW_DATA_PATH
TARGET_COL     = "F3924"
REPORTS_DIR    = EDA_REPORTS_DIR
os.makedirs(REPORTS_DIR, exist_ok=True)

# Features explicitly identified by the bank as important
BANK_KEY_FEATURES = [
    "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
    "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894",
]

# ── Load Dataset ─────────────────────────────────────────────────────────────
print("=" * 70)
print("Loading dataset...")
print("=" * 70)
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Shape        : {df.shape}")
print(f"  Target col   : {TARGET_COL}")
print(f"  Memory usage : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ── Section 1: Target Variable Analysis ─────────────────────────────────────
print("\n" + "=" * 70)
print("1. TARGET VARIABLE ANALYSIS")
print("=" * 70)
target_counts = df[TARGET_COL].value_counts()
target_pct    = df[TARGET_COL].value_counts(normalize=True) * 100
print(f"\n  Class distribution:\n{target_counts.to_frame()}")
print(f"\n  Percentage:\n{target_pct.to_frame().round(3)}")
imbalance_ratio = target_counts[0] / target_counts[1]
print(f"\n  Imbalance Ratio (Neg:Pos) = {imbalance_ratio:.1f}:1")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(['Legitimate (0)', 'Suspicious/Mule (1)'],
            [target_counts.get(0, 0), target_counts.get(1, 0)],
            color=['#4CAF50', '#F44336'], edgecolor='black', linewidth=1.2)
axes[0].set_title('Class Distribution (Absolute Count)', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate([target_counts.get(0, 0), target_counts.get(1, 0)]):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

wedges, texts, autotexts = axes[1].pie(
    [target_counts.get(0, 0), target_counts.get(1, 0)],
    labels=['Legitimate', 'Suspicious/Mule'],
    autopct='%1.2f%%', colors=['#4CAF50', '#F44336'],
    startangle=90, pctdistance=0.85
)
axes[1].set_title('Class Distribution (Proportion)', fontweight='bold')
plt.suptitle('Target Variable F3924 Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/01_target_distribution.png", dpi=150)
plt.close()
print(f"  [Saved] {REPORTS_DIR}/01_target_distribution.png")

# ── Section 2: Missing Value Audit ──────────────────────────────────────────
print("\n" + "=" * 70)
print("2. MISSING VALUE AUDIT")
print("=" * 70)
miss = df.isnull().sum()
miss_pct = (miss / len(df)) * 100
miss_report = pd.DataFrame({'missing_count': miss, 'missing_pct': miss_pct})
miss_report = miss_report[miss_report['missing_count'] > 0].sort_values('missing_pct', ascending=False)
print(f"  Total features with ANY missing: {len(miss_report)}")
print(f"  Features > 50% missing         : {(miss_report['missing_pct'] > 50).sum()}")
print(f"  Features > 90% missing         : {(miss_report['missing_pct'] > 90).sum()}")
miss_report.to_csv(f"{REPORTS_DIR}/missing_value_report.csv")
print(f"  [Saved] {REPORTS_DIR}/missing_value_report.csv")

if len(miss_report) > 0:
    top_missing = miss_report.head(40)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(top_missing.index, top_missing['missing_pct'], color='#FF7043')
    ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='50% threshold')
    ax.set_xlabel('Missing %')
    ax.set_title('Top 40 Features by Missing % (Drop candidates > 90%)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/02_missing_values.png", dpi=150)
    plt.close()
    print(f"  [Saved] {REPORTS_DIR}/02_missing_values.png")

# ── Section 3: Feature Data Type Audit ──────────────────────────────────────
print("\n" + "=" * 70)
print("3. FEATURE DATA TYPE AUDIT")
print("=" * 70)
feature_cols  = [c for c in df.columns if c != TARGET_COL]
dtypes_summary = df[feature_cols].dtypes.value_counts()
print(f"\n  DType Breakdown:\n{dtypes_summary}")
numeric_cols   = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categoric_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\n  Numeric features  : {len(numeric_cols)}")
print(f"  Categorical features : {len(categoric_cols)}")

# ── Section 4: Key Bank Features Deep-Dive ─────────────────────────────────
print("\n" + "=" * 70)
print("4. KEY BANK FEATURES ANALYSIS (Bank-Specified Important Features)")
print("=" * 70)
available_key = [f for f in BANK_KEY_FEATURES if f in df.columns]
print(f"  Bank specified features available: {len(available_key)} / {len(BANK_KEY_FEATURES)}")

if len(available_key) > 0:
    key_stats = df[available_key + [TARGET_COL]].groupby(TARGET_COL).describe().T
    key_stats.to_csv(f"{REPORTS_DIR}/key_features_by_class.csv")
    print(f"  [Saved] {REPORTS_DIR}/key_features_by_class.csv")

    # Visualization: KDE distributions of key features by class
    n_cols = 3
    n_rows = (len(available_key) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()
    colors_by_class = {0: '#4CAF50', 1: '#F44336'}
    labels_by_class = {0: 'Legitimate', 1: 'Suspicious/Mule'}
    for idx, col in enumerate(available_key):
        ax = axes[idx]
        if col in numeric_cols:
            plotted = False
            for cls in [0, 1]:
                sub = df[df[TARGET_COL] == cls][col].dropna()
                if len(sub) > 0:
                    try:
                        sub.plot.kde(
                            ax=ax,
                            color=colors_by_class[cls],
                            label=labels_by_class[cls],
                            linewidth=2,
                        )
                        plotted = True
                    except Exception:
                        pass
            if not plotted:
                for cls in [0, 1]:
                    sub = df[df[TARGET_COL] == cls][col].dropna()
                    if len(sub) > 0:
                        ax.hist(
                            sub,
                            bins=30,
                            density=True,
                            alpha=0.35,
                            color=colors_by_class[cls],
                            label=labels_by_class[cls],
                        )
            ax.set_title(f'{col}', fontweight='bold', fontsize=10)
            ax.legend(fontsize=8)
            ax.set_ylabel('Density')
        else:
            ct = pd.crosstab(df[col], df[TARGET_COL], normalize='index')
            ct.plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336'])
            ax.set_title(f'{col}', fontweight='bold', fontsize=10)
    for ax in axes[len(available_key):]:
        ax.set_visible(False)
    plt.suptitle('Bank-Specified Key Feature Distributions by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/03_key_features_kde_by_class.png", dpi=150)
    plt.close()
    print(f"  [Saved] {REPORTS_DIR}/03_key_features_kde_by_class.png")

# ── Section 5: Numeric Feature Statistics ───────────────────────────────────
print("\n" + "=" * 70)
print("5. NUMERIC FEATURE OVERVIEW (SAMPLE STATS)")
print("=" * 70)
num_summary = df[numeric_cols].describe().T
num_summary['skewness'] = df[numeric_cols].skew()
num_summary['kurtosis'] = df[numeric_cols].kurtosis()
num_summary.to_csv(f"{REPORTS_DIR}/numeric_features_summary.csv")
print(f"  Numeric features summary saved to {REPORTS_DIR}/numeric_features_summary.csv")
print(f"  Highly skewed features (|skew| > 5): {(num_summary['skewness'].abs() > 5).sum()}")
print(f"  Zero-variance features              : {(num_summary['std'] == 0).sum()}")

# ── Section 6: Correlation Heatmap (Key Features) ────────────────────────────
print("\n" + "=" * 70)
print("6. CORRELATION ANALYSIS")
print("=" * 70)
if len(available_key) > 1:
    corr_df = df[available_key + [TARGET_COL]].select_dtypes(include=[np.number])
    corr_matrix = corr_df.corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='RdYlGn', center=0,
                square=True, linewidths=0.5, annot=True, fmt='.2f',
                annot_kws={'size': 7}, ax=ax)
    ax.set_title('Correlation Matrix — Key Bank Features + Target F3924', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"  [Saved] {REPORTS_DIR}/04_correlation_heatmap.png")

# ── Section 7: Target Correlation for ALL Features ──────────────────────────
print("\n" + "=" * 70)
print("7. TARGET CORRELATION (ALL NUMERIC FEATURES)")
print("=" * 70)
target_corr = df[numeric_cols].corrwith(df[TARGET_COL]).abs().sort_values(ascending=False)
top_correlated = target_corr.head(50)
top_correlated.to_csv(f"{REPORTS_DIR}/top50_feature_target_correlation.csv")
print(f"  Top 10 correlated with F3924:\n{top_correlated.head(10).to_string()}")
print(f"  [Saved] {REPORTS_DIR}/top50_feature_target_correlation.csv")

fig, ax = plt.subplots(figsize=(14, 8))
ax.barh(top_correlated.index[::-1], top_correlated.values[::-1], color='#5C6BC0')
ax.set_xlabel('|Pearson Correlation| with F3924')
ax.set_title('Top 50 Features by Absolute Correlation with Target', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/05_top50_target_correlation.png", dpi=150)
plt.close()
print(f"  [Saved] {REPORTS_DIR}/05_top50_target_correlation.png")

# ── Section 8: Label Leakage Audit ────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. LABEL LEAKAGE AUDIT (|Pearson r| > 0.50 with F3924)")
print("=" * 70)
LEAKAGE_CORR_THRESHOLD = 0.50
leakage_rows = []
for col in numeric_cols:
    corr = df[col].corr(df[TARGET_COL])
    if corr is None or np.isnan(corr):
        continue
    if abs(corr) > LEAKAGE_CORR_THRESHOLD:
        leakage_rows.append({
            "feature": col,
            "correlation_with_target": corr,
            "abs_correlation": abs(corr),
            "excluded_from_model": True,
        })

leakage_report = pd.DataFrame(leakage_rows).sort_values(
    "abs_correlation", ascending=False
) if leakage_rows else pd.DataFrame(
    columns=["feature", "correlation_with_target", "abs_correlation", "excluded_from_model"]
)
leakage_report.to_csv(f"{REPORTS_DIR}/leakage_audit.csv", index=False)
print(f"  Features exceeding threshold: {len(leakage_report)}")
if len(leakage_report) > 0:
    print(f"  Top leakage candidates:\n{leakage_report.head(10).to_string(index=False)}")
print(f"  [Saved] {REPORTS_DIR}/leakage_audit.csv")

print("\n" + "=" * 70)
print("EDA COMPLETE — Review reports in:", REPORTS_DIR)
print("=" * 70)
