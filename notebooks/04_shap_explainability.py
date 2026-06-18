"""
==============================================================================
Phase 4: Model Explainability (SHAP) for Mule Account Detection
==============================================================================
Generates SHAP explanations for the LightGBM model to:
  1. Validate feature contributions make domain sense
  2. Produce per-account decision explanations for fraud investigators
  3. Identify the most globally influential features (model-wide)
  4. Create a narrative explanation template per-case

Output artifacts:
  - SHAP summary bar plot (global feature importance)
  - SHAP beeswarm plot (feature value vs impact direction)
  - SHAP waterfall plot (individual case explanation)
  - Per-account SHAP explanation CSV for investigator portal
==============================================================================
"""
import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib

warnings.filterwarnings('ignore')

from shared_config import EXPLAINABILITY_REPORTS_DIR, MODELS_DIR, load_engineered_data

# ── CONFIG ───────────────────────────────────────────────────────────────────
REPORTS_DIR  = EXPLAINABILITY_REPORTS_DIR
TARGET_COL   = "F3924"
N_BACKGROUND = 200   # Background samples for SHAP kernel
N_EXPLAIN    = 1000  # Number of test samples to compute SHAP for

os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Load Data & Model ─────────────────────────────────────────────────────────
print("=" * 70)
print("Loading Dataset & LightGBM Model for SHAP Analysis")
print("=" * 70)
df = load_engineered_data()
feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols]
y = df[TARGET_COL].values.astype(int)

lgbm_model = joblib.load(f"{MODELS_DIR}/lgbm_final.pkl")
print(f"  Data shape : {X.shape}")
print(f"  Model type : {type(lgbm_model).__name__}")

# Sample for analysis (take representative mix of both classes)
mule_idx  = np.where(y == 1)[0]
legit_idx = np.where(y == 0)[0]
np.random.seed(42)
sample_idx = np.concatenate([
    np.random.choice(mule_idx,  min(N_EXPLAIN // 2, len(mule_idx)),  replace=False),
    np.random.choice(legit_idx, min(N_EXPLAIN // 2, len(legit_idx)), replace=False),
])
np.random.shuffle(sample_idx)
X_sample = X.iloc[sample_idx].reset_index(drop=True)
y_sample = y[sample_idx]
print(f"  SHAP sample: {X_sample.shape}  (Mule: {y_sample.sum()}, Legit: {(y_sample==0).sum()})")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP TreeExplainer (optimized for tree-based models — sub-second per sample)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Computing SHAP Values (TreeExplainer)")
print("=" * 70)
explainer   = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_sample)

# For binary classification, LightGBM TreeExplainer returns list [neg_class, pos_class]
# We use the positive class (mule = 1) SHAP values
if isinstance(shap_values, list):
    shap_vals_pos = shap_values[1]
else:
    shap_vals_pos = shap_values

print(f"  SHAP values shape: {shap_vals_pos.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: SHAP Summary Bar (Global Feature Importance)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating: SHAP Summary Bar Plot...")
plt.figure(figsize=(12, 9))
shap.summary_plot(
    shap_vals_pos, X_sample,
    plot_type="bar",
    max_display=30,
    show=False,
)
plt.title('SHAP Feature Importance — Global (Top 30)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/shap_global_importance_bar.png", dpi=150)
plt.close()
print(f"  [Saved] {REPORTS_DIR}/shap_global_importance_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: SHAP Beeswarm (Feature Value vs Impact Direction)
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating: SHAP Beeswarm Plot...")
plt.figure(figsize=(13, 10))
shap.summary_plot(
    shap_vals_pos, X_sample,
    plot_type="dot",
    max_display=30,
    show=False,
)
plt.title('SHAP Beeswarm — Feature Value Impact on Mule Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/shap_beeswarm.png", dpi=150)
plt.close()
print(f"  [Saved] {REPORTS_DIR}/shap_beeswarm.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: SHAP Waterfall Plot — Individual Case Explanations
#    Show for: (a) True Positive (Mule caught), (b) False Positive
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating: SHAP Waterfall Plots for Individual Cases...")
pred_probs = lgbm_model.predict_proba(X_sample)[:, 1]
expected_val = explainer.expected_value
if isinstance(expected_val, list):
    expected_val = expected_val[1]

# Case A: Highest-scored confirmed mule account
mule_mask    = y_sample == 1
if mule_mask.sum() > 0:
    highest_mule_idx = np.where(mule_mask)[0][np.argmax(pred_probs[mule_mask])]
    shap_exp = shap.Explanation(
        values        = shap_vals_pos[highest_mule_idx],
        base_values   = expected_val,
        data          = X_sample.iloc[highest_mule_idx].values,
        feature_names = feature_cols,
    )
    fig, ax = plt.subplots(figsize=(14, 8))
    shap.plots.waterfall(shap_exp, max_display=20, show=False)
    plt.title(
        f'SHAP Waterfall — Confirmed Mule Account '
        f'(Pred Score: {pred_probs[highest_mule_idx]:.3f})',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_waterfall_true_positive.png", dpi=150)
    plt.close()
    print(f"  [Saved] {REPORTS_DIR}/shap_waterfall_true_positive.png")

# Case B: Highest-scored legitimate account (potential false positive case for review)
legit_mask   = y_sample == 0
if legit_mask.sum() > 0:
    highest_fp_idx = np.where(legit_mask)[0][np.argmax(pred_probs[legit_mask])]
    shap_exp_fp = shap.Explanation(
        values        = shap_vals_pos[highest_fp_idx],
        base_values   = expected_val,
        data          = X_sample.iloc[highest_fp_idx].values,
        feature_names = feature_cols,
    )
    fig, ax = plt.subplots(figsize=(14, 8))
    shap.plots.waterfall(shap_exp_fp, max_display=20, show=False)
    plt.title(
        f'SHAP Waterfall — High-Scored Legitimate Account '
        f'(Potential FP, Score: {pred_probs[highest_fp_idx]:.3f})',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_waterfall_false_positive.png", dpi=150)
    plt.close()
    print(f"  [Saved] {REPORTS_DIR}/shap_waterfall_false_positive.png")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTABLE EXPLANATION: Per-Account Top SHAP Drivers for Investigator Portal
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Generating Per-Account Explanation Export")
print("=" * 70)

def generate_case_narrative(account_idx, shap_vals_row, feature_names, pred_prob, actual_label):
    """
    Produces a human-readable risk narrative string for an account,
    listing the top 3 factors pushing the score HIGH (mule indicators).
    """
    shap_series = pd.Series(shap_vals_row, index=feature_names)
    top_drivers = shap_series.nlargest(3)
    bottom_drivers = shap_series.nsmallest(3)

    risk_level = "CRITICAL" if pred_prob >= 0.85 else ("HIGH" if pred_prob >= 0.60 else "MEDIUM")
    lines = [
        f"Risk Level     : {risk_level}",
        f"Model Score    : {pred_prob:.4f}",
        f"Actual Label   : {'MULE/SUSPICIOUS' if actual_label == 1 else 'LEGITIMATE'}",
        "",
        "Top Risk Drivers (Factors Increasing Mule Score):",
    ]
    for feat, val in top_drivers.items():
        lines.append(f"  + {feat:<40} SHAP={val:+.4f}")
    lines.append("")
    lines.append("Top Mitigating Factors (Factors Reducing Mule Score):")
    for feat, val in bottom_drivers.items():
        lines.append(f"  - {feat:<40} SHAP={val:+.4f}")
    return "\n".join(lines)

# Build explanation dataframe
records = []
for i in range(len(X_sample)):
    shap_row  = shap_vals_pos[i]
    shap_ser  = pd.Series(shap_row, index=feature_cols)
    top3      = shap_ser.nlargest(3)
    records.append({
        'account_index'    : sample_idx[i],
        'pred_score'       : pred_probs[i],
        'actual_label'     : y_sample[i],
        'top_driver_1'     : top3.index[0] if len(top3) > 0 else None,
        'top_driver_1_shap': top3.iloc[0]  if len(top3) > 0 else 0.0,
        'top_driver_2'     : top3.index[1] if len(top3) > 1 else None,
        'top_driver_2_shap': top3.iloc[1]  if len(top3) > 1 else 0.0,
        'top_driver_3'     : top3.index[2] if len(top3) > 2 else None,
        'top_driver_3_shap': top3.iloc[2]  if len(top3) > 2 else 0.0,
        'narrative'        : generate_case_narrative(i, shap_row, feature_cols, pred_probs[i], y_sample[i]),
    })

explanation_df = pd.DataFrame(records).sort_values('pred_score', ascending=False)
explanation_df.to_csv(f"{REPORTS_DIR}/per_account_shap_explanations.csv", index=False)
print(f"  [Saved] {REPORTS_DIR}/per_account_shap_explanations.csv")

# Print sample narrative for top predicted mule
print("\n  === Sample Narrative for Highest Risk Account ===")
print(explanation_df.iloc[0]['narrative'])

print("\n" + "=" * 70)
print("EXPLAINABILITY ANALYSIS COMPLETE")
print("=" * 70)
