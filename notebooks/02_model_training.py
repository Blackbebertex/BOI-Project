"""
==============================================================================
Phase 2: Model Training — Multi-Model Ensemble for Mule Account Detection
==============================================================================
Dataset: Engineered feature set (output of 01_feature_engineering.py)
Target:  F3924 (Binary: 1=Suspicious/Mule, 0=Legitimate)

Models Trained:
  1. LightGBM Classifier       (primary workhorse for high-dimensional tabular)
  2. XGBoost Classifier         (complementary gradient boosted tree)
  3. Random Forest Classifier   (bagging-based diverse learner)
  4. Logistic Regression        (linear baseline for calibration reference)
  5. Stacking Ensemble          (meta-classifier over base model predictions)

Key Fraud Detection Metrics:
  - Primary  : Average Precision (AP) Score / PR-AUC
  - Secondary: ROC-AUC, Recall @ FPR=1%, F1 Score
  - Threshold-Free: PR Curve plotted for all models

Class Imbalance Strategy:
  - LightGBM  : scale_pos_weight / focal loss
  - XGBoost   : scale_pos_weight
  - Random Forest: class_weight='balanced'
  - Logistic Regression: class_weight='balanced'
  - Stacking: final estimator trained on balanced labels

Training Protocol:
  - Stratified K-Fold (5 splits) for stable cross-validated evaluation
  - Hyperparameter search via Optuna for LightGBM (primary model)
  - Out-of-fold (OOF) predictions for the stacking meta-learner
==============================================================================
"""
import os
import warnings
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    classification_report,
    confusion_matrix,
    f1_score,
)
import lightgbm as lgb
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("  [INFO] Optuna not installed. Using default LightGBM hyperparameters.")

import joblib

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from shared_config import MODEL_REPORTS_DIR, MODELS_DIR, load_engineered_data

# ── CONFIG ───────────────────────────────────────────────────────────────────
REPORTS_DIR  = MODEL_REPORTS_DIR
TARGET_COL   = "F3924"
N_SPLITS     = 5
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 40   # set higher (e.g. 100) for better hyperparameters

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("Loading Engineered Dataset")
print("=" * 70)
df = load_engineered_data()
feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].values.astype(np.float32)
y = df[TARGET_COL].values.astype(int)
print(f"  Shape  : {X.shape}")
print(f"  Pos (Mule)  : {y.sum():,} ({100*y.mean():.3f}%)")
print(f"  Neg (Legit) : {(y == 0).sum():,}")
scale_weight = (y == 0).sum() / (y == 1).sum()
print(f"  Scale Pos Weight: {scale_weight:.2f}")

# ── STRATIFIED K-FOLD SETUP ──────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ────────────────────────────────────────────────────────────────────────────
# Utility: Cross-validation with out-of-fold collection
# ────────────────────────────────────────────────────────────────────────────
def cross_validate_model(model, X, y, skf, model_name="Model"):
    """Runs stratified K-fold and returns OOF predictions + per-fold metrics."""
    oof_preds  = np.zeros(len(y), dtype=np.float64)
    fold_aucs  = []
    fold_aps   = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = proba
        fold_aucs.append(roc_auc_score(y_val, proba))
        fold_aps.append(average_precision_score(y_val, proba))
        print(f"    Fold {fold+1}/{N_SPLITS}  ROC-AUC={fold_aucs[-1]:.4f}  PR-AUC={fold_aps[-1]:.4f}")
    mean_auc = np.mean(fold_aucs)
    mean_ap  = np.mean(fold_aps)
    print(f"  ── {model_name} CV Summary: ROC-AUC={mean_auc:.4f}±{np.std(fold_aucs):.4f}  "
          f"PR-AUC={mean_ap:.4f}±{np.std(fold_aps):.4f}")
    return oof_preds, mean_auc, mean_ap


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: LightGBM (with optional Optuna HPO)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 1: LightGBM Classifier")
print("=" * 70)

DEFAULT_LGBM_PARAMS = {
    'n_estimators'     : 1000,
    'learning_rate'    : 0.05,
    'num_leaves'       : 63,
    'max_depth'        : -1,
    'min_child_samples': 50,
    'feature_fraction' : 0.8,
    'bagging_fraction' : 0.8,
    'bagging_freq'     : 5,
    'reg_alpha'        : 0.1,
    'reg_lambda'       : 1.0,
    'scale_pos_weight' : scale_weight,
    'objective'        : 'binary',
    'metric'           : 'average_precision',
    'n_jobs'           : -1,
    'random_state'     : RANDOM_STATE,
    'verbose'          : -1,
}

if OPTUNA_AVAILABLE:
    print("  Running Optuna HPO for LightGBM...")
    def lgbm_objective(trial):
        params = {
            'n_estimators'       : trial.suggest_int('n_estimators', 300, 2000),
            'learning_rate'      : trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves'         : trial.suggest_int('num_leaves', 20, 200),
            'max_depth'          : trial.suggest_int('max_depth', 3, 10),
            'min_child_samples'  : trial.suggest_int('min_child_samples', 20, 200),
            'feature_fraction'   : trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction'   : trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq'       : trial.suggest_int('bagging_freq', 1, 10),
            'reg_alpha'          : trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda'         : trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'scale_pos_weight'   : scale_weight,
            'objective'          : 'binary',
            'metric'             : 'average_precision',
            'n_jobs'             : -1,
            'random_state'       : RANDOM_STATE,
            'verbose'            : -1,
        }
        model = lgb.LGBMClassifier(**params)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[val_idx])[:, 1]
            scores.append(average_precision_score(y[val_idx], proba))
        return np.mean(scores)

    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lgbm_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    best_lgbm_params = study_lgbm.best_params
    best_lgbm_params.update({'objective': 'binary', 'metric': 'average_precision',
                              'scale_pos_weight': scale_weight, 'n_jobs': -1,
                              'random_state': RANDOM_STATE, 'verbose': -1})
    print(f"  Best Optuna params: {best_lgbm_params}")
    with open(f"{MODELS_DIR}/lgbm_best_params.json", 'w') as f:
        json.dump(best_lgbm_params, f, indent=2)
    lgbm_model = lgb.LGBMClassifier(**best_lgbm_params)
else:
    lgbm_model = lgb.LGBMClassifier(**DEFAULT_LGBM_PARAMS)

oof_lgbm, auc_lgbm, ap_lgbm = cross_validate_model(lgbm_model, X, y, skf, "LightGBM")

# Refit final model on full training data for production
lgbm_model.fit(X, y)
joblib.dump(lgbm_model, f"{MODELS_DIR}/lgbm_final.pkl")
print(f"  [Saved] {MODELS_DIR}/lgbm_final.pkl")

# Feature Importance
feat_importance = pd.Series(
    lgbm_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)
feat_importance.to_csv(f"{REPORTS_DIR}/lgbm_feature_importance.csv", header=['importance'])
print(f"  [Saved] {REPORTS_DIR}/lgbm_feature_importance.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: XGBoost Classifier
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 2: XGBoost Classifier")
print("=" * 70)
xgb_model = xgb.XGBClassifier(
    n_estimators        = 800,
    learning_rate       = 0.05,
    max_depth           = 6,
    min_child_weight    = 5,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    reg_alpha           = 0.1,
    reg_lambda          = 1.0,
    scale_pos_weight    = scale_weight,
    eval_metric         = 'aucpr',
    use_label_encoder   = False,
    tree_method         = 'hist',
    random_state        = RANDOM_STATE,
    n_jobs              = 1,
)
oof_xgb, auc_xgb, ap_xgb = cross_validate_model(xgb_model, X, y, skf, "XGBoost")
xgb_model.fit(X, y)
joblib.dump(xgb_model, f"{MODELS_DIR}/xgb_final.pkl")
print(f"  [Saved] {MODELS_DIR}/xgb_final.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: Random Forest Classifier
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 3: Random Forest Classifier")
print("=" * 70)
rf_model = RandomForestClassifier(
    n_estimators    = 500,
    max_depth       = 12,
    min_samples_leaf= 10,
    max_features    = 'sqrt',
    class_weight    = 'balanced',
    n_jobs          = 1,
    random_state    = RANDOM_STATE,
)
oof_rf, auc_rf, ap_rf = cross_validate_model(rf_model, X, y, skf, "RandomForest")
rf_model.fit(X, y)
joblib.dump(rf_model, f"{MODELS_DIR}/rf_final.pkl")
print(f"  [Saved] {MODELS_DIR}/rf_final.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: Logistic Regression (Baseline)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 4: Logistic Regression Baseline")
print("=" * 70)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr_model = LogisticRegression(
    C            = 0.1,
    solver       = 'saga',
    class_weight = 'balanced',
    max_iter     = 1000,
    n_jobs       = 1,
    random_state = RANDOM_STATE,
)
oof_lr, auc_lr, ap_lr = cross_validate_model(lr_model, X_scaled, y, skf, "LogisticRegression")
lr_model.fit(X_scaled, y)
joblib.dump(lr_model, f"{MODELS_DIR}/lr_final.pkl")
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
print(f"  [Saved] {MODELS_DIR}/lr_final.pkl, {MODELS_DIR}/scaler.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 5: Stacking Ensemble
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 5: Stacking Ensemble (Meta-Learner over OOF Predictions)")
print("=" * 70)
# Stack OOF predictions from models 1-4 as new feature matrix for meta-learner
meta_X = np.column_stack([oof_lgbm, oof_xgb, oof_rf, oof_lr])
meta_learner = LogisticRegression(
    C=1.0, class_weight='balanced', random_state=RANDOM_STATE
)
oof_meta_preds = np.zeros(len(y))
fold_aucs_meta = []
fold_aps_meta  = []
for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, y)):
    meta_learner.fit(meta_X[train_idx], y[train_idx])
    proba_meta = meta_learner.predict_proba(meta_X[val_idx])[:, 1]
    oof_meta_preds[val_idx] = proba_meta
    fold_aucs_meta.append(roc_auc_score(y[val_idx], proba_meta))
    fold_aps_meta.append(average_precision_score(y[val_idx], proba_meta))
    print(f"    Fold {fold+1}/{N_SPLITS}  ROC-AUC={fold_aucs_meta[-1]:.4f}  PR-AUC={fold_aps_meta[-1]:.4f}")
auc_meta = np.mean(fold_aucs_meta)
ap_meta  = np.mean(fold_aps_meta)
print(f"  ── Stacking CV Summary: ROC-AUC={auc_meta:.4f}  PR-AUC={ap_meta:.4f}")
meta_learner.fit(meta_X, y)
joblib.dump(meta_learner, f"{MODELS_DIR}/stacking_meta_learner.pkl")
print(f"  [Saved] {MODELS_DIR}/stacking_meta_learner.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON REPORT: All Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL COMPARISON REPORT")
print("=" * 70)
comparison = pd.DataFrame({
    'Model'   : ['LightGBM', 'XGBoost', 'RandomForest', 'LogisticRegression', 'Stacking Ensemble'],
    'ROC-AUC' : [auc_lgbm, auc_xgb, auc_rf, auc_lr, auc_meta],
    'PR-AUC'  : [ap_lgbm, ap_xgb, ap_rf, ap_lr, ap_meta],
})
comparison = comparison.sort_values('PR-AUC', ascending=False)
print(comparison.to_string(index=False))
comparison.to_csv(f"{REPORTS_DIR}/model_comparison.csv", index=False)

# ── VISUALIZATION: PR Curves for All Models ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
model_configs = [
    ('LightGBM',           oof_lgbm,      '#E53935', '--'),
    ('XGBoost',            oof_xgb,       '#1E88E5', '-.'),
    ('RandomForest',       oof_rf,        '#43A047', ':'),
    ('LogisticRegression', oof_lr,        '#FB8C00', (0, (3, 1, 1, 1))),
    ('Stacking Ensemble',  oof_meta_preds,'#8E24AA', '-'),
]

# PR Curve
ax = axes[0]
for name, oof_prob, color, ls in model_configs:
    prec, rec, _ = precision_recall_curve(y, oof_prob)
    ap = average_precision_score(y, oof_prob)
    ax.plot(rec, prec, color=color, linestyle=ls, linewidth=2.0, label=f'{name} (AP={ap:.4f})')
baseline_precision = y.mean()
ax.axhline(y=baseline_precision, color='gray', linestyle='--', linewidth=1, label=f'Random Baseline (AP={baseline_precision:.4f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve (OOF)', fontweight='bold', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ROC Curve
ax = axes[1]
for name, oof_prob, color, ls in model_configs:
    fpr, tpr, _ = roc_curve(y, oof_prob)
    auc_val = roc_auc_score(y, oof_prob)
    ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2.0, label=f'{name} (AUC={auc_val:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Baseline')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve (OOF)', fontweight='bold', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.suptitle('Model Evaluation — Mule Account Detection (F3924)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/model_comparison_curves.png", dpi=150)
plt.close()
print(f"  [Saved] {REPORTS_DIR}/model_comparison_curves.png")

# ── VISUALIZATION: LightGBM Feature Importance (Top 30) ───────────────────────
top_feat = feat_importance.head(30)
fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(top_feat.index[::-1], top_feat.values[::-1], color='#5C6BC0', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Importance Score (LightGBM)', fontsize=12)
ax.set_title('Top 30 Feature Importance — LightGBM', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/lgbm_top30_feature_importance.png", dpi=150)
plt.close()
print(f"  [Saved] {REPORTS_DIR}/lgbm_top30_feature_importance.png")

# ── VISUALIZATION: Confusion Matrix + Threshold Selection (Best Model) ─────────
best_oof  = oof_lgbm if ap_lgbm >= max(ap_xgb, ap_rf, ap_lr, ap_meta) else oof_meta_preds
best_name = "LightGBM" if ap_lgbm >= max(ap_xgb, ap_rf, ap_lr, ap_meta) else "Stacking Ensemble"
print(f"\n  Best Model by PR-AUC: {best_name}")

# Find optimal threshold via F1 maximization
thresholds = np.linspace(0.01, 0.99, 99)
f1_scores  = [f1_score(y, (best_oof >= t).astype(int), zero_division=0) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]
optimal_f1        = max(f1_scores)
print(f"  Optimal Decision Threshold (Max-F1): {optimal_threshold:.3f}  →  F1={optimal_f1:.4f}")

y_pred_optimal = (best_oof >= optimal_threshold).astype(int)
cm = confusion_matrix(y, y_pred_optimal)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Predicted Legit', 'Predicted Mule'],
            yticklabels=['Actual Legit', 'Actual Mule'],
            annot_kws={"size": 14})
axes[0].set_title(f'Confusion Matrix — {best_name} @ threshold={optimal_threshold:.3f}', fontweight='bold')

axes[1].plot(thresholds, f1_scores, color='#E53935', linewidth=2)
axes[1].axvline(x=optimal_threshold, color='navy', linestyle='--', linewidth=1.5, label=f'Optimal @ {optimal_threshold:.3f}')
axes[1].set_xlabel('Decision Threshold', fontsize=12)
axes[1].set_ylabel('F1 Score', fontsize=12)
axes[1].set_title('F1 Score vs Decision Threshold', fontweight='bold', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/confusion_matrix_threshold.png", dpi=150)
plt.close()
print(f"  [Saved] {REPORTS_DIR}/confusion_matrix_threshold.png")

# Final classification report
print("\n  Classification Report @ Optimal Threshold:")
print(classification_report(y, y_pred_optimal, target_names=['Legitimate', 'Mule/Suspicious']))

# Save metadata
metadata = {
    "best_model"         : best_name,
    "optimal_threshold"  : float(optimal_threshold),
    "best_roc_auc"       : float(roc_auc_score(y, best_oof)),
    "best_pr_auc"        : float(average_precision_score(y, best_oof)),
    "optimal_f1"         : float(optimal_f1),
}
with open(f"{MODELS_DIR}/best_model_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  [Saved] {MODELS_DIR}/best_model_metadata.json")

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)
