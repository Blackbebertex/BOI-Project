"""
Phase 2: Multi-Model Zoo + Precision-Optimized Selection
"""
import json
import os
import sys
import warnings

import joblib
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from serving.stacking_bundle import StackingBundle
from shared_config import MODEL_REPORTS_DIR, MODELS_DIR, load_engineered_data, load_holdout_data

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except (ImportError, ValueError, ModuleNotFoundError):
    CATBOOST_AVAILABLE = False

REPORTS_DIR = MODEL_REPORTS_DIR
TARGET_COL = "F3924"
N_SPLITS = 5
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 15
MIN_RECALL_FOR_PRECISION = 0.80

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def cross_validate_model(model, X, y, skf, model_name="Model"):
    oof_preds = np.zeros(len(y), dtype=np.float64)
    fold_aucs, fold_aps = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[val_idx])[:, 1]
        oof_preds[val_idx] = proba
        fold_aucs.append(roc_auc_score(y[val_idx], proba))
        fold_aps.append(average_precision_score(y[val_idx], proba))
        print(f"    Fold {fold+1}/{N_SPLITS}  ROC-AUC={fold_aucs[-1]:.4f}  PR-AUC={fold_aps[-1]:.4f}")
    print(
        f"  -- {model_name} CV: ROC-AUC={np.mean(fold_aucs):.4f}  PR-AUC={np.mean(fold_aps):.4f}"
    )
    return oof_preds, float(np.mean(fold_aucs)), float(np.mean(fold_aps))


def recall_at_fpr(y_true, y_prob, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def precision_optimal_threshold(y_true, y_prob, min_recall=MIN_RECALL_FOR_PRECISION):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_prec, best_rec = 0.5, 0.0, 0.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if rec >= min_recall and prec > best_prec:
            best_prec, best_rec, best_t = prec, rec, t
    if best_prec == 0.0:
        f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
        best_t = thresholds[int(np.argmax(f1s))]
        pred = (y_prob >= best_t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        best_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        best_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(best_t), float(best_prec), float(best_rec)


def holdout_metrics_for_model(model, X_holdout, y_holdout, needs_scale=False, scaler=None):
    Xh = scaler.transform(X_holdout) if needs_scale and scaler is not None else X_holdout
    proba = model.predict_proba(Xh)[:, 1]
    t, prec, rec = precision_optimal_threshold(y_holdout, proba)
    return {
        "holdout_roc_auc": float(roc_auc_score(y_holdout, proba)),
        "holdout_pr_auc": float(average_precision_score(y_holdout, proba)),
        "holdout_precision": prec,
        "holdout_recall": rec,
        "recall_at_fpr_1pct": recall_at_fpr(y_holdout, proba),
        "precision_optimal_threshold": t,
        "proba": proba,
    }


print("=" * 70)
print("Loading Engineered Dataset")
print("=" * 70)
df = load_engineered_data()
feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].values.astype(np.float32)
y = df[TARGET_COL].values.astype(int)
scale_weight = (y == 0).sum() / max((y == 1).sum(), 1)
print(f"  Train shape: {X.shape}  positives: {y.sum()}")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

trained = {}
oof_map = {}

# LightGBM
print("\n" + "=" * 70)
print("MODEL: LightGBM")
lgbm_params = {
    "n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63,
    "scale_pos_weight": scale_weight, "objective": "binary",
    "metric": "average_precision", "n_jobs": -1, "random_state": RANDOM_STATE, "verbose": -1,
}
if OPTUNA_AVAILABLE:
    def lgbm_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "scale_pos_weight": scale_weight, "objective": "binary",
            "metric": "average_precision", "n_jobs": -1, "random_state": RANDOM_STATE, "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        scores = []
        for tr, va in skf.split(X, y):
            model.fit(X[tr], y[tr])
            scores.append(average_precision_score(y[va], model.predict_proba(X[va])[:, 1]))
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize")
    study.optimize(lgbm_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    lgbm_params.update(study.best_params)
    lgbm_params.update({"objective": "binary", "metric": "average_precision", "scale_pos_weight": scale_weight,
                        "n_jobs": -1, "random_state": RANDOM_STATE, "verbose": -1})
lgbm_model = lgb.LGBMClassifier(**lgbm_params)
oof_lgbm, auc_lgbm, ap_lgbm = cross_validate_model(lgbm_model, X, y, skf, "LightGBM")
lgbm_model.fit(X, y)
joblib.dump(lgbm_model, f"{MODELS_DIR}/lgbm_final.pkl")
trained["LightGBM"] = lgbm_model
oof_map["LightGBM"] = oof_lgbm

# XGBoost
print("\n" + "=" * 70)
print("MODEL: XGBoost")
xgb_model = xgb.XGBClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=6, scale_pos_weight=scale_weight,
    eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=1,
)
oof_xgb, auc_xgb, ap_xgb = cross_validate_model(xgb_model, X, y, skf, "XGBoost")
xgb_model.fit(X, y)
joblib.dump(xgb_model, f"{MODELS_DIR}/xgb_final.pkl")
trained["XGBoost"] = xgb_model
oof_map["XGBoost"] = oof_xgb

# CatBoost
if CATBOOST_AVAILABLE:
    print("\n" + "=" * 70)
    print("MODEL: CatBoost")
    cat_model = CatBoostClassifier(
        iterations=600, learning_rate=0.05, depth=6,
        auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0,
    )
    oof_cat, auc_cat, ap_cat = cross_validate_model(cat_model, X, y, skf, "CatBoost")
    cat_model.fit(X, y)
    joblib.dump(cat_model, f"{MODELS_DIR}/catboost_final.pkl")
    trained["CatBoost"] = cat_model
    oof_map["CatBoost"] = oof_cat

# Random Forest
print("\n" + "=" * 70)
print("MODEL: RandomForest")
rf_model = RandomForestClassifier(
    n_estimators=400, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
)
oof_rf, auc_rf, ap_rf = cross_validate_model(rf_model, X, y, skf, "RandomForest")
rf_model.fit(X, y)
joblib.dump(rf_model, f"{MODELS_DIR}/rf_final.pkl")
trained["RandomForest"] = rf_model
oof_map["RandomForest"] = oof_rf

# Extra Trees
print("\n" + "=" * 70)
print("MODEL: ExtraTrees")
et_model = ExtraTreesClassifier(
    n_estimators=400, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
)
oof_et, auc_et, ap_et = cross_validate_model(et_model, X, y, skf, "ExtraTrees")
et_model.fit(X, y)
joblib.dump(et_model, f"{MODELS_DIR}/extratrees_final.pkl")
trained["ExtraTrees"] = et_model
oof_map["ExtraTrees"] = oof_et

# Decision Tree
print("\n" + "=" * 70)
print("MODEL: DecisionTree")
dt_model = DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=RANDOM_STATE)
oof_dt, auc_dt, ap_dt = cross_validate_model(dt_model, X, y, skf, "DecisionTree")
dt_model.fit(X, y)
joblib.dump(dt_model, f"{MODELS_DIR}/decision_tree_final.pkl")
trained["DecisionTree"] = dt_model
oof_map["DecisionTree"] = oof_dt

# Logistic Regression
print("\n" + "=" * 70)
print("MODEL: LogisticRegression")
lr_model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
oof_lr, auc_lr, ap_lr = cross_validate_model(lr_model, X_scaled, y, skf, "LogisticRegression")
lr_model.fit(X_scaled, y)
joblib.dump(lr_model, f"{MODELS_DIR}/lr_final.pkl")
trained["LogisticRegression"] = lr_model
oof_map["LogisticRegression"] = oof_lr

# MLP
print("\n" + "=" * 70)
print("MODEL: MLP")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64), activation="relu", max_iter=300,
    early_stopping=True, random_state=RANDOM_STATE,
)
oof_mlp, auc_mlp, ap_mlp = cross_validate_model(mlp_model, X_scaled, y, skf, "MLP")
mlp_model.fit(X_scaled, y)
joblib.dump(mlp_model, f"{MODELS_DIR}/mlp_final.pkl")
trained["MLP"] = mlp_model
oof_map["MLP"] = oof_mlp

scaled_model_names = {"LogisticRegression", "MLP"}

# Stacking
print("\n" + "=" * 70)
print("MODEL: Stacking Ensemble")
base_order = list(oof_map.keys())
meta_X = np.column_stack([oof_map[n] for n in base_order])
meta_learner = LogisticRegression(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
oof_meta = np.zeros(len(y))
for fold, (tr, va) in enumerate(skf.split(meta_X, y)):
    meta_learner.fit(meta_X[tr], y[tr])
    oof_meta[va] = meta_learner.predict_proba(meta_X[va])[:, 1]
auc_meta = roc_auc_score(y, oof_meta)
ap_meta = average_precision_score(y, oof_meta)
print(f"  -- Stacking CV: ROC-AUC={auc_meta:.4f}  PR-AUC={ap_meta:.4f}")
meta_learner.fit(meta_X, y)
base_models = [trained[n] for n in base_order]
scaled_indices = [i for i, n in enumerate(base_order) if n in scaled_model_names]
stacking_bundle = StackingBundle(base_models, meta_learner, scaled_indices, scaler)
stacking_bundle.save(f"{MODELS_DIR}/stacking_bundle.pkl")
joblib.dump(meta_learner, f"{MODELS_DIR}/stacking_meta_learner.pkl")
trained["Stacking Ensemble"] = stacking_bundle
oof_map["Stacking Ensemble"] = oof_meta

# CV comparison
cv_rows = []
for name, oof in oof_map.items():
    cv_rows.append({
        "Model": name,
        "ROC-AUC": roc_auc_score(y, oof),
        "PR-AUC": average_precision_score(y, oof),
    })
cv_comparison = pd.DataFrame(cv_rows).sort_values("PR-AUC", ascending=False)
print("\nCV Model Comparison:\n", cv_comparison.to_string(index=False))

# Holdout evaluation
print("\n" + "=" * 70)
print("HOLDOUT EVALUATION")
holdout_rows = []
best_holdout_name = None
best_holdout_ap = -1.0
best_holdout_bundle = None
df_holdout = load_holdout_data()
X_holdout = df_holdout[feature_cols].values.astype(np.float32)
y_holdout = df_holdout[TARGET_COL].values.astype(int)

for name, model in trained.items():
    if name == "Stacking Ensemble":
        hm = holdout_metrics_for_model(model, X_holdout, y_holdout)
    else:
        hm = holdout_metrics_for_model(
            model, X_holdout, y_holdout,
            needs_scale=name in scaled_model_names, scaler=scaler,
        )
    row = {"Model": name, **{k: v for k, v in hm.items() if k != "proba"}}
    holdout_rows.append(row)
    if hm["holdout_pr_auc"] > best_holdout_ap:
        best_holdout_ap = hm["holdout_pr_auc"]
        best_holdout_name = name
        best_holdout_bundle = (model, hm)

holdout_df = pd.DataFrame(holdout_rows).sort_values("holdout_pr_auc", ascending=False)
comparison = cv_comparison.merge(holdout_df, on="Model", how="left")
comparison.to_csv(f"{REPORTS_DIR}/model_comparison.csv", index=False)
print(holdout_df.to_string(index=False))

best_model, best_hm = best_holdout_bundle
if isinstance(best_model, StackingBundle):
    best_model.save(f"{MODELS_DIR}/best_model.pkl")
else:
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")

f1_thresholds = np.linspace(0.01, 0.99, 99)
best_oof = oof_map[best_holdout_name]
f1_scores = [f1_score(y, (best_oof >= t).astype(int), zero_division=0) for t in f1_thresholds]
optimal_f1_threshold = float(f1_thresholds[int(np.argmax(f1_scores))])

metadata = {
    "best_model": best_holdout_name,
    "best_model_type": "stacking" if best_holdout_name == "Stacking Ensemble" else "single",
    "optimal_threshold": optimal_f1_threshold,
    "precision_optimal_threshold": best_hm["precision_optimal_threshold"],
    "best_roc_auc": float(roc_auc_score(y, best_oof)),
    "best_pr_auc": float(average_precision_score(y, best_oof)),
    "optimal_f1": float(max(f1_scores)),
    "holdout_roc_auc": best_hm["holdout_roc_auc"],
    "holdout_pr_auc": best_hm["holdout_pr_auc"],
    "holdout_precision_at_optimal": best_hm["holdout_precision"],
    "holdout_recall_at_optimal": best_hm["holdout_recall"],
    "recall_at_fpr_1pct": best_hm["recall_at_fpr_1pct"],
    "holdout_n_samples": int(len(y_holdout)),
    "holdout_n_positives": int(y_holdout.sum()),
    "model_zoo_comparison": "reports/models/model_comparison.csv",
    "scaled_model_names": list(scaled_model_names),
}
with open(f"{MODELS_DIR}/best_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

feat_importance = pd.Series(lgbm_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
feat_importance.to_csv(f"{REPORTS_DIR}/lgbm_feature_importance.csv", header=["importance"])

# Plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for name, oof in oof_map.items():
    prec, rec, _ = precision_recall_curve(y, oof)
    axes[0].plot(rec, prec, label=f"{name} (AP={average_precision_score(y, oof):.3f})")
axes[0].set_title("PR Curve (OOF)")
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)
for name, oof in oof_map.items():
    fpr, tpr, _ = roc_curve(y, oof)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, oof):.3f})")
axes[1].set_title("ROC Curve (OOF)")
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/model_comparison_curves.png", dpi=150)
plt.close()

print(f"\n  Best model (holdout PR-AUC): {best_holdout_name}")
print(f"  Precision-optimal threshold: {best_hm['precision_optimal_threshold']:.3f}")
print(f"  Holdout precision/recall: {best_hm['holdout_precision']:.3f} / {best_hm['holdout_recall']:.3f}")
print("\nMODEL TRAINING COMPLETE")
