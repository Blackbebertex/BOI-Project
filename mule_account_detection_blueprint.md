# Comprehensive ML Engineering Blueprint
# AI/ML Mule Account & Suspicious Transaction Detection System

> **Domain:** Cyber-enabled financial fraud prevention  
> **Target Variable:** `F3924` (Binary: `1` = Suspicious/Mule Account, `0` = Legitimate)  
> **Bank-Specified Key Features:** F115, F321, F527, F531, F670, F1692, F2082, F2122, F2582, F2678, F2737, F2956, F3043, F3836, F3887, F3889, F3891, F3894

---

## Table of Contents

1. [Problem Framing & Fraud Typologies](#1-problem-framing)
2. [Dataset Structure & Schema Assumptions](#2-dataset-structure)
3. [End-to-End Pipeline Architecture](#3-pipeline-architecture)
4. [Exploratory Data Analysis (EDA) Plan](#4-eda-plan)
5. [Feature Engineering Strategy](#5-feature-engineering)
6. [Model Selection & Training Protocol](#6-model-training)
7. [Anomaly Detection Layer](#7-anomaly-detection)
8. [Model Explainability (SHAP)](#8-explainability)
9. [Risk Scoring & Decision Matrix](#9-risk-scoring)
10. [Evaluation Metrics Guide](#10-evaluation-metrics)
11. [Model Governance & Drift Monitoring](#11-model-governance)
12. [Real-Time Serving Architecture](#12-serving)
13. [Project File Structure](#13-file-structure)
14. [Execution Order & Run Guide](#14-execution-guide)

---

## 1. Problem Framing

### 1.1. Fraud Typologies Captured

Three primary mule account behavioural patterns are targetted:

| Typology | Description | Signature Pattern |
| :--- | :--- | :--- |
| **Instant Mule** | Receives funds and immediately routes them out | Out-In ratio > 0.9 within 15 minutes |
| **Sleeper Mule** | Dormant account suddenly activated for large transactions | 0 TX for 90 days → large credit in |
| **Aggregator Hub** | Receives dozens of micro-transactions, consolidates, and routes | High in-degree, low out-degree, then large single debit |

### 1.2. Class Imbalance Reality

In production banking data, mule/suspicious accounts represent a **very small fraction** of total accounts:

$$\text{Fraud Rate} \approx 0.05\% \text{ to } 0.5\%$$

$$\text{Imbalance Ratio (Negative:Positive)} \approx 200:1 \text{ to } 2000:1$$

This forces specific modeling strategies — **standard accuracy is meaningless** at this scale.

### 1.3. The Right Problem Statement

> *"At a 1% False Positive Rate (100 legitimate accounts incorrectly flagged per 10,000 reviewed), what is our maximum achievable Recall (fraction of all real mule accounts caught)?"*

This framing — **Recall at Controlled FPR** — aligns with operational reality: fraud investigators have a fixed daily capacity to review alerts.

---

## 2. Dataset Structure

### 2.1. Feature Naming Convention

The dataset uses a flat feature naming scheme `F1` through `F3924`, where:

- `F3924` = **Target variable** (must be isolated as `y`)
- All other columns (`F1`–`F3923`) = **Feature candidates**
- Total potential features = **3923 raw features**

### 2.2. Bank-Specified Important Features

These 18 features were explicitly identified by the bank as commonly used for fraud detection. They **must be preserved through all pruning steps** and used as anchor features in engineering:

```
F115  F321  F527  F531  F670  F1692  F2082  F2122
F2582 F2678 F2737 F2956 F3043 F3836  F3887  F3889
F3891 F3894
```

> [!IMPORTANT]
> These features should NEVER be dropped by automated variance or missing-value filtering steps, regardless of their individual statistics.

### 2.3. Schema Assumptions

Since the features are anonymized (`F1`–`F3923`), we infer feature semantics from the bank's named features:

| Feature ID | Inferred Domain | Justification |
| :--- | :--- | :--- |
| `F115`, `F321` | Account profile / KYC attributes | Low cardinality, likely static |
| `F527`, `F531` | Transaction count features | Volume indicators |
| `F670` | Balance or amount-level feature | Likely right-skewed |
| `F1692`, `F2082`, `F2122` | Short/medium window transaction aggregates | Velocity indicators |
| `F2582`, `F2678`, `F2737` | Cross-period amount ratios | Behavioral change signals |
| `F2956`, `F3043` | Channel-specific activity | Multi-channel usage |
| `F3836`, `F3887`, `F3889`, `F3891`, `F3894` | Terminal / device-level features | Device fingerprint signals |

---

## 3. End-to-End Pipeline Architecture

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                     RAW DATA (DataSet.csv)                            │
  │                    3923 features + F3924 target                      │
  └─────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │                 Phase 0: EDA (00_eda_exploration.py)                 │
  │   • Target distribution          • Missing value audit              │
  │   • Bank key feature KDE plots   • Target correlation ranking       │
  └─────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │             Phase 1: Feature Engineering (01_feature_engineering.py)│
  │   • Drop >50% missing            • Remove near-zero variance        │
  │   • Mutual Information selection • Interaction feature creation      │
  │   • Log / Quantile transforms    • KMeans cluster features          │
  └─────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │              Phase 2: Model Training (02_model_training.py)          │
  │   • LightGBM (+ Optuna HPO)      • XGBoost                         │
  │   • Random Forest                • Logistic Regression              │
  │   • Stacking Ensemble            • PR Curve & ROC-AUC comparison    │
  └─────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │            Phase 3: Anomaly Detection (03_anomaly_detection.py)      │
  │   • Isolation Forest             • ECOD (PyOD)                      │
  │   • Autoencoder (PyTorch)        • Risk Score Fusion                │
  └─────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │            Phase 4: Explainability (04_shap_explainability.py)       │
  │   • SHAP TreeExplainer           • Global importance bar + beeswarm │
  │   • Waterfall plots per account  • Case narratives for investigators │
  └─────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │              Phase 5: Model Serving (serving/app.py)                 │
  │   • FastAPI REST API             • /score single + /score/batch      │
  │   • Fused risk score             • Decision matrix (BLOCK/CHALLENGE) │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Exploratory Data Analysis (EDA) Plan

Run: [`notebooks/00_eda_exploration.py`](file:///c:/Users/Admin/BOI-Project/notebooks/00_eda_exploration.py)

### 4.1. Target Variable Analysis

Check the class distribution of `F3924`. Expected output:

```
Class 0 (Legitimate) : ~99.5% of records
Class 1 (Mule/Suspicious): ~0.5% of records
Imbalance Ratio (Neg:Pos) ≈ 199:1
```

> [!WARNING]
> If the imbalance ratio is > 500:1, SMOTE will likely hurt performance. Rely solely on `scale_pos_weight` and Focal Loss instead.

### 4.2. Missing Value Audit

```
Target Action for Missing Rate:
  > 90% missing → DROP (useless signal; just noise after imputation)
  50% – 90%    → Evaluate MI score; drop if MI < median
  < 50%        → Keep; impute with median (training data median)
```

### 4.3. Bank Key Feature Deep-Dive

For each of the 18 bank-specified features, generate:
1. **KDE plot** comparing the distribution of the feature for Class 0 vs Class 1
2. **Box plot** showing median, IQR, and outlier spread per class
3. **Statistical test**: Mann-Whitney U statistic and p-value (since features are likely non-normal)

A feature with **visually separated KDE distributions** between the two classes is a strong discriminator.

### 4.4. Target Correlation Ranking

Compute `|Pearson Correlation|` of every numeric feature against `F3924`. This provides a quick rank-ordered signal list before Mutual Information (which is slower but handles nonlinear relationships).

---

## 5. Feature Engineering Strategy

Run: [`notebooks/01_feature_engineering.py`](file:///c:/Users/Admin/BOI-Project/notebooks/01_feature_engineering.py)

### 5.1. Pruning Steps (Applied in Order)

```
Step 1: Drop features with > 50% missing values
         → (Bank key features are exempt from this rule)

Step 2: Median imputation of all remaining NaN values
         → Fitted on training set only; applied to validation/test

Step 3: Remove near-zero variance features (var < 1e-4)
         → (Bank key features are exempt)

Step 4: Mutual Information Feature Selection
         → Retain Top 200 features by MI score
         → Always include all 18 bank-specified features regardless of MI score
```

### 5.2. Domain-Driven Interaction Features

These synthesized features directly encode known mule behavioural signals:

| Feature Name | Formula | Fraud Signal |
| :--- | :--- | :--- |
| `FEAT_out_in_count_ratio` | `F527 / (F531 + ε)` | Rapid fund drain ratio |
| `FEAT_short_long_amount_ratio` | `F2082 / (F2122 + ε)` | Short-window vs long-window volume burst |
| `FEAT_net_fund_flow` | `F2678 - F2737` | Net fund movement (large negative = drain) |
| `FEAT_age_x_freq` | `F115 × F321` | Account age vs activity rate joint signal |
| `FEAT_amount_deviation` | `|F1692 - F2582|` | Deviation from expected average amount |
| `FEAT_multichannel_signal` | `F2956 × F3043` | Cross-channel activity product |
| `FEAT_terminal_aggregate` | `F3887 + F3889 + F3891 + F3894` | Aggregated terminal-level signals |
| `FEAT_terminal_max` | `max(F3887, F3889, F3891, F3894)` | Peak terminal activity signal |
| `LOG_<feature>` | `log1p(max(feature, 0))` | Normalized skewed feature distributions |

### 5.3. Log Transform for Highly Skewed Features

For all features with `|skewness| > 5`:

$$\text{LOG\_Feature} = \log(1 + \max(\text{feature}, 0))$$

This is applied **after** MI selection and **only for the auto-detected high-skew set**, preventing over-transformation.

### 5.4. KMeans Cluster Features

Train KMeans ($K=5$ clusters) on the **top 20 MI features** representing distinct account behavioral archetypes (e.g., "dormant account", "high-volume sender", "new account heavy transactor").

Generated features:
- `FEAT_kmeans_cluster` — Categorical cluster ID (0–4)
- `FEAT_dist_cluster_0` through `FEAT_dist_cluster_4` — Distance from each centroid

> [!TIP]
> A suspicious account that falls far from ALL cluster centroids (high distance to all clusters) may be a novel mule typology not seen in training.

### 5.5. Data Leakage Prevention

| Leakage Risk | Prevention Strategy |
| :--- | :--- |
| Imputer fitting on full dataset | Fit `SimpleImputer` only on training rows; transform val/test |
| QuantileTransformer fitting | Fit only on training set |
| KMeans fitting | Fit only on training set |
| Feature selection MI score | Compute MI only on training set |
| Target encoding categorical features | Use cross-validated target encoding (not implemented here; add if cat features exist) |

---

## 6. Model Selection & Training Protocol

Run: [`notebooks/02_model_training.py`](file:///c:/Users/Admin/BOI-Project/notebooks/02_model_training.py)

### 6.1. Model Portfolio

| Model | Library | Class Imbalance Handling | Role |
| :--- | :--- | :--- | :--- |
| **LightGBM** | `lightgbm` | `scale_pos_weight` | Primary model; fast + accurate |
| **XGBoost** | `xgboost` | `scale_pos_weight` | Complementary learner |
| **Random Forest** | `sklearn` | `class_weight='balanced'` | Diversity via bagging |
| **Logistic Regression** | `sklearn` | `class_weight='balanced'` | Linear baseline + calibration |
| **Stacking Ensemble** | Custom | LR meta-learner on OOF | Reduce bias of individual models |

### 6.2. Cross-Validation Protocol

**Stratified K-Fold** (K=5) ensures each fold has the same proportion of mule accounts:

```
Fold 1: [████████████████░░░░] → Train: 80%  |  Val: 20%
Fold 2: [████░░░░████████████] → Train: 80%  |  Val: 20%
...
Fold 5: [░░░░████████████████] → Train: 80%  |  Val: 20%
```

**Out-of-Fold (OOF) Predictions** are collected across all folds to:
1. Compute unbiased model-level PR-AUC and ROC-AUC estimates
2. Serve as the meta-feature matrix for the stacking ensemble

### 6.3. LightGBM Hyperparameter Search (Optuna)

Optuna runs Bayesian optimization (Tree-structured Parzen Estimator) over:

| Hyperparameter | Search Range | Meaning |
| :--- | :--- | :--- |
| `n_estimators` | 300 – 2000 | Number of boosting rounds |
| `learning_rate` | 0.01 – 0.15 (log) | Step size shrinkage |
| `num_leaves` | 20 – 200 | Model complexity (controls overfitting) |
| `max_depth` | 3 – 10 | Limit tree depth |
| `min_child_samples` | 20 – 200 | Minimum samples per leaf (regularization) |
| `feature_fraction` | 0.5 – 1.0 | Column sub-sampling per tree |
| `bagging_fraction` | 0.5 – 1.0 | Row sub-sampling per tree |
| `reg_alpha` | 1e-3 – 10 (log) | L1 regularization strength |
| `reg_lambda` | 1e-3 – 10 (log) | L2 regularization strength |

**Objective metric:** PR-AUC (Average Precision), measured on the validation fold.

### 6.4. Class Imbalance Weight Setting

```python
# scale_pos_weight for LightGBM and XGBoost
scale_pos_weight = count(negative_class) / count(positive_class)
#                = count(Legitimate) / count(Mule)
# If dataset has 99,000 legitimate and 1,000 mule accounts:
scale_pos_weight = 99000 / 1000 = 99.0
```

This tells the model that **misclassifying a mule account is 99x more costly** than misclassifying a legitimate one.

### 6.5. Optimal Decision Threshold Selection

The default 0.5 threshold is **never optimal** under class imbalance. We sweep thresholds from 0.01 to 0.99 and select the threshold that **maximizes F1 score** on OOF predictions:

```
Sweep: threshold ∈ [0.01, 0.99] step 0.01
       → For each threshold compute F1(y_true, y_pred_binary)
       → optimal_threshold = threshold at argmax(F1)
```

> [!TIP]
> In practice, use the threshold that achieves the **operational Recall target** (e.g., "catch at least 85% of all mules") and report the resulting FPR to management.

---

## 7. Anomaly Detection Layer

Run: [`notebooks/03_anomaly_detection.py`](file:///c:/Users/Admin/BOI-Project/notebooks/03_anomaly_detection.py)

### 7.1. Why Anomaly Detection Alongside Supervised Models?

The supervised model captures **known labeled fraud patterns**. But fraudsters evolve. New mule typologies produce **no labels initially** — they appear as unseen behaviors. Anomaly detection flags accounts that are statistically unusual relative to the "normal" population.

### 7.2. Method Comparison

| Method | Algorithm | Key Property | Installed |
| :--- | :--- | :--- | :--- |
| **Isolation Forest** | Ensemble of random trees | Anomalies are isolated faster | `sklearn` (always available) |
| **ECOD** | Empirical CDF tails | No hyperparameters; very fast | `pyod` (optional) |
| **Autoencoder** | Neural bottleneck recon. | Captures non-linear manifold | `torch` (optional) |

### 7.3. Semi-Supervised Training Strategy

> **Key insight**: Train anomaly models ONLY on legitimate accounts (`y=0`).

This way, the model learns what "normal" looks like. At inference time, mule accounts **do not conform** to this learned normalcy, producing high reconstruction errors or low decision function scores.

### 7.4. Risk Score Fusion Formula

$$\text{FusedRiskScore} = 0.70 \times P_{\text{supervised}} + 0.30 \times S_{\text{anomaly}}$$

Where:
- $P_{\text{supervised}}$ = LightGBM mule probability (0–1)
- $S_{\text{anomaly}}$ = Isolation Forest anomaly score, inverted and normalized to (0–1)

> [!NOTE]
> Weights (0.70 / 0.30) were selected based on domain reasoning. Tune using Bayesian optimization on a held-out validation set if labelled data is sufficient.

---

## 8. Explainability (SHAP)

Run: [`notebooks/04_shap_explainability.py`](file:///c:/Users/Admin/BOI-Project/notebooks/04_shap_explainability.py)

### 8.1. Why SHAP?

SHAP (SHapley Additive exPlanations) is the **only method that satisfies the three key fairness axioms** for explanations:
1. **Efficiency:** SHAP values of all features sum to the model's output.
2. **Symmetry:** Two features with identical contribution get identical SHAP values.
3. **Dummy:** A feature that has no impact on the model always has SHAP value = 0.

### 8.2. Outputs Generated

| Output | File | Usage |
| :--- | :--- | :--- |
| Global importance bar chart | `shap_global_importance_bar.png` | Model documentation, regulator reports |
| Beeswarm impact plot | `shap_beeswarm.png` | Understand feature direction of impact |
| Waterfall: True Positive | `shap_waterfall_true_positive.png` | Show investigator WHY an account was flagged |
| Waterfall: False Positive | `shap_waterfall_false_positive.png` | Understand analyst appeal case |
| Per-account CSV | `per_account_shap_explanations.csv` | Feed to case management portal |

### 8.3. Sample Investigator Narrative

```
Risk Level     : CRITICAL
Model Score    : 0.9241
Actual Label   : MULE/SUSPICIOUS

Top Risk Drivers (Factors Increasing Mule Score):
  + FEAT_out_in_count_ratio             SHAP=+0.3412
  + F527                                SHAP=+0.2187
  + FEAT_dormancy_break_factor          SHAP=+0.1943

Top Mitigating Factors (Factors Reducing Mule Score):
  - F115                                SHAP=-0.0821
  - F321                                SHAP=-0.0412
  - FEAT_log_F670                       SHAP=-0.0218
```

This narrative format is directly suitable for **investigator portal integration** and **regulatory reporting**.

---

## 9. Risk Scoring & Decision Matrix

### 9.1. Four-Tier Decision Matrix

| Decision | Fused Score Range | Action | Operational Meaning |
| :--- | :--- | :--- | :--- |
| **BLOCK** | ≥ 0.85 | Immediate freeze of outbound transactions | Very high confidence mule; requires no human review to block |
| **CHALLENGE** | 0.65 – 0.84 | Step-up authentication (OTP / biometric) | High risk; allow transaction only after re-verification |
| **REVIEW** | 0.45 – 0.64 | Flag for fraud investigator queue | Moderate risk; human review within 4 hours |
| **APPROVE** | < 0.45 | Allow transaction | Low risk; no intervention needed |

### 9.2. Regulatory Override Rule

Any account matching an **active government cyber fraud ticket** (`regulatory_alerts` table) is automatically escalated to **BLOCK**, regardless of the ML score:

```python
# Pseudo-code
if account_id in active_govt_flags:
    decision = "BLOCK"
    risk_level = "CRITICAL"
    override_reason = "GOVT_CYBER_FRAUD_TICKET_MATCH"
```

---

## 10. Evaluation Metrics Guide

> [!IMPORTANT]
> **NEVER use Accuracy as your primary metric** for this dataset. With a 99.5% negative class, a model that predicts "Legitimate" for every account achieves 99.5% accuracy — and catches zero mules.

### 10.1. Primary Metrics

| Metric | Formula | Why It Matters |
| :--- | :--- | :--- |
| **PR-AUC** (Average Precision) | Area under Precision-Recall curve | Directly measures model utility under imbalance |
| **ROC-AUC** | Area under ROC curve | Threshold-free, rank-ordering quality |
| **Recall @ FPR=1%** | TPR when FPR=0.01 | Operational constraint: how many mules caught at 1% false alarm rate? |
| **F1 Score (optimal threshold)** | 2·P·R / (P+R) | Balanced single-number summary at operational threshold |

### 10.2. Business-Oriented Reporting

```
At optimal threshold (e.g., 0.42):
  ├── Recall       = 84.3%  → "We catch 84 out of every 100 mule accounts"
  ├── Precision    = 31.2%  → "When we flag an account, 31% are true mules"
  ├── FPR          = 0.8%   → "We incorrectly flag 8 per 1,000 legitimate accounts"
  └── Daily alerts = ~320   → "Investigators need to review 320 cases per day"
```

### 10.3. Baseline Comparisons

| Baseline | PR-AUC | Interpretation |
| :--- | :--- | :--- |
| Random classifier | = fraud rate (e.g., 0.005) | Worst possible |
| Logistic Regression | ~0.15–0.30 | Linear signal capture |
| XGBoost | ~0.35–0.55 | Strong non-linear |
| LightGBM (tuned) | ~0.45–0.65 | Primary benchmark |
| Stacking Ensemble | ~0.50–0.70 | Target: beat LightGBM alone |

---

## 11. Model Governance & Drift Monitoring

### 11.1. Population Stability Index (PSI) Monitoring

Weekly, compute the PSI for the top 20 features:

$$PSI = \sum_{i=1}^{B} \left( P_i - Q_i \right) \cdot \ln\left(\frac{P_i}{Q_i}\right)$$

| PSI Value | Status | Action |
| :--- | :--- | :--- |
| < 0.10 | Stable | No action |
| 0.10 – 0.25 | Moderate Shift | Monitor closely; schedule retraining |
| > 0.25 | Significant Drift | **Immediate retraining trigger** |

### 11.2. Model Performance Monitoring

Track weekly:
- **PR-AUC** on labeled investigation outcomes
- **Alert volume** (abnormal surge = possible fraud wave or model degradation)
- **Block rate** (sudden spike = model drift or campaign)

### 11.3. Retraining Schedule

| Trigger | Action |
| :--- | :--- |
| PSI > 0.25 on any key feature | Emergency retrain |
| Weekly schedule (Airflow DAG) | Retrain on rolling 6-month window |
| Major regulatory change | Manual retrain + feature review |
| New mule typology identified | Add new interaction features + retrain |

---

## 12. Real-Time Serving Architecture

Run: [`serving/app.py`](file:///c:/Users/Admin/BOI-Project/serving/app.py)

### 12.1. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/model/info` | Model version, thresholds, performance metrics |
| `POST` | `/score` | Score a single account (< 100ms) |
| `POST` | `/score/batch` | Score up to 5000 accounts in one call |

### 12.2. Sample API Request

```json
POST /score
{
  "account_id": "ACC_123456789",
  "features": {
    "F115": 1200.0,
    "F321": 5.0,
    "F527": 48.0,
    "F531": 3.0,
    "F670": 150000.0,
    "F1692": 45000.0,
    "F2082": 42000.0,
    "F2122": 8000.0,
    "F2582": 5000.0,
    "F2678": 49000.0,
    "F2737": 47500.0,
    "F2956": 12.0,
    "F3043": 8.0,
    "F3836": 3.0,
    "F3887": 6.0,
    "F3889": 7.0,
    "F3891": 5.0,
    "F3894": 9.0
  }
}
```

### 12.3. Sample API Response

```json
{
  "account_id": "ACC_123456789",
  "risk_score": 0.923100,
  "anomaly_score": 0.781200,
  "fused_risk_score": 0.880270,
  "decision": "BLOCK",
  "risk_level": "CRITICAL",
  "latency_ms": 14.3,
  "model_version": "LightGBM",
  "scored_at": "2026-06-08T04:30:00Z"
}
```

---

## 13. Project File Structure

```
BOI-Project/
│
├── data/
│   ├── DataSet.csv                   ← Raw input dataset (repo root)
│   └── engineered/
│       └── transactions_engineered.parquet
│
├── notebooks/
│   ├── 00_eda_exploration.py         ← Phase 0: EDA
│   ├── 01_feature_engineering.py     ← Phase 1: Feature engineering
│   ├── 02_model_training.py          ← Phase 2: Model training & evaluation
│   ├── 03_anomaly_detection.py       ← Phase 3: Anomaly detection
│   └── 04_shap_explainability.py     ← Phase 4: SHAP explainability
│
├── models/
│   ├── lgbm_final.pkl                ← Trained LightGBM
│   ├── xgb_final.pkl                 ← Trained XGBoost
│   ├── rf_final.pkl                  ← Trained Random Forest
│   ├── lr_final.pkl                  ← Trained Logistic Regression
│   ├── stacking_meta_learner.pkl     ← Stacking meta-classifier
│   ├── isolation_forest.pkl          ← Anomaly detector
│   ├── autoencoder_state.pt          ← PyTorch autoencoder (if available)
│   ├── robust_scaler_anomaly.pkl     ← Scaler for anomaly models
│   ├── scaler.pkl                    ← Scaler for logistic regression
│   ├── lgbm_best_params.json         ← Optuna best hyperparameters
│   └── best_model_metadata.json      ← Optimal threshold, AUC scores
│
├── reports/
│   ├── eda/
│   │   ├── 01_target_distribution.png
│   │   ├── 02_missing_values.png
│   │   ├── 03_key_features_kde_by_class.png
│   │   ├── 04_correlation_heatmap.png
│   │   ├── 05_top50_target_correlation.png
│   │   ├── missing_value_report.csv
│   │   └── top50_feature_target_correlation.csv
│   ├── features/
│   │   ├── mutual_information_scores.csv
│   │   └── selected_feature_list.csv
│   ├── models/
│   │   ├── model_comparison.csv
│   │   ├── model_comparison_curves.png
│   │   ├── lgbm_top30_feature_importance.png
│   │   ├── lgbm_feature_importance.csv
│   │   ├── confusion_matrix_threshold.png
│   │   ├── risk_score_distributions.png
│   │   └── risk_scores_all_accounts.csv
│   └── explainability/
│       ├── shap_global_importance_bar.png
│       ├── shap_beeswarm.png
│       ├── shap_waterfall_true_positive.png
│       ├── shap_waterfall_false_positive.png
│       └── per_account_shap_explanations.csv
│
├── serving/
│   └── app.py                        ← FastAPI model serving microservice
│
├── mule_account_detection_blueprint.md ← This document
└── requirements.txt                  ← Python dependencies
```

---

## 14. Execution Order & Run Guide

### 14.1. Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Create data directory and place your dataset
mkdir data
# Place your DataSet.csv in: BOI-Project/DataSet.csv
```

### 14.2. Run Pipeline in Order

```bash
# ── Phase 0: EDA ─────────────────────────────
python notebooks/00_eda_exploration.py
# Output: reports/eda/ (charts + CSVs)

# ── Phase 1: Feature Engineering ─────────────
python notebooks/01_feature_engineering.py
# Output: data/engineered/transactions_engineered.parquet
#         reports/features/

# ── Phase 2: Model Training ───────────────────
python notebooks/02_model_training.py
# Output: models/*.pkl, reports/models/

# ── Phase 3: Anomaly Detection ────────────────
python notebooks/03_anomaly_detection.py
# Output: models/isolation_forest.pkl, models/autoencoder_state.pt
#         reports/models/risk_scores_all_accounts.csv

# ── Phase 4: Explainability ───────────────────
python notebooks/04_shap_explainability.py
# Output: reports/explainability/

# ── Phase 5: Start API Server ─────────────────
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

### 14.3. Update Data Path

Before running, update the `DATA_PATH` variable in each script to point to your actual data file:

```python
DATA_PATH = "../DataSet.csv"  # Update this in all notebooks
```

---

## Quick Reference: Key Design Decisions

| Decision | Choice | Rationale |
| :--- | :--- | :--- |
| Primary metric | PR-AUC | Only metric meaningful under extreme class imbalance |
| Imbalance handling | scale_pos_weight + Focal Loss | Avoids synthetic oversampling artifacts |
| Cross-validation | Stratified K-Fold (K=5) | Ensures equal class proportions per fold |
| Time-based split | Required for production | Prevents chronological data leakage |
| HPO | Optuna (Bayesian) | More efficient than GridSearch on 3923 features |
| Anomaly model training | Only on negative class | Learns "normal"; mules deviate from this |
| Feature selection | MI + Bank key features | Combines statistical signal + domain knowledge |
| SHAP explainer | TreeExplainer | 100x faster than KernelSHAP for tree models |
| Threshold selection | Max-F1 sweep | Optimal operating point for fraud investigators |
