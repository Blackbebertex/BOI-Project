# BOI Mule Account Detection: End-to-End Architecture

This document explains how the project works as one connected system:

- how raw data is prepared
- how missing values are handled
- how models are trained
- how the API loads and scores
- how the browser frontend talks to the backend
- how decisions, overrides, and explanations are produced

The main goal of the project is to score transactions or accounts for mule/fraud risk and turn that score into an operational decision such as `APPROVE`, `REVIEW`, `CHALLENGE`, or `BLOCK`.

## 1. High-Level Flow

```text
DataSet.csv
  -> notebooks/00_eda_exploration.py
  -> notebooks/01_feature_engineering.py
  -> notebooks/02_model_training.py
  -> notebooks/03_anomaly_detection.py
  -> notebooks/04_shap_explainability.py
  -> models/* and reports/*
  -> serving/app.py
  -> frontend/index.html
```

In plain terms:

1. Raw data is inspected and cleaned.
2. Missing values are analyzed and imputed or dropped.
3. Features are ranked and enriched.
4. Supervised models are trained.
5. An anomaly detector is trained.
6. SHAP explainability produces per-account narratives for investigators.
7. The backend loads those artifacts and serves predictions.
8. The frontend uploads CSVs or manual inputs and displays decisions.

## 2. Repository Map

Important files:

- [DataSet.csv](DataSet.csv) — Raw 111 MB dataset (Git‑LFS tracked)
- [notebooks/shared_config.py](notebooks/shared_config.py) — Central path config and data‑loading helpers
- [notebooks/00_eda_exploration.py](notebooks/00_eda_exploration.py) — Phase 0: EDA
- [notebooks/01_feature_engineering.py](notebooks/01_feature_engineering.py) — Phase 1: FeaturePipeline (train-only fit)
- [notebooks/shared_splits.py](notebooks/shared_splits.py) — Stratified 80/20 split indices
- [serving/feature_pipeline.py](serving/feature_pipeline.py) — Serialized FE pipeline (train + serve)
- [notebooks/02_model_training.py](notebooks/02_model_training.py) — Phase 2: Model training
- [notebooks/03_anomaly_detection.py](notebooks/03_anomaly_detection.py) — Phase 3: Anomaly detection
- [notebooks/04_shap_explainability.py](notebooks/04_shap_explainability.py) — Phase 4: SHAP explainability
- [serving/app.py](serving/app.py) — FastAPI backend
- [frontend/index.html](frontend/index.html) — Browser UI
- [reports/features/selected_feature_list.csv](reports/features/selected_feature_list.csv) — Feature contract
- [models/best_model_metadata.json](models/best_model_metadata.json) — Model metadata

## 3. Shared Configuration

All notebooks import paths from [notebooks/shared_config.py](notebooks/shared_config.py) rather than hard‑coding them. This file defines:

- `BASE_DIR` — project root
- `MODELS_DIR`, `REPORTS_DIR`, `DATA_DIR` — standard output directories
- `RAW_DATA_PATH` — resolved path to the raw CSV (checks `DataSet.csv` then `data/transactions.csv`)
- `ENGINEERED_DATA_PATH` — path to the engineered parquet
- `load_engineered_data()` — helper that reads parquet or CSV fallback
- Report subdirectories: `EDA_REPORTS_DIR`, `FEATURE_REPORTS_DIR`, `MODEL_REPORTS_DIR`, `EXPLAINABILITY_REPORTS_DIR`

## 4. Raw Data And Missing Values

The raw dataset is stored in [DataSet.csv](DataSet.csv). It is not used directly by the API. It first goes through the feature pipeline.

### 4.1 EDA Stage

In [notebooks/00_eda_exploration.py](notebooks/00_eda_exploration.py), the project measures how much data is missing per column:

- `df.isnull().sum()`
- `df.isnull().mean()`

This produces missing-value reports and helps identify columns that are too sparse to keep.

### 4.2 Feature Engineering Stage

In [notebooks/01_feature_engineering.py](notebooks/01_feature_engineering.py), missing values are handled in two ways:

1. Columns with more than 50% missing data are dropped.
2. The remaining numeric columns are imputed with median values using `SimpleImputer(strategy='median')`.

This means the model is trained on a cleaned numeric feature matrix rather than raw NaNs.

### 4.3 Serving-Time Feature Engineering (Train/Serve Parity)

At serving time, clients send the **18 bank key features** (and optionally additional raw `F*` columns). The API loads `models/feature_pipeline.pkl` and runs the same transforms used during training:

```python
feat_frame = feature_pipeline.transform_from_bank_keys(features)
```

Missing raw columns are median-imputed using statistics fitted on the **training split only**. Engineered interaction features (`FEAT_*`), log transforms, quantile scaling, and KMeans cluster distances are computed server-side — clients do not need to supply ~260 engineered columns.

**Label leakage audit:** Features with |Pearson r| > 0.50 vs `F3924` (e.g. `F3912`) are excluded during pipeline fitting. See `reports/eda/leakage_audit.csv`.

## 5. Feature Engineering

The feature pipeline is the bridge between raw data and the model.

### 5.1 Feature Pruning

In [notebooks/01_feature_engineering.py](notebooks/01_feature_engineering.py), the pipeline:

- drops columns with too much missingness
- keeps the bank's required key features (18 specified features are exempt from pruning)
- removes zero or near-zero variance features

This avoids training on noisy or useless columns.

### 5.2 Feature Ranking

The notebook uses mutual information to rank features and selects the top features combined with the required bank features to form the selected model input set.

### 5.3 Feature Enrichment

The notebook creates derived features such as:

- ratios (e.g. `FEAT_out_in_count_ratio`)
- log transforms (`LOG_<feature>`)
- aggregate behavior signals
- KMeans cluster labels (`FEAT_kmeans_cluster`)
- cluster distances (`FEAT_dist_cluster_0` … `FEAT_dist_cluster_4`)

This expands the signal available to the model without changing the raw source dataset.

### 5.4 Saved Feature Contract

The final feature list is saved to:

- [reports/features/selected_feature_list.csv](reports/features/selected_feature_list.csv)

That file is critical because the backend uses it to align incoming request values with the exact feature order used in training.

## 6. Supervised Model Training

The supervised training stage is in [notebooks/02_model_training.py](notebooks/02_model_training.py).

### 6.1 Models Trained

The notebook trains a **model zoo** of eight candidates:

- LightGBM (with Optuna hyperparameter optimisation)
- XGBoost
- CatBoost (optional, skipped if not installed)
- Random Forest
- Extra Trees
- Decision Tree
- Logistic Regression
- MLP (sklearn neural network)
- Stacking Ensemble (LR meta-learner on out-of-fold predictions from all base learners)

### 6.2 Why Multiple Models

The project compares several learners because fraud detection is usually class-imbalanced and benefits from model diversity. The **holdout PR-AUC winner** is copied to `models/best_model.pkl` for production serving.

### 6.3 Threshold Selection

Two thresholds are saved in metadata:

- **`optimal_threshold`** — maximizes F1 on train OOF predictions
- **`precision_optimal_threshold`** — maximizes precision subject to recall ≥ 0.80 on holdout (used as default for decisions/suspicion mapping)

Current metadata (holdout-honest, post leakage fix):
```json
{
  "best_model": "XGBoost",
  "best_model_type": "single",
  "optimal_threshold": 0.18,
  "precision_optimal_threshold": 0.10,
  "holdout_pr_auc": 0.888,
  "holdout_roc_auc": 0.999,
  "holdout_precision_at_optimal": 0.778,
  "holdout_recall_at_optimal": 0.875,
  "recall_at_fpr_1pct": 1.0
}
```

That file is loaded by the backend during startup.

### 6.4 Saved Artifacts

The training stage writes:

- `best_model.pkl` — production winner (currently XGBoost)
- `lgbm_final.pkl`, `xgb_final.pkl`, `rf_final.pkl`, `extratrees_final.pkl`
- `decision_tree_final.pkl`, `lr_final.pkl`, `mlp_final.pkl`
- `stacking_meta_learner.pkl`, `stacking_bundle.pkl`
- `scaler.pkl` (for logistic regression / MLP)
- `best_model_metadata.json`
- `reports/models/model_comparison.csv`

These are stored in [models/](models/) and [reports/models/](reports/models/).

## 7. Anomaly Detection And Risk Fusion

The anomaly stage is in [notebooks/03_anomaly_detection.py](notebooks/03_anomaly_detection.py).

### 7.1 Why This Exists

The supervised model learns patterns from labeled data.
The anomaly model looks for accounts that are unusual even if labels are incomplete or new behavior appears.

### 7.2 Main Components

- `RobustScaler` for scaling
- `IsolationForest` for outlier scoring
- fused score generation

### 7.3 Fusion Formula

The final risk score is a weighted blend:

- 70% supervised probability
- 30% anomaly score

That final score is what the application turns into a decision.

### 7.4 Saved Outputs

The anomaly notebook saves:

- `isolation_forest.pkl`
- `robust_scaler_anomaly.pkl`
- `risk_scores_all_accounts.csv`
- plots under `reports/models/`

## 8. SHAP Explainability

The explainability stage is in [notebooks/04_shap_explainability.py](notebooks/04_shap_explainability.py).

### 8.1 What It Does

Uses SHAP TreeExplainer on the production model (`best_model.pkl`, currently XGBoost) to:

1. Validate feature contributions make domain sense.
2. Produce per-account decision explanations for fraud investigators.
3. Identify the most globally influential features.
4. Generate human-readable narratives per case.

### 8.2 Generated Outputs

| Output                         | File                                        | Usage                                     |
| :----------------------------- | :------------------------------------------ | :---------------------------------------- |
| Global importance bar chart    | `reports/explainability/shap_global_importance_bar.png` | Model documentation, regulator reports |
| Beeswarm impact plot           | `reports/explainability/shap_beeswarm.png`  | Feature value vs impact direction         |
| Waterfall: True Positive       | `reports/explainability/shap_waterfall_true_positive.png` | Why an account was flagged            |
| Waterfall: False Positive      | `reports/explainability/shap_waterfall_false_positive.png` | Analyst appeal case                  |
| Per-account CSV                | `reports/explainability/per_account_shap_explanations.csv` | Investigator portal feed            |

### 8.3 Sample Investigator Narrative

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

## 9. Backend API

The backend lives in [serving/app.py](serving/app.py).

It is a FastAPI app that does three jobs:

1. loads trained artifacts at startup
2. exposes scoring, explanation, and decision management endpoints
3. serves the static frontend

### 9.1 Startup Loading

At startup the backend loads:

- the production model (`best_model.pkl`, fallback `lgbm_final.pkl`)
- the serialized feature pipeline (`feature_pipeline.pkl`)
- typology thresholds (`typology_thresholds.json`)
- the Isolation Forest (`isolation_forest.pkl`)
- the anomaly scaler (`robust_scaler_anomaly.pkl`)
- the best model metadata (`best_model_metadata.json`)
- the feature list (`reports/features/selected_feature_list.csv`)

If the artifacts are missing, the app fails fast with a `RuntimeError` instead of returning dummy scores.

### 9.2 Feature Alignment

The backend reads [reports/features/selected_feature_list.csv](reports/features/selected_feature_list.csv) and uses it as the authoritative feature order.

That matters because the frontend and backend must agree on column order and names.

### 9.3 Score Calculation

For each account:

1. Build a feature vector aligned to `FEATURE_COLS` order.
2. Predict supervised probability via `best_model.pkl` (auto-selected from the model zoo).
3. Scale features with `RobustScaler` and compute Isolation Forest anomaly score.
4. Fuse the two scores (70/30 weighted blend).
5. Run typology rules (`TypologyEngine`) on bank keys; apply typology boost (max +0.10) → `adjusted_fused_risk_score`.
6. Map adjusted score to `decision` and `suspicion_level` (1–4) using the active policy.
7. Attach `typology_flags` (`silent_account`, `large_amount_mover`, `instant_mule`, `aggregator_hub`).
8. Check for manual overrides.
9. Cache the result for subsequent `/explain/{id}` and `/alerts/suspicious-list` calls.

### 9.4 Decision Policy

The live policy controls how the fused score maps to action:

| Policy     | BLOCK ≥  | CHALLENGE ≥ | REVIEW ≥ | APPROVE    |
| :--------- | :------- | :---------- | :------- | :--------- |
| `strict`   | 0.75     | 0.55        | 0.35     | < 0.35     |
| `balanced` | 0.85     | 0.65        | 0.45     | < 0.45     |
| `loose`    | 0.95     | 0.80        | 0.60     | < 0.60     |

The frontend can request a policy change through `POST /decision/policy`.

Aliases are supported: `stricter` → `strict`, `default` → `balanced`, `looser` → `loose`.

### 9.5 Manual Overrides

Investigators can override a selected account through `POST /decision/override`.

That is an application-level decision override, not a model retrain.

The override is stored in memory and will disappear when the backend restarts.

When an override is active, scoring responses include `decision_overridden: true`, the `original_decision` (what the model would have chosen), and the `override_reason`.

Existing overrides can be queried with `GET /decision/override/{account_id}`.

### 9.6 Scoring Endpoints

| Method | Endpoint                          | Description                               |
| :----- | :-------------------------------- | :---------------------------------------- |
| `GET`  | `/health`                         | Liveness probe                            |
| `GET`  | `/model/info`                     | Model metadata, feature list, policy      |
| `POST` | `/score`                          | Score a single account (< 100 ms)         |
| `POST` | `/score/batch`                    | Score up to 200,000 accounts              |
| `GET`  | `/explain/{id}`                   | Feature-level explanation for a scored ID |
| `GET`  | `/alerts/suspicious-list`         | Suspicion watch list from cached scores   |
| `POST` | `/decision/policy`                | Switch decision preset                    |
| `POST` | `/decision/override`              | Apply an investigator override            |
| `GET`  | `/decision/override/{account_id}` | Retrieve override for an account          |

### 9.7 Batch Scoring Strategy

For batches ≤ 100 accounts, the API loops through `score_single()` for each row.

For batches > 100, the API uses a fully vectorized path:
- Builds a NumPy feature matrix directly.
- Runs a single `predict_proba` call on the entire matrix.
- Runs a single `decision_function` call for anomaly scoring.
- Computes fused scores as a vectorized operation.

This enables scoring up to 200,000 rows in one call (`MAX_BATCH_SIZE`).

### 9.8 Explain Endpoint

`GET /explain/{id}` returns TreeExplainer SHAP values for any account previously scored via `POST /score`. The response includes:

- `base_value` — SHAP expected value from TreeExplainer on the production model
- `fused_risk_score` — the cached fused score from scoring
- `explanations` — top 10 features by absolute SHAP contribution

Accounts must be scored first; otherwise the endpoint returns HTTP 404.

### 9.9 Static Frontend Mount

The frontend is served from the same FastAPI app:
```python
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

This is why the app can be opened in the browser after starting the backend.

### 9.10 In-Memory Caches

- `SCORED_ACCOUNTS_CACHE` — stores up to 1,000 recent scoring results for the `/explain/{id}` endpoint. Older entries are evicted FIFO.
- `MANUAL_OVERRIDES` — stores investigator overrides. Not persisted across restarts.

## 10. Frontend Application

The browser UI is in [frontend/index.html](frontend/index.html).

It is a single-page UI that:

- accepts CSV upload
- supports manual single-account entry
- calls backend APIs
- renders the results table
- displays explanations
- exposes policy and override controls

### 10.1 Batch Upload Flow

1. User drops or selects a CSV.
2. `parseCSV()` converts the text into row objects.
3. `buildFullFeaturePayload()` maps each row to the backend's feature schema.
4. `processBatchScoring()` sends the payload to `/score/batch`.
5. The response is rendered into the table and summary cards.

### 10.2 How The Frontend Learns The Feature List

Before scoring, the frontend calls `/model/info` and reads:

- `feature_columns`
- `decision_policy`
- model metadata

This keeps the UI aligned with whatever feature set the backend is currently using.

### 10.3 Single Account Flow

The single-account panel lets a user manually enter features and call `/score`.

### 10.4 Decision Policy UI

The UI lets investigators choose:

- stricter
- balanced
- looser

Then it calls `/decision/policy` and refreshes the displayed decisions.

### 10.5 Manual Override UI

The selected account panel includes a manual override control.

That lets an investigator force a row to:

- APPROVE
- REVIEW
- CHALLENGE
- BLOCK

### 10.6 Table And Explanation Panel

The results grid shows Account ID, supervised score, anomaly score, fused score, decision, **suspicion level (L1–L4 badge)**, and **typology flag chips**.

Filters are available for decision, suspicion level, and typology flag. **Download Suspicion List CSV** exports:
`account_id, supervised_score, anomaly_score, fused_risk_score, adjusted_fused_risk_score, suspicion_level, suspicion_label, decision, typology_flags`.

When a row is selected, the frontend:

- highlights the row
- shows the fused score, suspicion level, and typology flags
- shows the decision
- fetches SHAP explanation data from `/explain/{id}`

## 11. Missing Values: Exact Behavior

This is the part that usually causes confusion.

### 11.1 During EDA

Missingness is measured column by column.

### 11.2 During Feature Engineering

- Columns with too much missingness are dropped (bank key features are exempt).
- Remaining numeric columns are imputed with medians.
- The engineered dataset should not contain unresolved NaNs in the main training matrix.

### 11.3 During Browser Upload

If the CSV upload does not contain all model columns:

- the frontend fills missing columns with `0.0`
- the backend also fills missing fields with `0.0`

That is why scoring still works even with incomplete input.

### 11.4 Why Scores Can Collapse

If the upload file has:

- the wrong headers
- only a tiny subset of the engineered features
- mostly blank fields

then many rows can end up with very similar numeric vectors.

When that happens, the score can appear flat across all rows.

## 12. Output And Artifacts

The pipeline generates several important outputs:

| Directory                                 | Contents                                    |
| :---------------------------------------- | :------------------------------------------ |
| `models/`                                 | Trained `.pkl` files + `best_model_metadata.json` |
| `reports/eda/`                            | EDA charts and missing-value CSV            |
| `reports/features/`                       | MI scores and `selected_feature_list.csv`   |
| `reports/models/`                         | Model comparison, feature importance, risk scores |
| `reports/explainability/`                 | SHAP plots and per-account explanation CSV  |
| `data/engineered/`                        | Engineered parquet/CSV                      |

These artifacts are what connect training to serving.

## 13. Run Order

The intended local run order is:

1. Run EDA (`notebooks/00_eda_exploration.py`).
2. Run feature engineering (`notebooks/01_feature_engineering.py`).
3. Train supervised models (`notebooks/02_model_training.py`).
4. Train anomaly detection (`notebooks/03_anomaly_detection.py`).
5. Run SHAP explainability (`notebooks/04_shap_explainability.py`).
6. Start the backend API (`uvicorn serving.app:app --host 0.0.0.0 --port 8000`).
7. Open the frontend at `http://localhost:8000` and upload a CSV.

## 14. Short Version

If you only remember one thing:

- the notebooks prepare and train the model
- the backend loads the trained artifacts and runs the serialized feature pipeline
- clients send the 18 bank key features; engineered features are computed server-side
- the final decision is based on the fused score plus the current policy or manual override
- SHAP explanations use TreeExplainer online (`/explain/{id}`) and offline (`04_shap_explainability.py`)

## 15. Practical Caveat

If you upload a CSV and every row gets the same score, the first things to check are:

- is the backend actually running?
- does `/model/info` respond?
- does the file include the 18 required bank key features (`F115`–`F3894`)?
- is `models/feature_pipeline.pkl` present and loaded at startup?

That is usually the difference between a healthy scoring run and a flat, suspicious-looking output.
