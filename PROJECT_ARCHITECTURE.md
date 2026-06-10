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
6. The backend loads those artifacts and serves predictions.
7. The frontend uploads CSVs or manual inputs and displays decisions.

## 2. Repository Map

Important files:

- [DataSet.csv](/d:/BOI-Project/DataSet.csv)
- [notebooks/00_eda_exploration.py](/d:/BOI-Project/notebooks/00_eda_exploration.py)
- [notebooks/01_feature_engineering.py](/d:/BOI-Project/notebooks/01_feature_engineering.py)
- [notebooks/02_model_training.py](/d:/BOI-Project/notebooks/02_model_training.py)
- [notebooks/03_anomaly_detection.py](/d:/BOI-Project/notebooks/03_anomaly_detection.py)
- [serving/app.py](/d:/BOI-Project/serving/app.py)
- [frontend/index.html](/d:/BOI-Project/frontend/index.html)
- [reports/features/selected_feature_list.csv](/d:/BOI-Project/reports/features/selected_feature_list.csv)
- [models/best_model_metadata.json](/d:/BOI-Project/models/best_model_metadata.json)

## 3. Raw Data And Missing Values

The raw dataset is stored in [DataSet.csv](/d:/BOI-Project/DataSet.csv). It is not used directly by the API. It first goes through the feature pipeline.

### 3.1 EDA Stage

In [notebooks/00_eda_exploration.py](/d:/BOI-Project/notebooks/00_eda_exploration.py), the project measures how much data is missing per column:

- `df.isnull().sum()`
- `df.isnull().mean()`

This produces missing-value reports and helps identify columns that are too sparse to keep.

### 3.2 Feature Engineering Stage

In [notebooks/01_feature_engineering.py](/d:/BOI-Project/notebooks/01_feature_engineering.py), missing values are handled in two ways:

1. Columns with more than 50% missing data are dropped.
2. The remaining numeric columns are imputed with median values using `SimpleImputer(strategy='median')`.

Relevant lines in the notebook:

- [notebooks/01_feature_engineering.py](/d:/BOI-Project/notebooks/01_feature_engineering.py#L81)
- [notebooks/01_feature_engineering.py](/d:/BOI-Project/notebooks/01_feature_engineering.py#L106)
- [notebooks/01_feature_engineering.py](/d:/BOI-Project/notebooks/01_feature_engineering.py#L112)

This means the model is trained on a cleaned numeric feature matrix rather than raw NaNs.

### 3.3 Why Scoring Still Works With Missing Inputs

At serving time, the application can still score incomplete input because missing request fields are replaced with `0.0`.

That happens in:

- [serving/app.py](/d:/BOI-Project/serving/app.py#L242)
- [serving/app.py](/d:/BOI-Project/serving/app.py#L248)
- [serving/app.py](/d:/BOI-Project/serving/app.py#L534)

So if a row is missing some fields, the backend still creates a complete feature vector and produces a score.

Important caveat:

- This does not mean missing values are ignored.
- It means they are substituted.
- If too many important fields are missing, rows may look similar and scores can flatten.

## 4. Feature Engineering

The feature pipeline is the bridge between raw data and the model.

### 4.1 Feature Pruning

In [notebooks/01_feature_engineering.py](/d:/BOI-Project/notebooks/01_feature_engineering.py), the pipeline:

- drops columns with too much missingness
- keeps the bank's required key features
- removes zero or near-zero variance features

This avoids training on noisy or useless columns.

### 4.2 Feature Ranking

The notebook uses mutual information to rank features:

- [notebooks/01_feature_engineering.py](/d:/BOI-Project/notebooks/01_feature_engineering.py#L135)

The top features are combined with the required bank features to form the selected model input set.

### 4.3 Feature Enrichment

The notebook creates derived features such as:

- ratios
- log transforms
- aggregate behavior signals
- KMeans cluster labels
- cluster distances

This expands the signal available to the model without changing the raw source dataset.

### 4.4 Saved Feature Contract

The final feature list is saved to:

- [reports/features/selected_feature_list.csv](/d:/BOI-Project/reports/features/selected_feature_list.csv)

That file is critical because the backend uses it to align incoming request values with the exact feature order used in training.

## 5. Supervised Model Training

The supervised training stage is in [notebooks/02_model_training.py](/d:/BOI-Project/notebooks/02_model_training.py).

### 5.1 Models Trained

The notebook trains multiple classifiers:

- LightGBM
- XGBoost
- Random Forest
- Logistic Regression
- Stacking Ensemble

### 5.2 Why Multiple Models

The project compares several learners because fraud detection is usually class-imbalanced and benefits from model diversity.

### 5.3 Threshold Selection

The notebook sweeps thresholds and selects the best one by F1 score.

That threshold and the best model metadata are saved into:

- [models/best_model_metadata.json](/d:/BOI-Project/models/best_model_metadata.json)

That file is loaded by the backend during startup.

### 5.4 Saved Artifacts

The training stage writes:

- `lgbm_final.pkl`
- `xgb_final.pkl`
- `rf_final.pkl`
- `lr_final.pkl`
- `stacking_meta_learner.pkl`
- `best_model_metadata.json`

These are stored in [models/](/d:/BOI-Project/models).

## 6. Anomaly Detection And Risk Fusion

The anomaly stage is in [notebooks/03_anomaly_detection.py](/d:/BOI-Project/notebooks/03_anomaly_detection.py).

### 6.1 Why This Exists

The supervised model learns patterns from labeled data.
The anomaly model looks for accounts that are unusual even if labels are incomplete or new behavior appears.

### 6.2 Main Components

- `RobustScaler` for scaling
- `IsolationForest` for outlier scoring
- fused score generation

### 6.3 Fusion Formula

The final risk score is a weighted blend:

- 70% supervised probability
- 30% anomaly score

That final score is what the application turns into a decision.

### 6.4 Saved Outputs

The anomaly notebook saves:

- `isolation_forest.pkl`
- `robust_scaler_anomaly.pkl`
- `risk_scores_all_accounts.csv`
- plots under `reports/models/`

## 7. Backend API

The backend lives in [serving/app.py](/d:/BOI-Project/serving/app.py).

It is a FastAPI app that does three jobs:

1. loads trained artifacts at startup
2. exposes scoring and metadata endpoints
3. serves the static frontend

### 7.1 Startup Loading

At startup the backend loads:

- the LightGBM model
- the Isolation Forest
- the anomaly scaler
- the best model metadata
- the feature list

If the artifacts are missing, the app now fails fast instead of returning dummy scores.

### 7.2 Feature Alignment

The backend reads [reports/features/selected_feature_list.csv](/d:/BOI-Project/reports/features/selected_feature_list.csv) and uses it as the authoritative feature order.

That matters because the frontend and backend must agree on column order and names.

### 7.3 Score Calculation

For each account:

1. Build a feature vector.
2. Predict supervised probability.
3. Scale features and compute anomaly score.
4. Fuse the two scores.
5. Map the fused score to a decision.
6. Cache the result for explanations and overrides.

### 7.4 Decision Policy

The live policy controls how the fused score maps to action:

- `strict`
- `balanced`
- `loose`

That policy is exposed in:

- [serving/app.py](/d:/BOI-Project/serving/app.py#L76)
- [serving/app.py](/d:/BOI-Project/serving/app.py#L369)

The frontend can request a policy change through `/decision/policy`.

### 7.5 Manual Overrides

Investigators can override a selected account through `/decision/override`.

That is an application-level decision override, not a model retrain.

The override is stored in memory and will disappear when the backend restarts.

### 7.6 Scoring Endpoints

Important routes:

- `GET /health`
- `GET /model/info`
- `POST /score`
- `POST /score/batch`
- `GET /explain/{id}`
- `POST /decision/policy`
- `POST /decision/override`

### 7.7 Static Frontend Mount

The frontend is served from the same FastAPI app using:

- [serving/app.py](/d:/BOI-Project/serving/app.py#L647)

This is why the app can be opened in the browser after starting the backend.

## 8. Frontend Application

The browser UI is in [frontend/index.html](/d:/BOI-Project/frontend/index.html).

It is a single-page UI that:

- accepts CSV upload
- supports manual single-account entry
- calls backend APIs
- renders the results table
- displays explanations
- exposes policy and override controls

### 8.1 Batch Upload Flow

The batch upload flow:

1. User drops or selects a CSV.
2. `parseCSV()` converts the text into row objects.
3. `buildFullFeaturePayload()` maps each row to the backend's feature schema.
4. `processBatchScoring()` sends the payload to `/score/batch`.
5. The response is rendered into the table and summary cards.

Relevant code:

- [frontend/index.html](/d:/BOI-Project/frontend/index.html#L1203)
- [frontend/index.html](/d:/BOI-Project/frontend/index.html#L1224)

### 8.2 How The Frontend Learns The Feature List

Before scoring, the frontend calls `/model/info` and reads:

- `feature_columns`
- `decision_policy`
- model metadata

That happens in:

- [frontend/index.html](/d:/BOI-Project/frontend/index.html#L916)

This keeps the UI aligned with whatever feature set the backend is currently using.

### 8.3 Single Account Flow

The single-account panel lets a user manually enter features and call `/score`.

Relevant code:

- [frontend/index.html](/d:/BOI-Project/frontend/index.html#L1535)

### 8.4 Decision Policy UI

The UI lets investigators choose:

- stricter
- balanced
- looser

Then it calls `/decision/policy` and refreshes the displayed decisions.

Relevant code:

- [frontend/index.html](/d:/BOI-Project/frontend/index.html#L652)
- [frontend/index.html](/d:/BOI-Project/frontend/index.html#L1044)

### 8.5 Manual Override UI

The selected account panel includes a manual override control.

That lets an investigator force a row to:

- APPROVE
- REVIEW
- CHALLENGE
- BLOCK

Relevant code:

- [frontend/index.html](/d:/BOI-Project/frontend/index.html#L775)
- [frontend/index.html](/d:/BOI-Project/frontend/index.html#L1068)

### 8.6 Table And Explanation Panel

The results grid is rendered in the browser.

When a row is selected, the frontend:

- highlights the row
- shows the fused score
- shows the decision
- fetches the explanation data from `/explain/{id}`

Relevant code:

- [frontend/index.html](/d:/BOI-Project/frontend/index.html#L1371)
- [frontend/index.html](/d:/BOI-Project/frontend/index.html#L1429)

## 9. Missing Values: Exact Behavior

This is the part that usually causes confusion.

### 9.1 During EDA

Missingness is measured column by column.

### 9.2 During Feature Engineering

- Columns with too much missingness are dropped.
- Remaining numeric columns are imputed with medians.
- The engineered dataset should not contain unresolved NaNs in the main training matrix.

### 9.3 During Browser Upload

If the CSV upload does not contain all model columns:

- the frontend fills missing columns with `0.0`
- the backend also fills missing fields with `0.0`

That is why scoring still works even with incomplete input.

### 9.4 Why Scores Can Collapse

If the upload file has:

- the wrong headers
- only a tiny subset of the engineered features
- mostly blank fields

then many rows can end up with very similar numeric vectors.

When that happens, the score can appear flat across all rows.

## 10. Output And Artifacts

The pipeline generates several important outputs:

- [models/](/d:/BOI-Project/models)
- [reports/models/](/d:/BOI-Project/reports/models)
- [reports/features/selected_feature_list.csv](/d:/BOI-Project/reports/features/selected_feature_list.csv)

These artifacts are what connect training to serving.

## 11. Run Order

The intended local run order is:

1. Run EDA.
2. Run feature engineering.
3. Train supervised models.
4. Train anomaly detection.
5. Start the backend API.
6. Open the frontend and upload a CSV.

The quick-start instructions are also summarized in [README.md](/d:/BOI-Project/README.md).

## 12. Short Version

If you only remember one thing:

- the notebooks prepare and train the model
- the backend loads the trained artifacts and exposes scoring APIs
- the frontend is just a UI client that sends data to the backend
- missing values are imputed or zero-filled so scoring can still happen
- the final decision is based on the fused score plus the current policy or manual override

## 13. Practical Caveat

If you upload a CSV and every row gets the same score, the first things to check are:

- is the backend actually running?
- does `/model/info` respond?
- does the file match the engineered feature columns?
- are most of the important columns missing and being zero-filled?

That is usually the difference between a healthy scoring run and a flat, suspicious-looking output.

