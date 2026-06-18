# Mule Account Detection Platform

**ML-powered fraud detection for financial institutions — real-time scoring, typology intelligence, and investigator-ready explainability.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen.svg)](tests/)

---

## Problem Statement

Mule accounts are legitimate-looking bank accounts hijacked by fraud networks to receive, move, and launder illicit funds. They are difficult to catch because:

- **Behaviour is diverse** — silent dormant accounts, large-value movers, instant pass-through patterns, and aggregator hubs look different from each other.
- **Labels are scarce** — fraud is rare (~1% of accounts), so models must be evaluated on precision–recall trade-offs, not accuracy.
- **Rules alone fail** — static thresholds miss novel typologies and cannot explain *why* an account was flagged.
- **Investigators need workflow** — not just a score, but decisions, suspicion tiers, typology tags, SHAP narratives, and exportable watch lists.

---

## Solution

This platform delivers a **defense-in-depth scoring pipeline** that combines supervised ML, unsupervised anomaly detection, and rule-based typology intelligence into a single investigator-facing system.

| Layer | What it does |
| :---- | :----------- |
| **Feature pipeline** | 18 bank key inputs → 266 engineered features (train/serve parity via serialized pipeline) |
| **Model zoo** | LightGBM, XGBoost, RF, ExtraTrees, DecisionTree, LR, MLP, stacking — auto-selects holdout winner |
| **Anomaly detection** | Isolation Forest trained on legitimate accounts only |
| **Score fusion** | `70% supervised + 30% anomaly` → typology boost (max +0.10) → adjusted risk score |
| **Typology engine** | Tags `silent_account`, `large_amount_mover`, `instant_mule`, `aggregator_hub` |
| **Decision + suspicion** | Four-tier action (BLOCK / CHALLENGE / REVIEW / APPROVE) plus suspicion levels 1–4 |
| **Explainability** | Real SHAP TreeExplainer at `/explain/{id}` |
| **Investigator UI** | Batch CSV upload, filters, policy controls, overrides, Suspicion List CSV export |

### Holdout performance (honest evaluation)

| Metric | Value |
| :----- | :---- |
| Production model | **XGBoost** (auto-selected) |
| Holdout PR-AUC | **0.888** |
| Holdout ROC-AUC | **0.999** |
| Precision @ optimal threshold | **0.778** |
| Recall @ optimal threshold | **0.875** |
| Recall @ FPR 1% | **1.000** |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Investigator Browser (SPA)                          │
│   CSV upload · results grid · SHAP panel · policy · overrides · CSV export  │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ HTTP / JSON
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                          FastAPI Serving Layer (app.py)                     │
│  /score  /score/batch  /explain/{id}  /alerts/suspicious-list  /decision/*   │
└───────┬─────────────────┬──────────────────────┬────────────────────────────┘
        │                 │                      │
        ▼                 ▼                      ▼
┌───────────────┐ ┌───────────────┐    ┌────────────────────┐
│ FeaturePipeline│ │ best_model.pkl│    │  TypologyEngine    │
│ (266 features) │ │ (model zoo    │    │  rule tags + boost │
│               │ │  winner)      │    │                    │
└───────┬───────┘ └───────┬───────┘    └─────────┬──────────┘
        │                 │                        │
        └────────┬────────┘                        │
                 ▼                                  │
        ┌─────────────────┐                         │
        │ Isolation Forest │◄── RobustScaler        │
        └────────┬─────────┘                         │
                 │                                   │
                 ▼                                   ▼
        ┌────────────────────────────────────────────────────┐
        │  Fused score → typology boost → suspicion + decision │
        └────────────────────────────────────────────────────┘
```

**Training pipeline** (offline): EDA → leakage audit → feature engineering → model zoo → anomaly layer → SHAP reports.

See [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) for full design details and [`docs/diagrams/`](docs/diagrams/) for C4 and flow diagrams.

---

## Quick Start

```bash
git clone https://github.com/Blackbebertex/BOI-Project.git
cd BOI-Project

git lfs install && git lfs pull          # download DataSet.csv (~111 MB)

python -m venv .venv
.venv\Scripts\activate                   # Windows
# source .venv/bin/activate              # Linux/macOS

pip install -r requirements.txt

# Train pipeline (optional — pre-trained artifacts included)
python notebooks/01_feature_engineering.py
python notebooks/02_model_training.py
python notebooks/03_anomaly_detection.py

# Configure API keys for protected endpoints
export MULE_API_KEYS=demo-investigator-key
export MULE_ADMIN_API_KEYS=demo-admin-key

# Run tests
pytest

# Start API + UI
uvicorn serving.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Docker

```bash
docker build -t mule-detector .
docker run -p 8000:8000 mule-detector
```

---

## API Overview

| Method | Endpoint | Description |
| :----- | :------- | :---------- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/model/info` | Model metadata, features, decision policy |
| `POST` | `/score` | Score a single account |
| `POST` | `/score/batch` | Score up to the configured batch limit |
| `GET` | `/explain/{id}` | SHAP explanation for a scored account |
| `GET` | `/alerts/suspicious-list` | Filter by suspicion level / typology |
| `POST` | `/decision/policy` | Switch threshold preset |
| `POST` | `/decision/override` | Investigator manual override |
| `GET` | `/metrics` | Runtime audit and health counters |

**Input:** 18 bank key features (`F115`–`F3894`). The server computes all engineered features via `models/feature_pipeline.pkl`.

**Output:** `risk_score`, `anomaly_score`, `fused_risk_score`, `adjusted_fused_risk_score`, `decision`, `suspicion_level`, `suspicion_label`, `typology_flags`.

Interactive docs: [`http://localhost:8000/docs`](http://localhost:8000/docs)

---

### Runtime Security

Protected endpoints require an `X-API-Key` or `Authorization: Bearer ...` header.

Default local demo keys:

- Investigator key: `demo-investigator-key`
- Admin key: `demo-admin-key`

Configure these environment variables before production deployment:

- `MULE_API_KEYS`
- `MULE_ADMIN_API_KEYS`
- `MULE_AUDIT_DB_PATH`
- `MULE_MAX_BATCH_SIZE`
- `MULE_CORS_ORIGINS`

## Project Structure

```
BOI-Project/
├── serving/                 # FastAPI app, feature pipeline, typology engine
├── frontend/                # Investigator single-page UI
├── notebooks/               # EDA → FE → training → anomaly → SHAP
├── models/                  # best_model.pkl, feature_pipeline.pkl, zoo artifacts
├── tests/                   # 33 pytest tests (leakage, pipeline, serving, typology)
├── docs/                    # ML report, architecture diagrams, technical proposal
├── reports/                 # Metrics, SHAP plots, model comparison
└── scripts/                 # Diagram rendering & document generation
```

---

## Documentation

| Document | Description |
| :------- | :---------- |
| [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) | End-to-end system design and API reference |
| [`docs/MULE_ML_ENGINEER_REPORT.md`](docs/MULE_ML_ENGINEER_REPORT.md) | Feature engineering, model zoo, typology, metrics |
| [`mule_account_detection_blueprint.md`](mule_account_detection_blueprint.md) | Full ML engineering blueprint |
| [`ENTERPRISE_TECHNICAL_DOCUMENTATION.md`](ENTERPRISE_TECHNICAL_DOCUMENTATION.md) | Enterprise technical reference (C4, security, QA) |

Generate architecture diagrams and a technical Word proposal:

```bash
pip install -r requirements-docs.txt
python scripts/render_diagrams.py
python scripts/generate_hackathon_word_doc.py
```

---

## Future Plans & Advancement

Ideas under consideration for evolving the platform — no fixed schedule, ordered by theme.

### Machine learning & data

- Expand the model zoo with CatBoost, TabNet, and deeper ensemble weighting strategies
- Boruta and SHAP-stability feature selection on top of mutual-information pre-filtering
- Graph-based features (PageRank, motif detection, shared-device clusters) when transaction graph data is available
- TabTransformer or target-encoded categorical handling for non-numeric bank attributes
- Adversarial validation and periodic retraining on rolling windows
- Adaptive fusion weights (learned or Bayesian-optimised) instead of fixed 70/30 blend
- Contextual bandit or branch-level adaptive thresholds tied to recent false-positive rates
- Platt scaling and probability calibration on holdout scores
- ECOD / PyOD ensemble layer alongside Isolation Forest

### Model governance & reliability

- Population Stability Index (PSI) monitoring on top features with automated retrain triggers
- MLflow model registry with versioned artifacts and promotion workflow
- Airflow (or equivalent) orchestration for scheduled retraining and report generation
- A/B scoring between model versions before production cutover
- Champion/challenger evaluation harness with automated holdout gates
- ONNX export for lighter inference runtimes

### Platform & deployment

- Kubernetes deployment with horizontal scaling and health-checked pods
- Split `requirements-serve.txt` vs `requirements-train.txt` for lean production images
- Prometheus metrics, structured logging, and distributed tracing on scoring latency
- Redis or PostgreSQL persistence for scored accounts, overrides, and audit trails
- Rate limiting and request-size guards on public API endpoints
- Multi-environment config (dev / staging / production) with secrets management

### Security & compliance

- JWT or API-key authentication on all scoring and decision endpoints
- Restricted CORS and TLS termination at the ingress layer
- Immutable audit log for overrides, policy changes, and model version switches
- Role-based access for investigators vs administrators
- Data retention policies and PII masking in exported CSVs

### Investigator experience

- Real-time alert stream and webhook integration for case-management systems
- Network visualisation for linked accounts and typology clusters
- Bulk override and case-assignment workflow in the UI
- Investigator feedback loop (confirm / dismiss) feeding label enrichment for retraining
- Dashboard for alert volume, tier distribution, and score drift by region or branch
- Multi-select typology filters and saved filter presets in the results grid

### Integrations & streaming

- Kafka / Flink ingestion for near-real-time scoring on live transaction events
- Batch scoring API callbacks for asynchronous large-file processing
- Feast or Redis feature store for low-latency pre-computed aggregates
- Neo4j or TigerGraph backend for relationship queries and graph typologies

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.
