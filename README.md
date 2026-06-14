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
| `POST` | `/score/batch` | Score up to 200,000 accounts |
| `GET` | `/explain/{id}` | SHAP explanation for a scored account |
| `GET` | `/alerts/suspicious-list` | Filter by suspicion level / typology |
| `POST` | `/decision/policy` | Switch threshold preset |
| `POST` | `/decision/override` | Investigator manual override |

**Input:** 18 bank key features (`F115`–`F3894`). The server computes all engineered features via `models/feature_pipeline.pkl`.

**Output:** `risk_score`, `anomaly_score`, `fused_risk_score`, `adjusted_fused_risk_score`, `decision`, `suspicion_level`, `suspicion_label`, `typology_flags`.

Interactive docs: [`http://localhost:8000/docs`](http://localhost:8000/docs)

---

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

## License

MIT License — see [`LICENSE`](LICENSE) for details.
