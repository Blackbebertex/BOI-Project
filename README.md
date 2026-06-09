# BOI Mule Account Detection

**Bank of India (BOI) – PSB Cybersecurity, Fraud & AI Hackathon**

This repository contains an end‑to‑end solution for detecting mule accounts and suspicious transactions. It follows the comprehensive blueprint in `mule_account_detection_blueprint.md`.

## Quick Start
```bash
# Clone the repo
git clone https://github.com/Blackbebertex/BOI-Project.git
cd BOI-Project

git lfs install               # ensure LFS is enabled
git lfs pull                  # download the large DataSet.csv

# Create a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline step‑by‑step (see Section 14 of the blueprint)
python notebooks/00_eda_exploration.py
python notebooks/01_feature_engineering.py
python notebooks/02_model_training.py
python notebooks/03_anomaly_detection.py
python notebooks/04_shap_explainability.py

# Start the API service
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

## Key Features
- **Risk scoring & decision matrix** (four‑tier actions).
- **Model governance & drift monitoring** (PSI, periodic retraining).
- **Git‑LFS** for the 111 MB `DataSet.csv`.
- **Docker** container for reproducible serving.
- **CI pipeline** with linting, tests, and LFS verification.

## License
This project is licensed under the MIT License – see `LICENSE` for details.

---
*For full technical details, refer to the `mule_account_detection_blueprint.md` document.*
