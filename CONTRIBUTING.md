# Contributing Guidelines

We welcome contributions! Follow these steps:

1. **Fork the repository** and clone your fork.
2. **Create a feature branch** (`git checkout -b my-feature`).
3. **Install dependencies**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # Linux/macOS
   pip install -r requirements.txt
   pip install ruff pytest
   ```
4. **Run the pipeline locally** as described in the README (Quick Start section).
5. **Add or update tests** — see the Testing section below.
6. **Commit** with a clear message and **push** to your fork.
7. Open a **Pull Request** targeting `main`.

## Code Style
- Use **ruff** for linting (`ruff check .`).
- Follow the existing file structure and naming conventions.
- Keep docstrings up‑to‑date.
- Use the shared path config in `notebooks/shared_config.py` instead of hard‑coding paths.

## Testing

### Smoke Tests (require a running API server)
The repo includes two quick integration scripts:

| Script           | Purpose                                         |
| :--------------- | :---------------------------------------------- |
| `test_score.py`  | Hits `/score` with a single test account         |
| `test_batch.py`  | Sends 150 rows to `/score/batch` (vectorized path) |

Run them after starting the server:
```bash
# Terminal 1 — start the API
uvicorn serving.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — run tests
python test_score.py
python test_batch.py
```

### Unit / Integration Tests (pytest)

Tests live in `tests/` and cover leakage audit, feature pipeline parity, typology engine, model artifacts, and FastAPI serving:

```bash
pytest -v
```

| Test file | Coverage |
| :-------- | :------- |
| `test_leakage_audit.py` | Label leakage detection (F3912 exclusion) |
| `test_feature_pipeline.py` | Train/serve feature pipeline shape and save/load |
| `test_typology_engine.py` | Typology rules (silent_account, large_amount_mover, etc.) |
| `test_model_artifacts.py` | Model zoo `.pkl` artifacts after training |
| `test_serving.py` | `/health`, `/score`, `/explain`, `/alerts/suspicious-list` |

Integration tests require trained artifacts in `models/` (run the notebook pipeline first).

### Documentation Generation

Generate the hackathon judge Word proposal:

```bash
pip install -r requirements-docs.txt
python scripts/render_diagrams.py
python scripts/generate_hackathon_word_doc.py
```

Output: `docs/BOI_Mule_Detection_Hackathon_Proposal.docx`

### CI Pipeline
The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:
1. Checks out the repo with LFS support.
2. Creates a virtual environment and installs dependencies.
3. Runs `ruff check .` for linting.
4. Runs `pytest` for automated tests.
5. Verifies Git‑LFS objects are present.

Ensure CI passes before merging.

## Docker Testing
To verify the containerised build:
```bash
docker build -t boi-mule-detector .
docker run -p 8000:8000 boi-mule-detector
# Then run test_score.py / test_batch.py against the container
```

For any questions, open an issue or contact the repository maintainers.
