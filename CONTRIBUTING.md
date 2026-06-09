# Contributing Guidelines

We welcome contributions! Follow these steps:

1. **Fork the repository** and clone your fork.
2. **Create a feature branch** (`git checkout -b my-feature`).
3. **Install dependencies**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
4. **Run the pipeline locally** as described in the README (`Section 14 – Execution Order`).
5. **Add tests** under the `tests/` directory.
6. **Commit** with a clear message and **push** to your fork.
7. Open a **Pull Request** targeting `main`.

### Code Style
- Use **ruff** for linting (`ruff check .`).
- Follow the existing file structure and naming conventions.
- Keep docstrings up‑to‑date.

### Testing
- Run `pytest` locally.
- Ensure CI passes before merging.

For any questions, open an issue or contact the repository maintainers.
