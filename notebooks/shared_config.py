from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_CANDIDATES = (
    BASE_DIR / "DataSet.csv",
    DATA_DIR / "transactions.csv",
)

ENGINEERED_DATA_PATH = DATA_DIR / "engineered" / "transactions_engineered.parquet"
ENGINEERED_DATA_CSV_PATH = DATA_DIR / "engineered" / "transactions_engineered.csv"
ENGINEERED_HOLDOUT_PATH = DATA_DIR / "engineered" / "transactions_engineered_holdout.parquet"
ENGINEERED_HOLDOUT_CSV_PATH = DATA_DIR / "engineered" / "transactions_engineered_holdout.csv"
FEATURE_PIPELINE_PATH = MODELS_DIR / "feature_pipeline.pkl"


def resolve_raw_data_path() -> Path:
    for candidate in RAW_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    return RAW_DATA_CANDIDATES[0]


RAW_DATA_PATH = resolve_raw_data_path()
EDA_REPORTS_DIR = REPORTS_DIR / "eda"
FEATURE_REPORTS_DIR = REPORTS_DIR / "features"
MODEL_REPORTS_DIR = REPORTS_DIR / "models"
EXPLAINABILITY_REPORTS_DIR = REPORTS_DIR / "explainability"


def load_engineered_data():
    import pandas as pd

    for candidate in (ENGINEERED_DATA_PATH, ENGINEERED_DATA_CSV_PATH):
        if not candidate.exists():
            continue
        if candidate.suffix == ".parquet":
            try:
                return pd.read_parquet(candidate)
            except Exception:
                continue
        if candidate.suffix == ".csv":
            return pd.read_csv(candidate, low_memory=False)

    raise FileNotFoundError(
        "No engineered dataset found. Expected parquet or csv under data/engineered."
    )


def load_holdout_data():
    import pandas as pd

    for candidate in (ENGINEERED_HOLDOUT_PATH, ENGINEERED_HOLDOUT_CSV_PATH):
        if not candidate.exists():
            continue
        if candidate.suffix == ".parquet":
            try:
                return pd.read_parquet(candidate)
            except Exception:
                continue
        if candidate.suffix == ".csv":
            return pd.read_csv(candidate, low_memory=False)

    raise FileNotFoundError(
        "No holdout dataset found. Run notebooks/01_feature_engineering.py first."
    )
