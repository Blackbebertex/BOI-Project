"""Runtime configuration for the mule detection serving app."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = BASE_DIR / "data" / "runtime"


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    app_env: str
    auth_required: bool
    allow_demo_auth: bool
    api_keys: tuple[str, ...]
    admin_api_keys: tuple[str, ...]
    cors_origins: tuple[str, ...]
    read_rate_limit_per_minute: int
    batch_rate_limit_per_minute: int
    admin_rate_limit_per_minute: int
    max_body_bytes: int
    max_batch_size: int
    audit_db_path: Path
    request_window_seconds: int = 60


def load_config() -> AppConfig:
    app_env = os.getenv("APP_ENV", "development").strip().lower()
    allow_demo_auth = _bool_env("MULE_ALLOW_DEMO_AUTH", True)
    auth_required = _bool_env("MULE_AUTH_REQUIRED", True)
    api_keys = _split_csv(os.getenv("MULE_API_KEYS"))
    admin_api_keys = _split_csv(os.getenv("MULE_ADMIN_API_KEYS"))

    if not api_keys and not admin_api_keys and allow_demo_auth:
        api_keys = ("demo-investigator-key",)
        admin_api_keys = ("demo-admin-key",)

    if app_env in {"production", "staging"} and not os.getenv("MULE_API_KEYS") and not os.getenv("MULE_ADMIN_API_KEYS"):
        raise RuntimeError(
            "Production mode requires MULE_API_KEYS and MULE_ADMIN_API_KEYS to be configured."
        )

    cors_origins = _split_csv(
        os.getenv(
            "MULE_CORS_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000",
        )
    )

    read_rate_limit = int(os.getenv("MULE_READ_RATE_LIMIT_PER_MINUTE", "120"))
    batch_rate_limit = int(os.getenv("MULE_BATCH_RATE_LIMIT_PER_MINUTE", "12"))
    admin_rate_limit = int(os.getenv("MULE_ADMIN_RATE_LIMIT_PER_MINUTE", "60"))
    max_body_bytes = int(os.getenv("MULE_MAX_BODY_BYTES", "5000000"))
    max_batch_size = int(os.getenv("MULE_MAX_BATCH_SIZE", "20000"))

    runtime_db = Path(os.getenv("MULE_AUDIT_DB_PATH", str(RUNTIME_DIR / "mule_audit.sqlite3")))

    return AppConfig(
        app_env=app_env,
        auth_required=auth_required,
        allow_demo_auth=allow_demo_auth,
        api_keys=api_keys,
        admin_api_keys=admin_api_keys,
        cors_origins=cors_origins,
        read_rate_limit_per_minute=read_rate_limit,
        batch_rate_limit_per_minute=batch_rate_limit,
        admin_rate_limit_per_minute=admin_rate_limit,
        max_body_bytes=max_body_bytes,
        max_batch_size=max_batch_size,
        audit_db_path=runtime_db,
    )


settings = load_config()
