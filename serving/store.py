"""SQLite-backed persistence for audit, policy, and scoring history."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Iterable

from serving.config import settings


class AuditStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS score_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    request_id TEXT,
                    actor TEXT,
                    role TEXT,
                    model_version TEXT,
                    risk_score REAL,
                    anomaly_score REAL,
                    fused_risk_score REAL,
                    adjusted_fused_risk_score REAL,
                    suspicion_level INTEGER,
                    suspicion_label TEXT,
                    typology_flags TEXT,
                    decision TEXT,
                    risk_level TEXT,
                    original_decision TEXT,
                    decision_overridden INTEGER,
                    override_reason TEXT,
                    scored_at TEXT,
                    latency_ms REAL,
                    features_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS manual_overrides (
                    account_id TEXT PRIMARY KEY,
                    decision TEXT NOT NULL,
                    reason TEXT,
                    updated_at TEXT NOT NULL,
                    actor TEXT,
                    role TEXT,
                    request_id TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    thresholds_json TEXT NOT NULL,
                    actor TEXT,
                    role TEXT,
                    request_id TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def record_score(self, payload: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO score_events (
                    account_id, request_id, actor, role, model_version,
                    risk_score, anomaly_score, fused_risk_score,
                    adjusted_fused_risk_score, suspicion_level, suspicion_label,
                    typology_flags, decision, risk_level, original_decision,
                    decision_overridden, override_reason, scored_at, latency_ms,
                    features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    payload["account_id"],
                    payload.get("request_id"),
                    payload.get("actor"),
                    payload.get("role"),
                    payload.get("model_version"),
                    payload.get("risk_score"),
                    payload.get("anomaly_score"),
                    payload.get("fused_risk_score"),
                    payload.get("adjusted_fused_risk_score"),
                    payload.get("suspicion_level"),
                    payload.get("suspicion_label"),
                    json.dumps(payload.get("typology_flags", [])),
                    payload.get("decision"),
                    payload.get("risk_level"),
                    payload.get("original_decision"),
                    int(bool(payload.get("decision_overridden"))),
                    payload.get("override_reason"),
                    payload.get("scored_at"),
                    payload.get("latency_ms"),
                    json.dumps(payload.get("features", {})),
                ),
            )

    def record_scores(self, payloads: Iterable[dict]) -> None:
        rows = list(payloads)
        if not rows:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO score_events (
                    account_id, request_id, actor, role, model_version,
                    risk_score, anomaly_score, fused_risk_score,
                    adjusted_fused_risk_score, suspicion_level, suspicion_label,
                    typology_flags, decision, risk_level, original_decision,
                    decision_overridden, override_reason, scored_at, latency_ms,
                    features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    (
                        payload["account_id"],
                        payload.get("request_id"),
                        payload.get("actor"),
                        payload.get("role"),
                        payload.get("model_version"),
                        payload.get("risk_score"),
                        payload.get("anomaly_score"),
                        payload.get("fused_risk_score"),
                        payload.get("adjusted_fused_risk_score"),
                        payload.get("suspicion_level"),
                        payload.get("suspicion_label"),
                        json.dumps(payload.get("typology_flags", [])),
                        payload.get("decision"),
                        payload.get("risk_level"),
                        payload.get("original_decision"),
                        int(bool(payload.get("decision_overridden"))),
                        payload.get("override_reason"),
                        payload.get("scored_at"),
                        payload.get("latency_ms"),
                        json.dumps(payload.get("features", {})),
                    )
                    for payload in rows
                ],
            )

    def upsert_override(self, payload: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO manual_overrides (
                    account_id, decision, reason, updated_at, actor, role, request_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    decision=excluded.decision,
                    reason=excluded.reason,
                    updated_at=excluded.updated_at,
                    actor=excluded.actor,
                    role=excluded.role,
                    request_id=excluded.request_id;
                """,
                (
                    payload["account_id"],
                    payload["decision"],
                    payload.get("reason"),
                    payload["updated_at"],
                    payload.get("actor"),
                    payload.get("role"),
                    payload.get("request_id"),
                ),
            )

    def record_policy(self, payload: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO policy_events (
                    mode, thresholds_json, actor, role, request_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    payload["mode"],
                    json.dumps(payload.get("thresholds", {})),
                    payload.get("actor"),
                    payload.get("role"),
                    payload.get("request_id"),
                    payload["created_at"],
                ),
            )

    def load_overrides(self) -> dict[str, dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT account_id, decision, reason, updated_at, actor, role, request_id FROM manual_overrides"
            ).fetchall()
        return {
            row["account_id"]: {
                "decision": row["decision"],
                "reason": row["reason"],
                "updated_at": row["updated_at"],
                "actor": row["actor"],
                "role": row["role"],
                "request_id": row["request_id"],
            }
            for row in rows
        }

    def load_latest_policy(self) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT mode, thresholds_json, created_at
                FROM policy_events
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        return {
            "mode": row["mode"],
            "thresholds": json.loads(row["thresholds_json"]),
            "created_at": row["created_at"],
        }

    def load_recent_scores(self, limit: int) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM score_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        unique_rows: list[dict] = []
        seen: set[str] = set()
        for row in rows:
            if row["account_id"] in seen:
                continue
            seen.add(row["account_id"])
            unique_rows.append(dict(row))
        return unique_rows

    def load_score(self, account_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM score_events
                WHERE account_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (account_id,),
            ).fetchone()
        return dict(row) if row else None

    def load_metrics(self) -> dict:
        with self._lock, self._connect() as conn:
            score_count = conn.execute("SELECT COUNT(*) AS total FROM score_events").fetchone()["total"]
            override_count = conn.execute("SELECT COUNT(*) AS total FROM manual_overrides").fetchone()["total"]
            policy_count = conn.execute("SELECT COUNT(*) AS total FROM policy_events").fetchone()["total"]
            last_score = conn.execute(
                "SELECT scored_at FROM score_events ORDER BY id DESC LIMIT 1"
            ).fetchone()

        return {
            "score_events": int(score_count),
            "manual_overrides": int(override_count),
            "policy_events": int(policy_count),
            "last_score_at": last_score["scored_at"] if last_score else None,
            "db_path": str(self.db_path),
        }


audit_store = AuditStore(settings.audit_db_path)
