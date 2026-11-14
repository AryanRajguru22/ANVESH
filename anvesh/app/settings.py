"""Key/value settings persisted as JSON (alert rules, offence catalogue, retention, thresholds).
Reads fall back to the supplied default, so a fresh database needs no seeding."""

from __future__ import annotations

from anvesh.app.db import AppDatabase, dumps, loads
from anvesh.app.timeutil import now_iso


class SettingsService:
    def __init__(self, db: AppDatabase) -> None:
        self._db = db

    def get(self, key: str, default=None):
        row = self._db.one("SELECT value_json FROM settings WHERE key = ?", (key,))
        return loads(row["value_json"], default) if row else default

    def set(self, key: str, value, actor: str = "system") -> None:
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO settings(key, value_json, updated_at, updated_by) VALUES (?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at, updated_by = excluded.updated_by",
                (key, dumps(value), now_iso(), actor))

    def all(self) -> dict:
        return {r["key"]: loads(r["value_json"]) for r in self._db.query("SELECT key, value_json FROM settings ORDER BY key")}
