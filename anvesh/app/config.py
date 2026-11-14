"""Application configuration. Everything is overridable from the environment;
nothing secret has a default and nothing secret is ever read from a file in the
repository."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclasses.dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    host: str = "127.0.0.1"
    port: int = 8750
    model_path: str = str(REPO_ROOT / "models" / "yolov8n.pt")
    # "pretrained" (COCO YOLOv8n, the default) or "finetuned" (an explicitly configured ANVESH checkpoint).
    model_source: str = "pretrained"
    session_ttl_s: int = 8 * 3600
    session_idle_timeout_s: int = 30 * 60  # a session unused for this long ends (0 disables)
    secure_cookies: bool = False  # set True behind HTTPS
    max_upload_bytes: int = 2 * 1024 ** 3
    # Four-eyes: the officer who approves a case may not be the user who verified its evidence/plate.
    require_distinct_approver: bool = True
    evidence_retention_days: int = 365
    audit_retention_days: int = 2555
    # Alert delivery (all optional; unset = that channel is "not connected").
    webhook_url: str = None
    webhook_secret: str = None
    smtp_host: str = None
    smtp_port: int = 587
    smtp_user: str = None
    smtp_password: str = None
    smtp_from: str = None
    # Registry / e-challan production adapters: unset = "not connected". ANVESH ships NO endpoints for them.
    registry_base_url: str = None
    registry_api_key: str = None
    # An operational approval needs a registry result retrieved within this many seconds (a stale answer is not an answer).
    registry_result_max_age_s: int = 24 * 3600
    challan_base_url: str = None
    challan_api_key: str = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "anvesh_app.sqlite3"

    @property
    def evidence_dir(self) -> Path:
        return self.data_dir / "evidence"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outbox_dir(self) -> Path:
        return self.data_dir / "outbox"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"


def _resolve_model_path(env) -> str:
    """`ANVESH_FINETUNED_MODEL` (explicit fine-tuned checkpoint) wins over `ANVESH_MODEL`
    (pretrained path override) which wins over the bundled pretrained default.

    This only SELECTS the path; existence is checked where the detector is built, which raises
    `WeightsNotFoundError` -- an explicitly configured fine-tuned path that is missing is never
    silently replaced by the pretrained model."""
    return (env.get("ANVESH_FINETUNED_MODEL") or env.get("ANVESH_MODEL")
            or str(REPO_ROOT / "models" / "yolov8n.pt"))


def load_config(env=None, data_dir=None) -> AppConfig:
    env = os.environ if env is None else env
    base = Path(data_dir or env.get("ANVESH_DATA_DIR") or (REPO_ROOT / "outputs" / "app"))
    cfg = AppConfig(
        data_dir=base.resolve(),
        host=env.get("ANVESH_HOST", "127.0.0.1"),
        port=int(env.get("ANVESH_PORT", "8750")),
        model_path=_resolve_model_path(env),
        model_source="finetuned" if env.get("ANVESH_FINETUNED_MODEL") else "pretrained",
        secure_cookies=env.get("ANVESH_SECURE_COOKIES", "0") == "1",
        session_idle_timeout_s=int(env.get("ANVESH_SESSION_IDLE_S", str(30 * 60))),
        require_distinct_approver=env.get("ANVESH_DISTINCT_APPROVER", "1") != "0",
        webhook_url=env.get("ANVESH_ALERT_WEBHOOK_URL") or None,
        webhook_secret=env.get("ANVESH_ALERT_WEBHOOK_SECRET") or None,
        smtp_host=env.get("ANVESH_SMTP_HOST") or None,
        smtp_port=int(env.get("ANVESH_SMTP_PORT", "587")),
        smtp_user=env.get("ANVESH_SMTP_USER") or None,
        smtp_password=env.get("ANVESH_SMTP_PASSWORD") or None,
        smtp_from=env.get("ANVESH_SMTP_FROM") or None,
        registry_base_url=env.get("ANVESH_REGISTRY_URL") or None,
        registry_api_key=env.get("ANVESH_REGISTRY_API_KEY") or None,
        registry_result_max_age_s=int(env.get("ANVESH_REGISTRY_MAX_AGE_S", str(24 * 3600))),
        challan_base_url=env.get("ANVESH_CHALLAN_URL") or None,
        challan_api_key=env.get("ANVESH_CHALLAN_API_KEY") or None,
    )
    return cfg
