"""Service container: builds every application service from one `AppConfig`.

Kept as a plain attribute bag (no DI framework) so tests can build a full stack against a
temporary directory and swap individual adapters."""

from __future__ import annotations

import os

from anvesh.app.accounts import AccountService
from anvesh.app.alerts import AlertRouter, EmailAlertAdapter, TestAlertAdapter, WebhookAlertAdapter
from anvesh.app.audit import AuditLog
from anvesh.app.auth import AuthService
from anvesh.app.cameras import CameraService
from anvesh.app.cases import CaseService
from anvesh.app.config import REPO_ROOT, AppConfig
from anvesh.app.db import AppDatabase
from anvesh.app.evidence import EvidenceStore
from anvesh.app.handoff import ExportPackageAdapter, HandoffService, OfficialChallanAdapter, SandboxChallanAdapter
from anvesh.app.incidents import IncidentService
from anvesh.app.plates import PlateService, load_ocr_engine, load_plate_detector
from anvesh.app.registry import ProductionRegistryAdapter, RegistryService, SandboxRegistryAdapter
from anvesh.app.retention import RetentionService
from anvesh.app.settings import SettingsService


class Services:
    def __init__(self, config: AppConfig, ocr=None, plate_detector=None, sandbox_challan=None) -> None:
        self.config = config
        config.data_dir.mkdir(parents=True, exist_ok=True)
        for d in (config.evidence_dir, config.uploads_dir, config.outbox_dir, config.exports_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db = AppDatabase(config.db_path)
        self.audit = AuditLog(self.db)
        self.auth = AuthService(self.db, self.audit, session_ttl_s=config.session_ttl_s, idle_timeout_s=config.session_idle_timeout_s)
        self.accounts = AccountService(self.db, self.audit, self.auth)
        self.settings = SettingsService(self.db)
        extra_roots = tuple(p for p in os.environ.get("ANVESH_MEDIA_ROOTS", "").split(os.pathsep) if p)
        self.media_roots = (config.uploads_dir, REPO_ROOT) + tuple(__import__("pathlib").Path(p) for p in extra_roots)
        # The data directory (database, credentials file, evidence, exports) is protected: file sources may not point into it, except
        # into uploads_dir, which is allowed explicitly above.
        self.cameras = CameraService(self.db, self.audit, media_roots=self.media_roots, protected_roots=(config.data_dir,))
        self.incidents = IncidentService(self.db, self.audit)
        self.evidence = EvidenceStore(self.db, self.audit, config.evidence_dir)
        self.retention = RetentionService(self.db, self.audit, self.evidence, self.settings, config.evidence_retention_days)
        plate_weights = os.environ.get("ANVESH_PLATE_MODEL", str(REPO_ROOT / "models" / "plate_detector.pt"))
        self.plates = PlateService(
            self.db, self.audit, self.evidence,
            ocr=ocr if ocr is not None else load_ocr_engine(os.environ.get("ANVESH_OCR_MODEL_DIR") or None),
            detector=plate_detector if plate_detector is not None else load_plate_detector(plate_weights),
            require_distinct=config.require_distinct_approver)
        self.registry = RegistryService(
            self.db, self.audit, SandboxRegistryAdapter(REPO_ROOT / "configs" / "sandbox_registry.json"),
            ProductionRegistryAdapter(config.registry_base_url, config.registry_api_key))
        self.cases = CaseService(self.db, self.audit, self.incidents, self.evidence, self.plates, self.registry, self.settings,
                                 require_distinct=config.require_distinct_approver, registry_max_age_s=config.registry_result_max_age_s)
        self.handoff = HandoffService(self.db, self.audit, self.cases, {
            "export": ExportPackageAdapter(self),
            "sandbox": sandbox_challan or SandboxChallanAdapter(),
            "official": OfficialChallanAdapter(config.challan_base_url, config.challan_api_key),
        })
        self.alerts = AlertRouter(self.db, self.audit, self.incidents, {
            "test": TestAlertAdapter(config.outbox_dir),
            "webhook": WebhookAlertAdapter(config.webhook_url, config.webhook_secret),
            "email": EmailAlertAdapter(config.smtp_host, config.smtp_port, config.smtp_user, config.smtp_password, config.smtp_from),
        }, self.settings)

    def close(self) -> None:
        self.alerts.stop_worker()
        self.db.close()


def build_services(config: AppConfig, **kwargs) -> Services:
    return Services(config, **kwargs)
