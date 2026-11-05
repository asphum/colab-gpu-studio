"""Independent GPU repository backed by app.db; no Atelier imports."""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Mapping

from sqlalchemy import String, UniqueConstraint, inspect, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Mapped, mapped_column

from app import db as db_module
from app.db import Base, CommonMixin, Store, utcnow

from .manifest import digest


class GpuModelRow(CommonMixin, Base):
    __tablename__ = "gpu_models"


class GpuWorkflowRow(CommonMixin, Base):
    __tablename__ = "gpu_workflows"
    workflow_key: Mapped[str] = mapped_column(String(255), nullable=False)
    # Stored as text so both legacy integer versions and semver versions have
    # one database identity.  _row converts digit-only legacy values back to int.
    workflow_version: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "owner_id", "workflow_key", "workflow_version", name="uq_gpu_workflows_owner_key_version"
        ),
    )


class GpuJobRow(CommonMixin, Base):
    __tablename__ = "gpu_jobs"
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (UniqueConstraint("owner_id", "idempotency_key", name="uq_gpu_jobs_owner_idempotency"),)


class GpuLeaseRow(CommonMixin, Base):
    __tablename__ = "gpu_leases"


class GpuEventRow(CommonMixin, Base):
    __tablename__ = "gpu_events"


class GpuPresetRow(CommonMixin, Base):
    __tablename__ = "gpu_presets"


class GpuUsageRow(CommonMixin, Base):
    __tablename__ = "gpu_usage"


class GpuFileRow(CommonMixin, Base):
    __tablename__ = "gpu_files"


class GpuCredentialRow(CommonMixin, Base):
    __tablename__ = "gpu_credentials"


_GPU_MODELS = {
    "models": GpuModelRow,
    "workflows": GpuWorkflowRow,
    "jobs": GpuJobRow,
    "leases": GpuLeaseRow,
    "events": GpuEventRow,
    "presets": GpuPresetRow,
    "usage": GpuUsageRow,
    "files": GpuFileRow,
    "credentials": GpuCredentialRow,
}
_GPU_TABLES = [model.__table__ for model in _GPU_MODELS.values()]
for _kind, _model in _GPU_MODELS.items():
    db_module.MODELS.setdefault("gpu_" + _kind, _model)


def ensure_schema(store: Store) -> None:
    for attempt in range(6):
        try:
            with store.db_lock():
                Base.metadata.create_all(store.engine, tables=_GPU_TABLES, checkfirst=True)
                # create_all does not migrate a database made by an older
                # build.  Add the two identity columns and the unique index
                # idempotently before any workflow can be written.
                columns = {column["name"] for column in inspect(store.engine).get_columns("gpu_workflows")}
                with store.engine.begin() as connection:
                    if "workflow_key" not in columns:
                        connection.execute(
                            text("ALTER TABLE gpu_workflows ADD COLUMN workflow_key VARCHAR(255)")
                        )
                    if "workflow_version" not in columns:
                        connection.execute(
                            text("ALTER TABLE gpu_workflows ADD COLUMN workflow_version VARCHAR(64)")
                        )
                    if store.engine.dialect.name in {"sqlite", "postgresql"}:
                        connection.execute(
                            text(
                                "CREATE UNIQUE INDEX IF NOT EXISTS uq_gpu_workflows_owner_key_version ON gpu_workflows (owner_id, workflow_key, workflow_version)"
                            )
                        )
            return
        except OperationalError:
            if attempt == 5:
                raise
            time.sleep(0.03 * (attempt + 1))


class Repository:
    def __init__(self, store: Store):
        self.store = store
        ensure_schema(store)

    def dispatch_lock(self):
        """Expose the Store's cross-process tick lock to scheduler callers."""
        return self.store.dispatch_lock()

    @staticmethod
    def _model(kind):
        try:
            return _GPU_MODELS[kind]
        except KeyError as exc:
            raise ValueError(f"ไม่รู้จักประเภทเอนทิตี GPU (unknown GPU entity type): {kind}") from exc

    @staticmethod
    def _row(row):
        if row is None:
            return None
        reserved = {"id", "owner_id", "project_id", "state", "version", "created_at", "updated_at"}
        result = {
            "id": row.id,
            "owner_id": row.owner_id,
            "project_id": row.project_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
        result.update({k: copy.deepcopy(v) for k, v in (row.data or {}).items() if k not in reserved})
        if isinstance(row, GpuWorkflowRow):
            workflow_version = row.workflow_version
            if isinstance(workflow_version, str) and workflow_version.isdigit():
                workflow_version = int(workflow_version)
            result["workflow_key"] = row.workflow_key
            result["workflow_version"] = workflow_version
        if row.state is not None:
            result["state"] = row.state
        if row.version:
            result["version"] = row.version
        if isinstance(row, GpuJobRow):
            result.update(idempotency_key=row.idempotency_key, body_hash=row.body_hash)
        return result

    def create(self, kind, owner_id, data, project_id=None, record_id=None):
        model = self._model(kind)
        if not isinstance(data, Mapping):
            raise ValueError("data ของระเบียน GPU ต้องเป็น object (dict)")  # noqa: TRY004
        payload = copy.deepcopy(dict(data))
        rid = record_id or str(uuid.uuid4())
        effective_project = project_id if project_id is not None else payload.get("project_id")
        workflow_key = workflow_version = None
        if kind == "workflows":
            workflow_key = payload.get("workflow_key")
            workflow_version = payload.get("workflow_version")
            if not isinstance(workflow_key, str) or not workflow_key:
                raise ValueError("จำเป็นต้องระบุ workflow_key")
            if not (
                (
                    isinstance(workflow_version, int)
                    and not isinstance(workflow_version, bool)
                    and workflow_version > 0
                )
                or (isinstance(workflow_version, str) and workflow_version)
            ):
                raise ValueError("รูปแบบ workflow_version ไม่ถูกต้อง")
        if kind == "files" and effective_project is None:
            storage_key = payload.get("storage_key")
            parts = storage_key.split("/") if isinstance(storage_key, str) else []
            if len(parts) >= 2 and parts[0] == "gpu":
                source_job = self.get("jobs", parts[1], owner_id)
                effective_project = source_job.get("project_id") if source_job else None
        kwargs = {
            "id": rid,
            "owner_id": owner_id,
            "project_id": effective_project,
            "state": payload.get("state"),
            "data": payload,
            "version": 0,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        if model is GpuWorkflowRow:
            kwargs.update(workflow_key=workflow_key, workflow_version=str(workflow_version))
        if model is GpuJobRow:
            key = payload.get("idempotency_key") or f"__direct__:{rid}"
            kwargs.update(
                idempotency_key=key,
                body_hash=payload.get("body_hash") or digest(payload.get("request_body", payload)),
            )
        try:
            with self.store.db_lock(), self.store.Session.begin() as session:
                row = model(**kwargs)
                session.add(row)
                session.flush()
                return self._row(row)
        except IntegrityError as exc:
            raise ValueError("ระเบียน GPU มีอยู่แล้วหรือละเมิดข้อจำกัดความเป็นเอกลักษณ์ (unique / 唯一 / 存在)") from exc

    def get(self, kind, id, owner_id=None):
        model = self._model(kind)
        with self.store.db_lock(), self.store.Session() as session:
            stmt = select(model).where(model.id == id)
            if owner_id is not None:
                stmt = stmt.where(model.owner_id == owner_id)
            return self._row(session.scalar(stmt))

    def list(self, kind, owner_id=None, project_id=None):
        model = self._model(kind)
        with self.store.db_lock(), self.store.Session() as session:
            stmt = select(model).order_by(model.created_at.desc())
            if owner_id is not None:
                stmt = stmt.where(model.owner_id == owner_id)
            if project_id is not None:
                stmt = stmt.where(model.project_id == project_id)
            return [self._row(row) for row in session.scalars(stmt).all()]

    def update(self, kind, id, patch, owner_id=None):
        model = self._model(kind)
        if not isinstance(patch, Mapping):
            raise ValueError("patch ต้องเป็น object (dict)")  # noqa: TRY004
        with self.store.db_lock(), self.store.Session.begin() as session:
            stmt = select(model).where(model.id == id)
            if owner_id is not None:
                stmt = stmt.where(model.owner_id == owner_id)
            row = session.scalar(stmt)
            if row is None:
                raise KeyError(id)
            if kind == "workflows" and row.state == "published" and set(patch) - {"active"}:
                raise ValueError("เวอร์ชันเวิร์กโฟลว์ที่เผยแพร่แล้วไม่สามารถเขียนทับได้")
            changes = copy.deepcopy(dict(patch))
            row.data = {**(row.data or {}), **changes}
            if "state" in changes:
                row.state = changes["state"]
            if "project_id" in changes:
                row.project_id = changes["project_id"]
            if kind == "workflows":
                if "workflow_key" in changes:
                    if not isinstance(changes["workflow_key"], str) or not changes["workflow_key"]:
                        raise ValueError("รูปแบบ workflow_key ไม่ถูกต้อง")
                    row.workflow_key = changes["workflow_key"]
                if "workflow_version" in changes:
                    value = changes["workflow_version"]
                    if not (
                        (isinstance(value, int) and not isinstance(value, bool) and value > 0)
                        or (isinstance(value, str) and value)
                    ):
                        raise ValueError("รูปแบบ workflow_version ไม่ถูกต้อง")
                    row.workflow_version = str(value)
            row.version += 1
            row.updated_at = utcnow()
            session.flush()
            return self._row(row)

    def create_job(self, owner_id, body, idempotency_key):
        if not isinstance(body, Mapping):
            raise ValueError("job body ต้องเป็น object (dict)")  # noqa: TRY004
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("จำเป็นต้องระบุ idempotency_key")
        raw = copy.deepcopy(dict(body))
        body_hash = digest(raw)
        for attempt in range(8):
            try:
                with self.store.db_lock(), self.store.Session.begin() as session:
                    existing = session.scalar(
                        select(GpuJobRow).where(
                            GpuJobRow.owner_id == owner_id, GpuJobRow.idempotency_key == idempotency_key
                        )
                    )
                    if existing:
                        if existing.body_hash != body_hash:
                            raise ValueError("idempotency_key ของ owner เดียวกันถูกใช้กับ body อื่นไปแล้ว")
                        return self._row(existing), False
                    policy = raw.get("execution_policy") or {}
                    payload = {
                        **raw,
                        "request_body": raw,
                        "idempotency_key": idempotency_key,
                        "body_hash": body_hash,
                        "state": "queued",
                        "provider": policy.get("provider", "mock"),
                        "workflow_key": raw.get("workflow_key"),
                        "workflow_version": raw.get("workflow_version"),
                        "model_id": raw.get("model_id"),
                        "params": copy.deepcopy(raw.get("params", {})),
                        "resolved_seed": None,
                        "snapshot": None,
                        "max_runtime_minutes": policy.get("max_runtime_minutes"),
                        "estimated_units": None,
                        "unit_rate": None,
                        "requested_outcome": None,
                        "cancel_requested": False,
                        "output_asset_ids": [],
                    }
                    row = GpuJobRow(
                        id=str(uuid.uuid4()),
                        owner_id=owner_id,
                        project_id=raw.get("project_id"),
                        state="queued",
                        data=payload,
                        version=0,
                        idempotency_key=idempotency_key,
                        body_hash=body_hash,
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                    session.add(row)
                    session.flush()
                    return self._row(row), True
            except IntegrityError:
                if attempt == 7:
                    raise ValueError("การแย่งชิงสร้าง GPU job ล้มเหลว กรุณาลองใหม่")
                time.sleep(0.01 * (attempt + 1))
            except OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 7:
                    raise
                time.sleep(0.02 * (attempt + 1))
        raise RuntimeError("การสร้าง GPU job ยังไม่เสร็จสมบูรณ์")

    def event(self, job_id, owner_id, state, message, progress=None):
        job = self.get("jobs", job_id, owner_id)
        if job is None:
            raise KeyError(job_id)
        if progress is not None and (
            isinstance(progress, bool) or not isinstance(progress, (int, float)) or not 0 <= progress <= 1
        ):
            raise ValueError("progress ต้องอยู่ระหว่าง 0 ถึง 1")
        data = {"job_id": job_id, "message": message, "state": state}
        if progress is not None:
            data["progress"] = progress
        return self.create("events", owner_id, data, project_id=job.get("project_id"))
