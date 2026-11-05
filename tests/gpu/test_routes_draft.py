"""Atelier contract isolation test; runs Mock only without accessing charged GPU."""

import copy
import time

import pytest
from app.config import settings
from app.db import Store
from app.main import app
from app.security import sign_session
from fastapi.testclient import TestClient


@pytest.fixture
def gpu_client(tmp_path, monkeypatch):
    st = Store("sqlite:///" + str(tmp_path / "gpu-api.db"))
    user = st.create("users", "bootstrap", {"username": "gpu-admin", "role": "admin"})
    project = st.create("projects", user["id"], {"title": "โปรเจกต์ตรวจรับ GPU"})
    app.state.store = st
    monkeypatch.setattr(settings, "data_dir", tmp_path / "media")
    monkeypatch.setenv("GPU_REAL_ENABLED", "0")
    monkeypatch.setenv("GPU_STORAGE", "local")
    monkeypatch.setattr(settings, "studio_password", "test-long-login-password")
    monkeypatch.setattr(settings, "session_secret", "test-session-private-signing-key")
    from cryptography.fernet import Fernet

    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    with TestClient(app) as c:
        token = sign_session(
            {"uid": user["id"], "role": "admin", "iat": int(time.time()), "exp": time.time() + 3600}
        )
        c.headers["Authorization"] = "Bearer " + token
        yield c, st, user, project, {}


def request_body(client, project):
    workflows = client.get("/api/gpu/workflows").json()["data"]
    w = next(w for w in workflows if w["state"] == "published" and w["active"])
    return {
        "project_id": project["id"],
        "model_id": w["model_id"],
        "workflow_key": w["workflow_key"],
        "workflow_version": w["workflow_version"],
        "params": {"prompt": "การทดสอบไปป์ไลน์ที่ระบุชัดเจนว่าเป็น Mock"},
        "asset_ids": [],
        "execution_policy": {"provider": "mock", "shutdown_after_job": True, "max_runtime_minutes": 5},
    }


def test_models_require_existing_login(gpu_client):
    c, st, user, project, h = gpu_client
    assert c.get("/api/gpu/models").status_code == 200
    c.headers.pop("Authorization")
    assert c.get("/api/gpu/models").status_code == 401


def test_duplicate_mobile_requests_snapshot_and_project_outputs(gpu_client):
    from app.gpu.repository import Repository
    from app.gpu.scheduler import Scheduler

    c, st, user, project, h = gpu_client
    body = request_body(c, project)
    headers = {**h, "Idempotency-Key": "phone-retry-1"}
    responses = [c.post("/api/gpu/jobs", json=body, headers=headers) for _ in range(5)]
    assert all(r.status_code == 202 for r in responses), [r.text for r in responses]
    ids = {r.json()["data"]["id"] for r in responses}
    assert len(ids) == 1
    jid = ids.pop()
    repo = Repository(st)
    snapshot = copy.deepcopy(repo.get("jobs", jid)["snapshot"])
    for _ in range(22):
        Scheduler(st).tick()
        state = c.get("/api/gpu/jobs/" + jid).json()["data"]["state"]
        if state in {"completed", "failed", "cancelled"}:
            break
    job = c.get("/api/gpu/jobs/" + jid).json()["data"]
    assert job["state"] == "completed", job
    assert repo.get("jobs", jid)["snapshot"] == snapshot
    assert all(lease["state"] == "deleted" for lease in repo.list("leases"))
    outputs = c.get("/api/gpu/jobs/" + jid + "/outputs").json()["data"]
    assert outputs
    assets = st.list("assets", user["id"], project["id"])
    assert assets and assets[0].get("mock") is True
    assert c.get("/api/assets/" + assets[0]["id"] + "/download").status_code == 200
    changed = copy.deepcopy(body)
    changed["params"]["prompt"] = "changed"
    assert c.post("/api/gpu/jobs", json=changed, headers=headers).status_code == 409


def test_invalid_params_and_foreign_project_rejected_before_allocation(gpu_client):
    from app.gpu.repository import Repository

    c, st, user, project, h = gpu_client
    body = request_body(c, project)
    body["params"]["nonexistent_cfg"] = 100
    assert (
        c.post("/api/gpu/jobs", json=body, headers={**h, "Idempotency-Key": "bad-param"}).status_code == 422
    )
    body = request_body(c, project)
    body["project_id"] = st.create("projects", "other-owner", {"title": "private"})["id"]
    assert (
        c.post("/api/gpu/jobs", json=body, headers={**h, "Idempotency-Key": "bad-owner"}).status_code == 404
    )
    assert not Repository(st).list("leases")


def test_cancel_queue_csrf_and_real_gpu_fail_closed(gpu_client):
    c, st, user, project, h = gpu_client
    body = request_body(c, project)
    assert c.post("/api/gpu/jobs", json=body).status_code == 422
    r = c.post("/api/gpu/jobs", json=body, headers={**h, "Idempotency-Key": "cancel"})
    jid = r.json()["data"]["id"]
    assert c.post("/api/gpu/jobs/" + jid + "/cancel", headers=h).json()["data"]["state"] == "cancelled"
    body["execution_policy"]["provider"] = "colab"
    r = c.post("/api/gpu/jobs", json=body, headers={**h, "Idempotency-Key": "real-disabled"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "REAL_GPU_DISABLED"


def test_agent_cannot_publish_or_change_budget(gpu_client, monkeypatch):
    c, st, user, project, h = gpu_client
    monkeypatch.setenv("GPU_AGENT_TOKEN", "test-agent-private-key")
    monkeypatch.setenv("GPU_AGENT_OWNER_ID", user["id"])
    c.headers.pop("Authorization")
    auth = {"Authorization": "Bearer test-agent-private-key"}
    r = c.post(
        "/api/gpu/agent/invoke", json={"tool": "budget.set", "arguments": {"daily": 999}}, headers=auth
    )
    assert r.status_code == 403
    r = c.post(
        "/api/gpu/agent/invoke",
        json={"tool": "workflow.publish", "arguments": {"confirmed": True}},
        headers=auth,
    )
    assert r.status_code == 403
    r = c.post("/api/gpu/agent/invoke", json={"tool": "usage.inspect", "arguments": {}}, headers=auth)
    assert r.status_code == 200
    assert "test-agent-private-key" not in r.text
