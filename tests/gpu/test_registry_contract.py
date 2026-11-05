from concurrent.futures import ThreadPoolExecutor

import pytest
from app.db import Store
from app.gpu.registry import Registry
from app.gpu.repository import Repository, ensure_schema
from sqlalchemy import inspect


def _store(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'gpu.db'}")
    ensure_schema(store)
    return store


def test_independent_store_has_basic_tables_and_crud(tmp_path):
    store = _store(tmp_path)
    tables = set(inspect(store.engine).get_table_names())
    assert {"users", "projects", "assets", "settings", "provider_credentials"}.issubset(tables)
    row = store.create("projects", "owner", {"title": "x", "owner_id": "spoof"})
    assert row["owner_id"] == "owner"
    assert store.get("projects", row["id"], "other") is None
    assert store.update("projects", row["id"], {"title": "y"}, "owner")["title"] == "y"
    assert store.delete("projects", row["id"], "owner") is True


def test_gpu_repository_uses_isolated_tables_and_acl(tmp_path):
    store = _store(tmp_path)
    tables = set(inspect(store.engine).get_table_names())
    assert {"gpu_models", "gpu_workflows", "gpu_jobs", "gpu_events"}.issubset(tables)
    repo = Repository(store)
    row = repo.create("models", "owner", {"name": "demo", "enabled": True})
    assert repo.get("models", row["id"], "owner")["name"] == "demo"
    assert repo.get("models", row["id"], "other") is None
    assert repo.update("models", row["id"], {"enabled": False}, "owner")["enabled"] is False


def test_create_job_is_owner_scoped_and_rejects_different_raw_body(tmp_path):
    repo = Repository(_store(tmp_path))
    body = {"workflow_key": "demo", "workflow_version": 1, "params": {"seed": -1}}
    first, created = repo.create_job("owner", body, "same-key")
    assert created is True
    second, created = repo.create_job("owner", body, "same-key")
    assert not created and first["id"] == second["id"]
    assert repo.create_job("other", {**body, "params": {"seed": 2}}, "same-key")[1] is True
    try:
        repo.create_job("owner", {**body, "params": {"seed": 3}}, "same-key")
    except ValueError:
        pass
    else:
        raise AssertionError("same owner+key with another body must be rejected")


def test_create_job_is_atomic_under_concurrency(tmp_path):
    repo = Repository(_store(tmp_path))
    body = {"workflow_key": "demo", "workflow_version": 1, "params": {"seed": -1}}

    def submit(_):
        return repo.create_job("owner", body, "concurrent")

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(submit, range(6)))
    assert len({item[0]["id"] for item in results}) == 1 and sum(item[1] for item in results) == 1


def test_event_is_append_only_and_owner_scoped(tmp_path):
    repo = Repository(_store(tmp_path))
    job, _ = repo.create_job("owner", {"workflow_key": "demo"}, "event-key")
    event = repo.event(job["id"], "owner", "running", "started", 0.25)
    assert event["state"] == "running" and repo.list("events", "owner")[0]["message"] == "started"


def test_registry_bootstrap_marks_demo_published_and_real_models_disabled(tmp_path):
    registry = Registry(Repository(_store(tmp_path)))
    result = registry.bootstrap("owner")
    assert result["demo_workflow"]["state"] == "published"
    models = {m["model_id"]: m for m in registry.models("owner")}
    assert models["demo-mock"]["enabled"] is True
    for model_id in ("moss", "krea", "h3"):
        assert models[model_id]["enabled"] is False and models[model_id]["disabled_reason"]


def test_registry_versions_are_immutable_and_rollback_switches_active(tmp_path):
    repo = Repository(_store(tmp_path))
    registry = Registry(repo)
    registry.bootstrap("owner")
    bundle = registry.get_workflow("owner", "demo-mock", 1)
    imported = registry.import_workflow("owner", bundle)
    assert imported["state"] == "draft"
    validated = registry.validate("owner", imported["id"])
    assert validated["state"] == "validated"
    repo.update("workflows", imported["id"], {"smoke_test_passed": True, "smoke_test_kind": "mock"}, "owner")
    published = registry.publish("owner", imported["id"], confirmed=True)
    assert published["state"] == "published" and published["workflow_version"] != bundle["workflow_version"]
    assert registry.rollback("owner", "demo-mock", 1)["active"] is True


def test_registry_import_respects_semver_manifest_and_auto_increments_legacy_versions(tmp_path):
    repo = Repository(_store(tmp_path))
    registry = Registry(repo)
    registry.bootstrap("owner")
    bundle = registry.get_workflow("owner", "demo-mock", 1)
    semver = {**bundle, "manifest": {**bundle["manifest"], "version": "2.4.0"}}
    imported = registry.import_workflow("owner", semver)
    assert imported["workflow_version"] == "2.4.0"
    legacy = registry.import_workflow("owner", bundle)
    assert legacy["workflow_version"] == 2


def test_registry_validate_allows_disabled_real_model_but_publish_requires_license(tmp_path):
    repo = Repository(_store(tmp_path))
    registry = Registry(repo)
    repo.create("models", "owner", {"model_id": "real", "enabled": False, "is_mock": False})
    source = registry._template()
    row = repo.create(
        "workflows",
        "owner",
        {
            "workflow_key": "real",
            "workflow_version": 1,
            "model_id": "real",
            "manifest": source[0],
            "api_graph": source[1],
            "object_info": source[2],
            "editable_graph": source[1],
            "state": "draft",
        },
    )
    assert registry.validate("owner", row["id"])["state"] == "validated"
    repo.update("models", repo.list("models", "owner")[0]["id"], {"enabled": True}, "owner")
    repo.update(
        "workflows",
        row["id"],
        {"smoke_test_passed": True, "smoke_test_kind": "real", "review_approved": True},
        "owner",
    )
    with pytest.raises(ValueError, match="สัญญาอนุญาต|license|许可"):
        registry.publish("owner", row["id"], confirmed=True)


def test_workflow_owner_key_version_is_database_unique(tmp_path):
    repo = Repository(_store(tmp_path))
    data = {"workflow_key": "same", "workflow_version": 1, "model_id": "demo"}
    repo.create("workflows", "owner", data)
    with pytest.raises(ValueError, match="เอกลักษณ์|unique|唯一|存在"):
        repo.create("workflows", "owner", data)
    assert repo.create("workflows", "other", data)["owner_id"] == "other"


def test_registry_bootstrap_is_idempotent_under_concurrency(tmp_path):
    repo = Repository(_store(tmp_path))

    def bootstrap(_):
        return Registry(repo).bootstrap("owner")

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(bootstrap, range(6)))
    assert len(results) == 6
    registry = Registry(repo)
    assert len([row for row in registry.models("owner") if row["model_id"] == "demo-mock"]) == 1
    assert len([row for row in registry.workflows("owner") if row["workflow_key"] == "demo-mock"]) == 1
