from __future__ import annotations

import httpx
import pytest
from app.gpu.colab import ColabError, GoogleColabProvider
from app.security import encrypt_secret


class MemoryRepository:
    def __init__(self):
        self.rows = {"credentials": [], "leases": []}
        self.create(
            "credentials",
            "owner-1",
            {
                "provider": "google",
                "purpose": "colab",
                "encrypted_access_token": encrypt_secret("access"),
                "encrypted_refresh_token": encrypt_secret("refresh"),
                "expires_at": 9999999999,
            },
        )

    def create(self, kind, owner_id, data, **kwargs):
        row = {"id": f"{kind}-{len(self.rows.setdefault(kind, [])) + 1}", "owner_id": owner_id, **data}
        self.rows[kind].append(row)
        return dict(row)

    def get(self, kind, record_id, owner_id=None):
        return next(
            (
                dict(r)
                for r in self.rows.get(kind, [])
                if r["id"] == record_id and (owner_id is None or r["owner_id"] == owner_id)
            ),
            None,
        )

    def list(self, kind, owner_id=None, **kwargs):
        return [dict(r) for r in self.rows.get(kind, []) if owner_id is None or r["owner_id"] == owner_id]

    def update(self, kind, record_id, patch, owner_id=None):
        row = self.get(kind, record_id, owner_id)
        if row is None:
            raise KeyError(record_id)
        for current in self.rows[kind]:
            if current["id"] == record_id:
                current.update(patch)
                return dict(current)
        raise KeyError(record_id)


def make(handler):
    repo = MemoryRepository()
    repo.create(
        "leases",
        "owner-1",
        {
            "provider": "colab",
            "runtime_id": "atelier-runtime-1",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "owner_job_id": "job-1",
            "state": "creating",
        },
    )
    return GoogleColabProvider(repo, "owner-1", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_discovery_read_only_calls():
    paths = []

    def handler(request):
        paths.append((request.method, request.url.path))
        if request.url.path.endswith("runtimespecs"):
            return httpx.Response(200, json={"runtimeSpecs": [{"eligible": True}]})
        return httpx.Response(200, json={"name": "subscription", "tier": "SUBSCRIPTION_TIER_PRO"})

    provider = make(handler)
    assert provider.list_specs()[0]["eligible"] is True
    assert provider.subscription()["tier"] == "SUBSCRIPTION_TIER_PRO"
    assert paths == [("GET", "/v1beta/runtimespecs"), ("GET", "/v1beta/subscription")]


def test_create_matches_discovery_and_persists_operation():
    def handler(request):
        assert request.method == "POST" and request.url.path == "/v1beta/runtimes"
        assert request.url.params["requestId"] == "11111111-1111-4111-8111-111111111111"
        assert request.url.params["runtimeId"] == "atelier-runtime-1"
        return httpx.Response(
            200,
            json={
                "name": "operations/op-1",
                "done": True,
                "response": {"name": "runtimes/atelier-runtime-1"},
            },
        )

    provider = make(handler)
    result = provider.create(
        {
            "key": {"variant": "GPU", "accelerator": "T4", "shape": "STANDARD"},
            "eligible": True,
        },
        "11111111-1111-4111-8111-111111111111",
        "job-1",
        "atelier-runtime-1",
    )
    assert result == {
        "runtime_name": "runtimes/atelier-runtime-1",
        "operation_name": "operations/op-1",
        "ready": True,
        "runtime_id": "atelier-runtime-1",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "owner_id": "owner-1",
    }


def test_captured_real_spec_uses_official_enums_and_requires_eligibility():
    captured_specs = [
        {
            "key": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"},
            "eligible": True,
        },
        {
            "key": {"variant": "VARIANT_GPU", "accelerator": "H100", "shape": "SHAPE_STANDARD"},
        },
    ]

    assert GoogleColabProvider._spec(captured_specs[0]) == {
        "variant": "VARIANT_GPU",
        "accelerator": "T4",
        "shape": "SHAPE_STANDARD",
    }
    with pytest.raises(ColabError, match="eligible"):
        GoogleColabProvider._spec(captured_specs[1])


def test_captured_real_spec_create_payload_preserves_official_enums():
    captured_spec = {
        "key": {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_HIGHMEM"},
        "eligible": True,
    }

    def handler(request):
        assert request.method == "POST" and request.url.path == "/v1beta/runtimes"
        assert request.content == (
            b'{"runtimeSpec":{"variant":"VARIANT_GPU","accelerator":"A100","shape":"SHAPE_HIGHMEM"}}'
        )
        return httpx.Response(200, json={"name": "operations/op-captured", "done": True})

    result = make(handler).create(
        captured_spec,
        "11111111-1111-4111-8111-111111111111",
        "job-1",
        "atelier-runtime-1",
    )
    assert result["operation_name"] == "operations/op-captured"


def test_inspect_without_operation_uses_runtime_get():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        assert request.url.path == "/v1beta/runtimes/atelier-runtime-1"
        return httpx.Response(
            200,
            json={
                "name": "runtimes/atelier-runtime-1",
                "connectionInfo": {"url": "https://runtime", "token": "secret"},
            },
        )

    result = make(handler).inspect(
        {
            "runtime_name": "runtimes/atelier-runtime-1",
            "runtime_id": "atelier-runtime-1",
            "owner_id": "owner-1",
        }
    )
    assert result["ready"] is True and "token" not in str(result)
    assert paths == ["/v1beta/runtimes/atelier-runtime-1"]


def test_delete_requires_owner_and_get_404_confirmation():
    paths = []

    def handler(request):
        paths.append((request.method, request.url.path))
        if request.method == "DELETE":
            assert request.url.params["allowMissing"] == "true"
            return httpx.Response(204)
        return httpx.Response(404)

    provider = make(handler)
    assert (
        provider.delete(
            {
                "runtime_name": "runtimes/atelier-runtime-1",
                "runtime_id": "atelier-runtime-1",
                "owner_id": "owner-1",
            }
        )
        is True
    )
    # An already-absent owned runtime is confirmed by GET, without an unnecessary DELETE.
    assert paths == [("GET", "/v1beta/runtimes/atelier-runtime-1")]
    with pytest.raises(Exception, match="ความเป็นเจ้าของ|归属"):
        provider.delete({"runtime_name": "runtimes/other", "runtime_id": "other", "owner_id": "other-owner"})
