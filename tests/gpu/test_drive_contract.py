from __future__ import annotations

import hashlib

import httpx
from app.gpu.drive import DriveError, DriveStorage
from app.security import encrypt_secret


class Repo:
    def __init__(self):
        self.rows = {"credentials": [], "files": []}
        self.rows["credentials"].append(
            {
                "id": "cred",
                "owner_id": "owner-1",
                "provider": "google",
                "purpose": "drive",
                "encrypted_access_token": encrypt_secret("access"),
                "encrypted_refresh_token": encrypt_secret("refresh"),
                "expires_at": 9999999999,
            }
        )

    def list(self, kind, owner_id=None, **_):
        return [dict(x) for x in self.rows.get(kind, []) if owner_id is None or x["owner_id"] == owner_id]

    def get(self, kind, record_id, owner_id=None):
        return next(
            (
                dict(x)
                for x in self.rows.get(kind, [])
                if x["id"] == record_id and (owner_id is None or x["owner_id"] == owner_id)
            ),
            None,
        )

    def create(self, kind, owner_id, data, **_):
        row = {"id": f"{kind}-{len(self.rows.setdefault(kind, [])) + 1}", "owner_id": owner_id, **data}
        self.rows[kind].append(row)
        return dict(row)

    def update(self, kind, record_id, patch, owner_id=None):
        for row in self.rows.get(kind, []):
            if row["id"] == record_id and (owner_id is None or row["owner_id"] == owner_id):
                row.update(patch)
                return dict(row)
        raise KeyError(record_id)


def test_put_creates_hierarchy_uploads_sha256_and_verifies_media():
    content = b"hello drive"
    file_id = "file-1"
    ids = {"AI-Studio": "root-1", "owner-1": "owner-folder", "job-1": "job-folder"}

    def handler(request):
        path = request.url.path
        if path == "/drive/v3/files" and request.method == "GET":
            name = request.url.params["q"].split("name = '", 1)[1].split("'", 1)[0]
            return httpx.Response(
                200,
                json={
                    "files": [
                        {"id": ids[name], "name": name, "mimeType": "application/vnd.google-apps.folder"}
                    ]
                    if name in ids
                    else []
                },
            )
        if path == "/drive/v3/files" and request.method == "POST":
            name = (
                request.content.decode().split('"name": "', 1)[1].split('"', 1)[0]
                if b'"name": "' in request.content
                else "folder"
            )
            return httpx.Response(
                200,
                json={
                    "id": ids.get(name, name + "-folder"),
                    "name": name,
                    "mimeType": "application/vnd.google-apps.folder",
                },
            )
        if path == "/upload/drive/v3/files" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "id": file_id,
                    "name": "result.bin",
                    "size": str(len(content)),
                    "appProperties": {"atelier_sha256": hashlib.sha256(content).hexdigest()},
                },
            )
        if path == f"/drive/v3/files/{file_id}" and request.url.params.get("alt") == "media":
            return httpx.Response(200, content=content)
        raise AssertionError(f"unexpected {request.method} {request.url}")

    storage = DriveStorage(
        {"id": "job-1", "owner_id": "owner-1"},
        Repo(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    record = storage.put("outputs/result.bin", content, "application/octet-stream")
    assert record["id"] == file_id and record["verified"] is True
    assert record["sha256"] == hashlib.sha256(content).hexdigest()


def test_verify_rejects_same_size_wrong_content():
    def handler(request):
        return httpx.Response(200, content=b"wrong")

    storage = DriveStorage(
        "job-1", Repo(), "owner-1", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    record = {"id": "file-1", "size": 5, "sha256": hashlib.sha256(b"hello").hexdigest()}
    assert storage.verify(record, b"hello") is False


def test_put_uses_business_key_without_job_id_and_is_idempotent():
    content = b"workflow"
    digest = hashlib.sha256(content).hexdigest()
    calls = []
    ids = {"AI-Studio": "root", "owner-1": "owner", "workflows": "workflows"}

    def handler(request):
        calls.append(request)
        if request.url.path == "/drive/v3/files" and request.method == "GET":
            query = request.url.params["q"]
            name = query.split("name = '", 1)[1].split("'", 1)[0]
            if "appProperties has" in query:
                return httpx.Response(200, json={"files": []})
            return httpx.Response(
                200, json={"files": [{"id": ids[name], "name": name}] if name in ids else []}
            )
        if request.url.path == "/drive/v3/files" and request.method == "POST":
            return httpx.Response(200, json={"id": "folder-new", "name": "workflows"})
        if request.url.path == "/upload/drive/v3/files":
            return httpx.Response(
                200,
                json={
                    "id": "remote-1",
                    "name": "manifest.json",
                    "size": str(len(content)),
                    "appProperties": {
                        "studio_key_hash": hashlib.sha256(b"workflows/demo/1/manifest.json").hexdigest(),
                        "studio_sha256": digest,
                    },
                },
            )
        if request.url.path == "/drive/v3/files/remote-1":
            return httpx.Response(200, content=content)
        raise AssertionError(f"unexpected {request.method} {request.url}")

    repo = Repo()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    storage = DriveStorage({"id": "job-1", "owner_id": "owner-1"}, repo, client=client)
    first = storage.put("workflows/demo/1/manifest.json", content, "application/json")
    post_count = sum(1 for request in calls if request.method == "POST" and "/upload/" in request.url.path)
    second = storage.put("workflows/demo/1/manifest.json", content, "application/json")
    assert first["id"] == second["id"] == "remote-1"
    assert post_count == 1
    assert not any(
        "'job-1' in parents" in request.url.params.get("q", "")
        for request in calls
        if request.method == "GET"
    )
    assert any(
        row.get("kind") == "drive_root" and row.get("remote_id") == "root" for row in repo.rows["files"]
    )


def test_resumable_rejects_uncontrolled_location_before_bearer_is_sent():
    content = b"x" * (8 * 1024 * 1024 + 1)
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET" and request.url.path == "/drive/v3/files":
            return httpx.Response(200, json={"files": []})
        if request.method == "POST" and request.url.path == "/drive/v3/files":
            return httpx.Response(200, json={"id": "folder", "name": "folder"})
        if request.method == "POST" and request.url.path == "/upload/drive/v3/files":
            return httpx.Response(
                200, headers={"Location": "https://evil.example/upload/drive/v3/session"}, json={}
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    storage = DriveStorage(
        "job-1", Repo(), "owner-1", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    try:
        storage.put("outputs/2026/01/job-1/result.bin", content, "application/octet-stream")
    except DriveError as exc:
        assert "会话地址" in str(exc) or "เซสชัน" in str(exc)
    else:
        raise AssertionError("uncontrolled resumable location was accepted")
    assert all(request.url.host != "evil.example" for request in seen)


def test_resumable_upload_checks_308_range_and_resumes():
    content = b"0123456789abcdef" * 2
    sha256 = hashlib.sha256(content).hexdigest()
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "GET" and request.url.path == "/drive/v3/files":
            return httpx.Response(200, json={"files": []})
        if request.method == "POST" and request.url.path == "/drive/v3/files":
            return httpx.Response(200, json={"id": "folder", "name": "folder"})
        if request.method == "POST" and request.url.path == "/upload/drive/v3/files":
            return httpx.Response(
                200,
                headers={"Location": "https://www.googleapis.com/upload/drive/v3/files?upload_id=test"},
                json={},
            )
        if request.method == "PUT" and request.url.path.startswith("/upload/drive/v3/files"):
            content_range = request.headers.get("Content-Range", "")
            if content_range == "bytes 0-31/32":
                return httpx.Response(308, headers={"Range": "bytes=0-15"})
            assert content_range == "bytes 16-31/32"
            return httpx.Response(
                201,
                json={
                    "id": "remote-video",
                    "name": "clip.mp4",
                    "size": "32",
                    "appProperties": {"studio_sha256": sha256},
                },
            )
        if request.method == "GET" and request.url.path == "/drive/v3/files/remote-video":
            return httpx.Response(200, content=content)
        raise AssertionError(f"unexpected {request.method} {request.url}")

    storage = DriveStorage(
        "job-1", Repo(), "owner-1", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    result = storage.put("outputs/2026/01/job-1/clip.mp4", content, "video/mp4")
    assert result["id"] == "remote-video" and result["verified"] is True
    assert [request.headers.get("Content-Range") for request in calls if request.method == "PUT"] == [
        "bytes 0-31/32",
        "bytes 16-31/32",
    ]
