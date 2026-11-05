"""Colab v1beta runtime lifecycle + v1 long-running Operations (official discoveries)."""

from __future__ import annotations

import re
import uuid

import httpx

from .google import GoogleError, GoogleService


class ColabError(GoogleError):
    pass


_RUNTIME_ID = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_RUNTIME = re.compile(r"^runtimes/([a-z](?:[a-z0-9-]{0,61}[a-z0-9])?)$")
_OPERATION = re.compile(r"^operations/[A-Za-z0-9._~-]+$")
_VARIANTS = {"VARIANT_CPU", "VARIANT_GPU", "VARIANT_TPU"}
_VARIANT_ALIASES = {"CPU": "VARIANT_CPU", "GPU": "VARIANT_GPU", "TPU": "VARIANT_TPU"}
_SHAPES = {"SHAPE_STANDARD", "SHAPE_HIGHMEM"}
_SHAPE_ALIASES = {"STANDARD": "SHAPE_STANDARD", "HIGHMEM": "SHAPE_HIGHMEM"}


class GoogleColabProvider:
    is_mock = False

    def __init__(
        self,
        store,
        owner_id,
        *,
        client=None,
        google=None,
        base_url="https://colaboratory.googleapis.com",
        timeout=10.0,
    ):
        self.store, self.owner_id = store, owner_id
        self.client = client or httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=False)
        self._owns_client = client is None
        self.google = google or GoogleService(store, owner_id, client=self.client)
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_environment(cls, store=None, owner_id=None, **kwargs):
        if store is None or not owner_id:
            raise ColabError("Colab ต้องการผู้ใช้ที่ผ่านการยืนยันตัวตนแล้วและ Repository ที่คงทน")
        return cls(store, owner_id, **kwargs)

    def _request(self, method, path, *, params=None, json=None):
        token = self.google.access_token("colab")
        try:
            response = self.client.request(
                method,
                self.base_url + path,
                params=params,
                json=json,
                headers={"Authorization": "Bearer " + token, "Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise ColabError("ข้อผิดพลาดเครือข่ายของ Colab API ผลลัพธ์ปลายทางอาจไม่ทราบ จะคงสัญญาเช่าไว้เพื่อประสานข้อมูล") from exc
        if response.status_code >= 400 and response.status_code != 404:
            raise ColabError(f"คำขอ Colab API ล้มเหลว (HTTP {response.status_code}）")
        return response

    @staticmethod
    def _json(response):
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError()
            return payload
        except ValueError as exc:
            raise ColabError("Colab API ส่งกลับ JSON ที่ไม่ถูกต้อง") from exc

    def list_specs(self):
        return self._json(self._request("GET", "/v1beta/runtimespecs")).get("runtimeSpecs", [])

    def subscription(self):
        return self._json(self._request("GET", "/v1beta/subscription"))

    def list_runtimes(self):
        # Never return Google's raw connectionInfo to routes.
        rows = self._json(self._request("GET", "/v1beta/runtimes")).get("runtimes", [])
        return [
            {"name": r.get("name"), "runtimeSpec": r.get("runtimeSpec"), "version": r.get("version")}
            for r in rows
        ]

    def _lease(self, runtime_id, request_id=None, job_id=None):
        for lease in self.store.list("leases", self.owner_id):
            if (
                lease.get("runtime_id") != runtime_id
                or not lease.get("owner_job_id")
                or not lease.get("request_id")
            ):
                continue
            if request_id and request_id != lease.get("request_id"):
                continue
            if job_id and job_id != lease.get("owner_job_id"):
                continue
            if lease.get("provider", "colab") != "colab":
                continue
            return lease
        raise ColabError("ปฏิเสธการดำเนินการกับ runtime ที่ไม่ได้ลงทะเบียนความเป็นเจ้าของ (归属/ownership); จะไม่แตะต้อง runtime ที่ผู้ใช้สร้างเอง")

    def _owned(self, handle):
        if not isinstance(handle, dict) or handle.get("owner_id") != self.owner_id:
            raise ColabError("สัญญาเช่า Colab ความเป็นเจ้าของ (归属/ownership) ไม่ถูกต้อง")
        runtime = handle.get("runtime_name", "")
        match = _RUNTIME.fullmatch(runtime)
        if not match or handle.get("runtime_id") != match.group(1):
            raise ColabError("Handle ของ Colab runtime ไม่ถูกต้อง")
        return self._lease(
            match.group(1), handle.get("request_id"), handle.get("owner_job_id") or handle.get("job_id")
        )

    @staticmethod
    def _spec(spec):
        if not isinstance(spec, dict):
            raise ColabError("runtimeSpec ต้องมาจาก key สเปกที่ใช้งานได้จริงของบัญชีปัจจุบัน")
        if "key" in spec:
            if spec.get("eligible") is not True:
                raise ColabError("runtimeSpec ต้องระบุ eligible=true อย่างชัดเจน")
            key = spec["key"]
        else:
            key = spec
        if not isinstance(key, dict):
            raise ColabError("runtimeSpec ต้องมาจาก key สเปกที่ใช้งานได้จริงของบัญชีปัจจุบัน")
        key = dict(key)
        key["variant"] = _VARIANT_ALIASES.get(key.get("variant"), key.get("variant"))
        key["shape"] = _SHAPE_ALIASES.get(key.get("shape"), key.get("shape"))
        if (
            set(key) != {"variant", "accelerator", "shape"}
            or key.get("variant") not in _VARIANTS
            or key.get("shape") not in _SHAPES
        ):
            raise ColabError("runtimeSpec ต้องมาจาก key สเปกที่ใช้งานได้จริงของบัญชีปัจจุบัน")
        if not isinstance(key.get("accelerator"), str) or not key["accelerator"]:
            raise ColabError("accelerator ของ runtimeSpec ไม่ถูกต้อง")
        return dict(key)

    @staticmethod
    def _raise_operation_error(operation):
        if operation.get("error"):
            code = (operation["error"] or {}).get("code", "unknown")
            # Google's detail may contain credentials or notebook data: expose only safe code.
            raise ColabError(f"การจัดสรร Operation ของ Colab ล้มเหลว (code {code}) โปรดตรวจสอบสิทธิ์ โควตา และสต็อกการ์ดจอ")

    def create(self, spec, request_id, job_id, runtime_id):
        if not isinstance(runtime_id, str) or not _RUNTIME_ID.fullmatch(runtime_id):
            raise ColabError("runtimeId ต้องเป็น RFC1034 identifier ที่ขึ้นต้นด้วยตัวอักษร")
        try:
            request_uuid = uuid.UUID(request_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ColabError("requestId ของ Colab ต้องเป็น UUID4") from exc
        if request_uuid.version != 4:
            raise ColabError("requestId ของ Colab ต้องเป็น UUID4")
        lease = self._lease(runtime_id, request_id, job_id)
        key = self._spec(spec)
        response = self._request(
            "POST",
            "/v1beta/runtimes",
            params={"requestId": request_id, "runtimeId": runtime_id},
            json={"runtimeSpec": key},
        )
        operation = self._json(response)
        name = operation.get("name")
        if not isinstance(name, str) or not _OPERATION.fullmatch(name):
            raise ColabError("การตอบกลับการสร้างขาด Operation ที่ถูกต้อง โดยจะคง runtimeId ไว้เพื่อรอประสานข้อมูล")
        self.store.update(
            "leases",
            lease["id"],
            {"operation_name": name, "operation_done": operation.get("done") is True, "requested_spec": key},
            self.owner_id,
        )
        self._raise_operation_error(operation)
        return {
            "runtime_name": "runtimes/" + runtime_id,
            "operation_name": name,
            "ready": operation.get("done") is True,
            "runtime_id": runtime_id,
            "request_id": request_id,
            "owner_id": self.owner_id,
        }

    def _operation(self, handle, lease):
        name = handle.get("operation_name") or lease.get("operation_name")
        if not name:
            return None
        if not _OPERATION.fullmatch(name):
            raise ColabError("ชื่อ Operation ไม่ถูกต้อง")
        response = self._request("GET", "/v1/" + name)
        if response.status_code == 404:
            # Missing operation is not proof of runtime deletion; inspect runtime below.
            return None
        return self._json(response)

    def inspect(self, handle):
        lease = self._owned(handle)
        operation = self._operation(handle, lease)
        if operation is not None:
            if operation.get("done") is not True:
                return {"ready": False, "deleted": False, "operation_pending": True}
            self._raise_operation_error(operation)
        response = self._request("GET", "/v1beta/" + handle["runtime_name"])
        if response.status_code == 404:
            # No Operation + a previously attempted create whose response was lost is ambiguous.
            unknown = (
                operation is None
                and int(lease.get("create_attempts", 0)) > 0
                and not lease.get("operation_done")
            )
            return {"ready": False, "deleted": not unknown, "allocation_unknown": unknown}
        runtime = self._json(response)
        return {
            "ready": bool(runtime.get("connectionInfo")),
            "deleted": False,
            "runtime_name": handle["runtime_name"],
            "runtime_spec": runtime.get("runtimeSpec"),
            "version": runtime.get("version"),
        }

    def connection_info(self, handle):
        self._owned(handle)
        response = self._request("GET", "/v1beta/" + handle["runtime_name"])
        info = self._json(response).get("connectionInfo", {})
        if not info.get("url") or not info.get("token"):
            raise ColabError("connectionInfo ของ Colab ยังไม่พร้อมใช้งาน")
        return {"url": info["url"], "token": info["token"], "expire_time": info.get("expireTime")}

    def delete(self, handle):
        lease = self._owned(handle)
        operation = self._operation(handle, lease)
        if operation is not None and operation.get("done") is not True:
            # Deleting a missing runtime during a still-pending create is not release confirmation.
            return False
        # A failed completed Operation may still have allocated transient state; GET/DELETE regardless.
        current = self._request("GET", "/v1beta/" + handle["runtime_name"])
        if current.status_code == 404:
            if (
                operation is None
                and int(lease.get("create_attempts", 0)) > 0
                and not lease.get("operation_done")
            ):
                # Reconcile lost create response by looking for only our deterministic runtime metadata.
                response = self._request("GET", "/v1/operations")
                operations = self._json(response).get("operations", []) if response.status_code != 404 else []
                matches = [
                    o
                    for o in operations
                    if (o.get("metadata") or {}).get("runtime") == handle["runtime_name"]
                ]
                if not matches or any(o.get("done") is not True for o in matches):
                    return False
            return True
        self._request("DELETE", "/v1beta/" + handle["runtime_name"], params={"allowMissing": "true"})
        return self._request("GET", "/v1beta/" + handle["runtime_name"]).status_code == 404

    def close(self):
        if self._owns_client:
            self.client.close()
