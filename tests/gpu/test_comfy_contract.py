from __future__ import annotations

import pytest
from app.gpu.comfy import ComfyError, ComfyGPUExecutor


def test_comfy_fails_closed_without_verified_connection():
    with pytest.raises(ComfyError, match="connectionInfo"):
        ComfyGPUExecutor().start({}, {"snapshot": {}}, "execution-1")


def test_comfy_rejects_unreviewed_workflow_before_network():
    executor = ComfyGPUExecutor(base_url="https://runtime.example", token="token")
    with pytest.raises(ComfyError, match="การตรวจสอบ|审查"):
        executor.start(
            {
                "connection_url": "https://runtime.example",
                "connection_token": "token",
                "jupyter_url": "https://runtime.example",
            },
            {"snapshot": {"manifest": {}}},
            "execution-1",
        )
