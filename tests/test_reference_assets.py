import io

from PIL import Image

from tests.gpu.test_routes_draft import gpu_client  # noqa: F401


def test_reference_assets_survive_reload_and_exclude_mock(gpu_client):  # noqa: F811
    client, store, user, project, headers = gpu_client
    raw = io.BytesIO()
    Image.new("RGB", (32, 24), "#987654").save(raw, format="PNG")
    response = client.post(
        "/api/assets",
        data={"project_id": project["id"]},
        files={"file": ("reference_image.png", raw.getvalue(), "image/png")},
    )
    assert response.status_code == 201, response.text
    asset = response.json()["data"]
    store.create(
        "assets",
        user["id"],
        {
            "project_id": project["id"],
            "mime": "image/png",
            "filename": "mock-output.png",
            "metadata": {"mock": True},
        },
    )
    result = client.get("/api/assets", params={"project_id": project["id"]})
    assert result.status_code == 200
    assert [row["id"] for row in result.json()["data"]] == [asset["id"]]
    assert "storage_key" not in result.json()["data"][0]
    assert client.get("/api/assets?project_id=foreign-project").status_code == 404
