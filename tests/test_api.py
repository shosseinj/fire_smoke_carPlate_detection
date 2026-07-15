from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.runtime import build_runtime


def test_source_control_api_uses_persistent_registry(tmp_path: Path) -> None:
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            source_registry_path=tmp_path / "sources.json",
            plate_log_db_path=tmp_path / "plate_logs.sqlite3",
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        with TestClient(main_module.app) as client:
            response = client.get("/api/v1/sources")
            assert response.status_code == 200
            assert len(response.json()) == 8

            response = client.get("/dashboard")
            assert response.status_code == 200
            assert "Video AI Operations Wall" in response.text
            assert "synchronizeDashboard" in response.text

            response = client.get("/", follow_redirects=False)
            assert response.status_code in {302, 307}
            assert response.headers["location"] == "/dashboard"

            response = client.get("/openapi.json")
            assert response.status_code == 200
            assert "/dashboard" in response.json()["paths"]
            assert "/api/v1/plate-logs" in response.json()["paths"]

            response = client.post(
                "/api/v1/plate-logs",
                json={
                    "camera_id": "camera-01",
                    "time": "2026-07-15T08:15:30+03:30",
                    "plate": "23ن92917",
                },
            )
            assert response.status_code == 201
            assert response.json()["camera"] == "camera-01"
            assert response.json()["plate"] == "23ن92917"

            response = client.get(
                "/api/v1/plate-logs",
                params={"camera_id": "camera-01", "limit": 10},
            )
            assert response.status_code == 200
            assert len(response.json()) == 1
            assert response.json()[0]["plate"] == "23ن92917"

            response = client.get("/api/v1/broadcast/state")
            assert response.status_code == 200
            assert response.json()["enabled"] is True

            response = client.put("/api/v1/broadcast/state", json={"enabled": False})
            assert response.status_code == 200
            assert response.json()["enabled"] is False

            response = client.get("/api/v1/broadcast/snapshots/camera-01.jpg")
            assert response.status_code == 503

            response = client.put("/api/v1/broadcast/state", json={"enabled": True})
            assert response.status_code == 200

            response = client.post("/api/v1/sources/camera-01/disable")
            assert response.status_code == 200
            assert response.json()["enabled"] is False

            response = client.put(
                "/api/v1/sources/bulk/task-assignment",
                json={
                    "source_ids": ["camera-05", "camera-06"],
                    "tasks": ["plate_recognition", "fire_smoke"],
                    "enabled": True,
                },
            )
            assert response.status_code == 200
            assert all(len(item["tasks"]) == 2 for item in response.json())
        assert test_runtime.broadcast.enabled is False
    finally:
        main_module.runtime = old_runtime
