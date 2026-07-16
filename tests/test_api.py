from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sqlite3

from fastapi.testclient import TestClient

from app.config import settings
from app.core.deepstream_ingestor import DeepStreamIngestor
from app.runtime import build_runtime


def test_camera_table_imports_json_once_and_becomes_authoritative(tmp_path: Path) -> None:
    seed_path = tmp_path / "sources.json"
    database_path = tmp_path / "cameras.sqlite3"
    seed_path.write_text(
        json.dumps(
            [
                {
                    "source_id": "camera-seed",
                    "name": "Seed camera",
                    "enabled": True,
                    "tasks": ["fire_smoke"],
                    "source_uri": "rtsp://user:secret@example.test/live",
                    "metadata": {"area": "north"},
                }
            ]
        ),
        encoding="utf-8",
    )
    test_settings = replace(
        settings,
        processor_mode="mock",
        camera_db_path=database_path,
        source_registry_path=seed_path,
        plate_log_db_path=tmp_path / "plate_logs.sqlite3",
        video_ingestion_enabled=False,
    )

    first_runtime = build_runtime(test_settings)
    try:
        imported = first_runtime.registry.require("camera-seed")
        assert imported.name == "Seed camera"
        assert imported.source_uri == "rtsp://user:secret@example.test/live"
    finally:
        first_runtime.close()

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cameras)").fetchall()
        }
        assert {
            "camera_id",
            "name",
            "enabled",
            "tasks_json",
            "source_uri",
            "metadata_json",
            "created_at_utc",
            "updated_at_utc",
        }.issubset(columns)
        assert connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0] == 1

    seed_path.write_text("[]", encoding="utf-8")
    second_runtime = build_runtime(test_settings)
    try:
        assert second_runtime.registry.require("camera-seed").name == "Seed camera"
    finally:
        second_runtime.close()


def test_camera_crud_emits_online_websocket_events_and_keeps_source_alias(
    tmp_path: Path,
) -> None:
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            camera_db_path=tmp_path / "cameras.sqlite3",
            source_registry_path=tmp_path / "missing-sources.json",
            plate_log_db_path=tmp_path / "plate_logs.sqlite3",
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        with TestClient(main_module.app) as client:
            with client.websocket_connect("/api/v1/broadcast/ws") as websocket:
                created = client.post(
                    "/api/v1/cameras",
                    json={
                        "camera_id": "camera-live",
                        "name": "Live camera",
                        "enabled": True,
                        "tasks": ["fire_smoke"],
                        "source_uri": "rtsp://operator:secret@example.test/live",
                        "metadata": {"area": "gate"},
                    },
                )
                assert created.status_code == 201
                assert created.json()["camera_id"] == "camera-live"
                assert "secret" not in created.json()["source_uri"]
                event = websocket.receive_json()
                assert event == {
                    "type": "camera_changed",
                    "action": "created",
                    "camera_id": "camera-live",
                    "revision": event["revision"],
                    "camera": {
                        "camera_id": "camera-live",
                        "name": "Live camera",
                        "enabled": True,
                        "tasks": ["fire_smoke"],
                        "updated_at_utc": event["camera"]["updated_at_utc"],
                    },
                }

                source_alias = client.get("/api/v1/sources/camera-live")
                assert source_alias.status_code == 200
                assert source_alias.json()["source_id"] == "camera-live"
                assert "secret" not in source_alias.json()["source_uri"]

                updated = client.patch(
                    "/api/v1/cameras/camera-live",
                    json={
                        "name": "Updated camera",
                        "tasks": ["fire_smoke", "plate_recognition"],
                    },
                )
                assert updated.status_code == 200
                assert updated.json()["name"] == "Updated camera"
                assert set(updated.json()["tasks"]) == {
                    "fire_smoke",
                    "plate_recognition",
                }
                assert websocket.receive_json()["action"] == "updated"

                replaced = client.put(
                    "/api/v1/cameras/camera-live",
                    json={
                        "name": "Replacement camera",
                        "enabled": False,
                        "tasks": ["plate_recognition"],
                        "source_uri": "data/replacement.mp4",
                        "metadata": {},
                    },
                )
                assert replaced.status_code == 200
                assert replaced.json()["enabled"] is False
                assert websocket.receive_json()["camera"]["enabled"] is False

                deleted = client.delete("/api/v1/cameras/camera-live")
                assert deleted.status_code == 204
                deleted_event = websocket.receive_json()
                assert deleted_event["action"] == "deleted"
                assert deleted_event["camera"] is None
                assert client.get("/api/v1/cameras/camera-live").status_code == 404
    finally:
        main_module.runtime = old_runtime


def test_source_control_api_uses_persistent_registry(tmp_path: Path) -> None:
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            camera_db_path=tmp_path / "cameras.sqlite3",
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
            assert "/api/v1/cameras" in response.text
            assert "/api/v1/broadcast/ws" in response.text

            response = client.get("/", follow_redirects=False)
            assert response.status_code in {302, 307}
            assert response.headers["location"] == "/dashboard"

            response = client.get("/openapi.json")
            assert response.status_code == 200
            assert "/dashboard" in response.json()["paths"]
            assert "/api/v1/cameras" in response.json()["paths"]
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


def test_runtime_selects_deepstream_backend_without_loading_plugins(tmp_path: Path) -> None:
    runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            camera_db_path=tmp_path / "cameras.sqlite3",
            source_registry_path=tmp_path / "sources.json",
            plate_log_db_path=tmp_path / "plate_logs.sqlite3",
            video_ingest_backend="deepstream",
        )
    )
    try:
        assert isinstance(runtime.video_ingestor, DeepStreamIngestor)
        assert runtime.video_ingestor.status()["backend"] == "deepstream"
    finally:
        runtime.close()
