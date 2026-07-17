from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sqlite3

import numpy as np

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
                        "frame_width": 640,
                        "frame_height": 640,
                        "updated_at_utc": event["camera"]["updated_at_utc"],
                    },
                }

                source_alias = client.get("/api/v1/sources/camera-live")
                assert source_alias.status_code == 200
                assert source_alias.json()["source_id"] == "camera-live"
                assert source_alias.json()["frame_width"] == 640
                assert source_alias.json()["frame_height"] == 640
                assert "secret" not in source_alias.json()["source_uri"]

                updated = client.patch(
                    "/api/v1/cameras/camera-live",
                    json={
                        "name": "Updated camera",
                        "tasks": ["fire_smoke", "plate_recognition"],
                        "frame_width": 960,
                        "frame_height": 544,
                    },
                )
                assert updated.status_code == 200
                assert updated.json()["name"] == "Updated camera"
                assert updated.json()["frame_width"] == 960
                assert updated.json()["frame_height"] == 544
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


def test_swagger_organizes_diagnostics_and_model_test_sections(tmp_path: Path) -> None:
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            camera_db_path=tmp_path / "cameras.sqlite3",
            source_registry_path=tmp_path / "missing.json",
            plate_log_db_path=tmp_path / "logs.sqlite3",
            saved_media_path=tmp_path / "media",
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        with TestClient(main_module.app) as client:
            overview = client.get("/api/v1/diagnostics/overview")
            assert overview.status_code == 200
            model_configuration = overview.json()["model_configuration"]
            assert model_configuration["plate_pipeline"] == "vehicle -> plate -> OCR"
            assert model_configuration["fire_minimum_score"] == 0.3
            assert model_configuration["smoke_minimum_score"] == 0.3
            assert model_configuration["plate_minimum_score"] == 0.3
            assert model_configuration["vehicle_minimum_score"] == 0.35
            assert model_configuration["vehicle_detector"]["class_ids"] == [2, 3, 5, 7]
            assert overview.json()["plate_detection_policy"]["plate_confidence"] == 0.3
            checks = client.get("/api/v1/diagnostics/checks")
            assert checks.status_code == 200
            assert {item["section"] for item in checks.json()["checks"]} >= {
                "camera_registry",
                "deepstream",
                "fire_smoke_model",
                "vehicle_model",
                "plate_model",
                "persistent_logs",
            }
            schema = client.get("/openapi.json").json()
            assert "/api/v1/diagnostics/checks" in schema["paths"]
            assert "/api/v1/frame-rounds/jpeg" in schema["paths"]
            assert "/api/v1/plate-settings/general" in schema["paths"]
            assert "/api/v1/plate-settings/cameras/{camera_id}" in schema["paths"]
            assert "/api/v1/settings/general" in schema["paths"]
            assert "/api/v1/models/artifacts" in schema["paths"]
            assert "/api/v1/models/conversions" in schema["paths"]
            assert "/api/v1/cameras/{camera_id}/tasks" in schema["paths"]
            assert any(tag["name"] == "plate-settings" for tag in schema["tags"])
            assert schema["tags"][0]["name"] == "system-diagnostics"
    finally:
        main_module.runtime = old_runtime


def test_general_model_settings_and_play_only_camera_api(tmp_path: Path) -> None:
    import app.main as main_module

    model_root = tmp_path / "weights"
    fire_model = model_root / "fire_smoke/fire_nano.pt"
    vehicle_model = model_root / "vehicle_detector/yolo11n.pt"
    plate_model = model_root / "plate_detector/plate_small.pt"
    for model in (fire_model, vehicle_model, plate_model):
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"test")

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            camera_db_path=tmp_path / "cameras.sqlite3",
            source_registry_path=tmp_path / "missing.json",
            plate_log_db_path=tmp_path / "logs.sqlite3",
            saved_media_path=tmp_path / "media",
            model_root_path=model_root,
            fire_model_path=fire_model,
            vehicle_detector_weights=vehicle_model,
            plate_detector_weights=plate_model,
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        with TestClient(main_module.app) as client:
            artifacts = client.get("/api/v1/models/artifacts", params={"format": "pt"})
            assert artifacts.status_code == 200
            assert {item["variant"] for item in artifacts.json()} == {
                "nano",
                "small",
            }

            updated = client.patch(
                "/api/v1/settings/general",
                json={
                    "models": {"preferred_format": "pt"},
                    "plate_detection": {"plate_confidence": 0.52},
                },
            )
            assert updated.status_code == 200
            assert updated.json()["models"]["preferred_format"] == "pt"
            assert updated.json()["plate_detection"]["plate_confidence"] == 0.52
            assert updated.json()["camera_processing"]["modes"]["play_only"] == []

            camera_id = test_runtime.registry.list()[0].source_id
            play_only = client.put(
                f"/api/v1/cameras/{camera_id}/tasks",
                json={"tasks": []},
            )
            assert play_only.status_code == 200
            assert play_only.json()["tasks"] == []

            summary = test_runtime.router.submit_round(
                frames=[np.zeros((64, 64, 3), dtype=np.uint8)],
                source_ids=[camera_id],
                round_sequence=999,
            )
            assert summary["task_submissions"] == 0
            latest = test_runtime.broadcast.latest(camera_id)
            assert latest is not None
            assert latest.tasks == ()
    finally:
        main_module.runtime = old_runtime


def test_plate_settings_api_applies_general_and_camera_inheritance(
    tmp_path: Path,
) -> None:
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            camera_db_path=tmp_path / "cameras.sqlite3",
            source_registry_path=tmp_path / "missing.json",
            plate_log_db_path=tmp_path / "logs.sqlite3",
            saved_media_path=tmp_path / "media",
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        with TestClient(main_module.app) as client:
            camera_id = test_runtime.registry.list()[0].source_id
            inherited = client.get(
                f"/api/v1/plate-settings/cameras/{camera_id}"
            )
            assert inherited.status_code == 200
            assert inherited.json()["overrides"] == {}

            overridden = client.patch(
                f"/api/v1/plate-settings/cameras/{camera_id}",
                json={"plate_confidence": 0.61},
            )
            assert overridden.status_code == 200
            assert overridden.json()["effective"]["plate_confidence"] == 0.61
            assert "vehicle_confidence" in overridden.json()["inherited_fields"]

            general = client.put(
                "/api/v1/plate-settings/general",
                json={
                    "vehicle_confidence": 0.41,
                    "plate_confidence": 0.47,
                    "ocr_confidence": 0.55,
                    "min_vehicle_width_pixels": 150,
                    "min_vehicle_height_pixels": 90,
                    "min_vehicle_area_ratio": 0.03,
                    "vehicle_crop_padding_ratio": 0.04,
                },
            )
            assert general.status_code == 200

            effective = client.get(
                f"/api/v1/plate-settings/cameras/{camera_id}"
            ).json()["effective"]
            assert effective["plate_confidence"] == 0.61
            assert effective["vehicle_confidence"] == 0.41
            assert effective["min_vehicle_width_pixels"] == 150

            cleared = client.patch(
                f"/api/v1/plate-settings/cameras/{camera_id}",
                json={"plate_confidence": None},
            )
            assert cleared.status_code == 200
            assert cleared.json()["effective"]["plate_confidence"] == 0.47
            assert "plate_confidence" in cleared.json()["inherited_fields"]
    finally:
        main_module.runtime = old_runtime
