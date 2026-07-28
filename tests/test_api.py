from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import logging
import os
import time

import numpy as np
import pytest

from fastapi.testclient import TestClient

from app.config import settings
from app.core.deepstream_ingestor import DeepStreamIngestor
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.types import TaskName
from app.database import get_database, metadata
from app.runtime import build_runtime


pytestmark = pytest.mark.usefixtures("postgres_database")


def _test_database_url() -> str:
    return os.environ["TEST_DATABASE_URL"]


def test_source_registry_upsert_persists_source_fps() -> None:
    database = get_database(_test_database_url())
    registry = SourceRegistry(database)
    try:
        registry.upsert(
            SourceRecord(
                source_uri="data/upsert-fps.mp4",
                name="Upsert FPS",
                fps=37.5,
            )
        )
    finally:
        registry.close()

    reloaded = SourceRegistry(database)
    try:
        assert reloaded.require("data/upsert-fps.mp4").fps == 37.5
    finally:
        reloaded.close()


def test_fire_smoke_worker_uses_configured_queue_policy() -> None:
    runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            database_url=_test_database_url(),
            video_ingestion_enabled=False,
        )
    )
    try:
        fire_worker = runtime.router.workers[TaskName.FIRE_SMOKE]
        plate_worker = runtime.router.workers[TaskName.PLATE_RECOGNITION]
        assert fire_worker.buffer.stats().policy == settings.task_queue_policy
        assert plate_worker.buffer.stats().policy == settings.task_queue_policy
    finally:
        runtime.close()


def test_source_table_imports_json_once_and_becomes_authoritative(tmp_path: Path) -> None:

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
        database_url=_test_database_url(),

        video_ingestion_enabled=False,
    )

    first_runtime = build_runtime(test_settings)
    try:
        imported = first_runtime.registry.require("camera-seed")
        assert imported.name == "Seed camera"
        assert imported.source_uri == "rtsp://user:secret@example.test/live"
    finally:
        first_runtime.close()

    columns = set(metadata.tables["sources"].c.keys())
    with get_database(_test_database_url()).connection() as connection:
        assert {
            "id",
            "name",
            "enabled",
            "tasks_json",
            "source_uri",
            "metadata_json",
            "created_at_utc",
            "updated_at_utc",
        }.issubset(columns)
        assert connection.execute("SELECT COUNT(*) FROM sources WHERE source_uri != '__default__'").fetchone()[0] == 1

    seed_path.write_text("[]", encoding="utf-8")
    second_runtime = build_runtime(test_settings)
    try:
        assert second_runtime.registry.require("camera-seed").name == "Seed camera"
    finally:
        second_runtime.close()


def test_source_crud_emits_online_websocket_events_and_allows_renaming(
    tmp_path: Path,
) -> None:
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            database_url=_test_database_url(),
            
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    restart_calls: list[tuple[str, str | None]] = []
    original_restart = test_runtime._restart_ingestor_source

    def restart_spy(source_uri: str, *, previous_source_uri: str | None = None) -> None:
        restart_calls.append((source_uri, previous_source_uri))
        original_restart(source_uri, previous_source_uri=previous_source_uri)

    test_runtime._restart_ingestor_source = restart_spy  # type: ignore[method-assign]
    try:
        with TestClient(main_module.app) as client:
            with client.websocket_connect("/api/v1/broadcast/ws") as websocket:
                created = client.post(
                    "/api/v1/sources",
                    json={
                        "name": "Live source",
                        "source_uri": "data/live.mp4",
                        "source_type": "static_video",
                        "fps": 12.5,
                        "loop": False,
                        "draw_human": False,
                        "draw_zone": False,
                        "metadata": {"area": "gate"},
                    },
                )
                assert created.status_code == 201
                assert created.json()["source_uri"] == "data/live.mp4"
                assert created.json()["fps"] == 12.5
                assert created.json()["loop"] is False
                assert created.json()["draw_human"] is False
                assert created.json()["draw_zone"] is False
                event = websocket.receive_json()
                if event.get("type") != "camera_changed":
                    event = websocket.receive_json()
                assert event == {
                    "type": "camera_changed",
                    "action": "created",
                    "source_uri": "data/live.mp4",
                    "previous_source_uri": None,
                    "revision": event["revision"],
                    "camera": {
                        "source_uri": "data/live.mp4",
                        "name": "Live source",
                        "enabled": True,
                        "tasks": [],
                        "frame_width": 640,
                        "frame_height": 640,
                        "fps": 12.5,
                        "loop": False,
                        "draw_human": False,
                        "draw_zone": False,
                        "draw_fire": True,
                        "draw_smoke": True,
                        "draw_vehicle": True,
                        "draw_plate": True,
                        "updated_at_utc": event["camera"]["updated_at_utc"],
                    },
                }

                source_alias = client.get("/api/v1/sources/data/live.mp4")
                assert source_alias.status_code == 200
                assert source_alias.json()["id"] == created.json()["id"]
                assert source_alias.json()["frame_width"] == 640
                assert source_alias.json()["frame_height"] == 640
                assert source_alias.json()["source_uri"] == "data/live.mp4"
                assert source_alias.json()["fps"] == 12.5
                assert source_alias.json()["loop"] is False
                assert source_alias.json()["draw_human"] is False
                assert source_alias.json()["draw_zone"] is False

                updated = client.patch(
                    "/api/v1/sources/data/live.mp4",
                    json={
                        "name": "Updated source",
                        "source_uri": "data/renamed.mp4",
                        "frame_width": 960,
                        "frame_height": 544,
                        "fps": 6.0,
                        "draw_fire": False,
                    },
                )
                assert updated.status_code == 200
                assert updated.json()["name"] == "Updated source"
                assert updated.json()["frame_width"] == 960
                assert updated.json()["frame_height"] == 544
                assert updated.json()["fps"] == 6.0
                assert updated.json()["draw_fire"] is False
                updated_event = websocket.receive_json()
                if updated_event.get("type") != "camera_changed":
                    updated_event = websocket.receive_json()
                assert updated_event["action"] == "updated"
                assert updated_event["source_uri"] == "data/renamed.mp4"
                assert updated_event["previous_source_uri"] == "data/live.mp4"
                assert client.get("/api/v1/sources/data/live.mp4").status_code == 404
                assert client.get("/api/v1/sources/data/renamed.mp4").status_code == 200
                assert any(
                    source_uri == "data/renamed.mp4" and previous == "data/live.mp4"
                    for source_uri, previous in restart_calls
                )

                replaced = client.put(
                    "/api/v1/sources/data/renamed.mp4",
                    json={
                        "name": "Replacement source",
                        "source_uri": "data/replacement.mp4",
                        "metadata": {},
                        "source_type": "static_video",
                    },
                )
                assert replaced.status_code == 200

                deleted = client.delete("/api/v1/sources/data/replacement.mp4")
                assert deleted.status_code == 204
                deleted_event = websocket.receive_json()
                assert deleted_event["action"] == "deleted"
                assert deleted_event["camera"] is None
                assert client.get("/api/v1/sources/data/replacement.mp4").status_code == 404
    finally:
        main_module.runtime = old_runtime


def test_broadcast_zone_refresh_uses_source_registry_source_uris(tmp_path: Path) -> None:
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            database_url=_test_database_url(),
            
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        recorded: list[tuple[str, list[list[list[float]]]]] = []
        cleared: list[str] = []

        test_runtime.broadcast.set_source_zones = lambda source_uri, zones: recorded.append((source_uri, zones))  # type: ignore[method-assign]
        test_runtime.broadcast.clear_source_zones = lambda source_uri: cleared.append(source_uri)  # type: ignore[method-assign]

        test_runtime._refresh_all_source_zones()

        source_uris = [record.source_uri for record in test_runtime.registry.list()]
        assert set(source_uris)
        assert all(uri in source_uris for uri, _ in recorded)
        assert all(uri in source_uris for uri in cleared)
    finally:
        main_module.runtime = old_runtime


def test_single_and_bulk_source_updates(tmp_path: Path) -> None:
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            database_url=_test_database_url(),
            
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        with TestClient(main_module.app) as client:
            source_ids = [record.source_uri for record in test_runtime.registry.list()[:2]]

            single = client.patch(
                f"/api/v1/sources/{source_ids[0]}",
                json={"name": "Single update"},
            )
            assert single.status_code == 200
            assert single.json()["name"] == "Single update"
            assert "metadata" in single.json()
            assert "updated_at_utc" in single.json()

            bulk = client.put(
                "/api/v1/sources/bulk/task-assignment",
                json=[
                    {
                        "source_id": source_ids[0],
                    },
                    {
                        "source_id": source_ids[1],
                    },
                ],
            )
            assert bulk.status_code == 422

            bulk = client.put(
                "/api/v1/sources/bulk/task-assignment",
                json={
                    "source_ids": [source_ids[0], source_ids[1]],
                    "tasks": ["plate_recognition", "fire_smoke"],
                    "enabled": True,
                },
            )
            assert bulk.status_code == 200
            assert [item["source_uri"] for item in bulk.json()] == source_ids
            assert all(item["tasks"] == ["fire_smoke", "plate_recognition"] for item in bulk.json())
            assert all(item["enabled"] is True for item in bulk.json())

            rejected = client.put(
                "/api/v1/sources/bulk/task-assignment",
                json={
                    "source_ids": [source_ids[0], "missing-source"],
                    "tasks": ["plate_recognition"],
                },
            )
            assert rejected.status_code == 404
            assert client.get(f"/api/v1/sources/{source_ids[0]}").json()["name"] == (
                "Single update"
            )

            schema = client.get("/openapi.json")
            assert schema.status_code == 200
            assert "/api/v1/sources/bulk/task-assignment" in schema.json()["paths"]
            assert "/api/v1/cameras" not in schema.json()["paths"]
    finally:
        main_module.runtime = old_runtime
        test_runtime.close()


def test_bulk_update_sources(tmp_path: Path) -> None:
    """PUT /api/v1/sources/bulk accepts a list of {id, ...fields} and updates all."""
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            database_url=_test_database_url(),
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        with TestClient(main_module.app) as client:
            records = test_runtime.registry.list()
            assert len(records) >= 2, "Need at least 2 sources for bulk test"
            id_a = records[0].id
            id_b = records[1].id

            # Bulk update two sources: change name and confidence overrides
            resp = client.put(
                "/api/v1/sources/bulk",
                json=[
                    {
                        "id": id_a,
                        "name": "Bulk A",
                        "tasks": ["plate_recognition"],
                        "fps": 7.5,
                        "draw_smoke": False,
                        "fire_confidence": 0.8,
                    },
                    {
                        "id": id_b,
                        "name": "Bulk B",
                        "enabled": False,
                        "smoke_confidence": 0.9,
                        "frame_width": 320,
                        "frame_height": 320,
                        "draw_plate": False,
                    },
                ],
            )
            assert resp.status_code == 200, resp.text
            items = resp.json()
            assert len(items) == 2
            by_id = {item["id"]: item for item in items}
            assert by_id[id_a]["name"] == "Bulk A"
            assert by_id[id_a]["tasks"] == ["plate_recognition"]
            assert by_id[id_a]["fps"] == 7.5
            assert by_id[id_a]["draw_smoke"] is False
            assert by_id[id_b]["name"] == "Bulk B"
            assert by_id[id_b]["enabled"] is False
            assert by_id[id_b]["frame_width"] == 320
            assert by_id[id_b]["frame_height"] == 320
            assert by_id[id_b]["draw_plate"] is False
            # Verify persisted via GET
            get_a = client.get(f"/api/v1/sources/{id_a}")
            assert get_a.status_code == 200
            assert get_a.json()["name"] == "Bulk A"
            assert get_a.json()["fps"] == 7.5
            assert get_a.json()["draw_smoke"] is False

            native_fps = client.patch(
                f"/api/v1/sources/{id_a}",
                json={"fps": None},
            )
            assert native_fps.status_code == 200
            assert native_fps.json()["fps"] is None
            assert client.get(f"/api/v1/sources/{id_a}").json()["fps"] is None

            # Missing id returns 404
            resp_missing = client.put(
                "/api/v1/sources/bulk",
                json=[{"id": 999999, "name": "Nope"}],
            )
            assert resp_missing.status_code == 404

            # Empty list returns 400
            resp_empty = client.put("/api/v1/sources/bulk", json=[])
            assert resp_empty.status_code == 400

            # Schema registration
            schema = client.get("/openapi.json")
            assert schema.status_code == 200
            assert "/api/v1/sources/bulk" in schema.json()["paths"]
    finally:
        main_module.runtime = old_runtime
        test_runtime.close()


def test_source_control_api_uses_persistent_registry(tmp_path: Path) -> None:
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            database_url=_test_database_url(),
            
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
            assert "/api/v1/sources/preview-config" in response.text
            assert "/api/v1/broadcast/ws" in response.text
            assert "Fullscreen wall" in response.text
            assert "data-fullscreen-source" in response.text
            assert "Browser RX: 0.0 FPS" in response.text
            assert "function receivedFps" in response.text

            response = client.get("/", follow_redirects=False)
            assert response.status_code in {302, 307}
            assert response.headers["location"] == "/dashboard"

            response = client.get("/openapi.json")
            assert response.status_code == 200
            assert "/dashboard" in response.json()["paths"]
            assert "/api/v1/cameras" not in response.json()["paths"]
            assert "/api/v1/sources" in response.json()["paths"]
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
            database_url=_test_database_url(),
            
            video_ingest_backend="deepstream",
        )
    )
    try:
        from app.core.rtsp_process_supervisor import RtspProcessSupervisor

        assert isinstance(runtime.video_ingestor, RtspProcessSupervisor)
        assert isinstance(runtime.static_video_ingestor, DeepStreamIngestor)
        assert (
            runtime.video_ingestor.status()["backend"]
            == "deepstream-process-isolated"
        )
        assert runtime.static_video_ingestor.source_type_filter == "static_video"
        assert runtime.video_ingestor.demand_controller is runtime.stream_demand
        assert (
            runtime.static_video_ingestor.demand_controller
            is runtime.stream_demand
        )
    finally:
        runtime.close()


def test_swagger_organizes_diagnostics_and_model_test_sections(tmp_path: Path) -> None:
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            database_url=_test_database_url(),
            
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
            assert model_configuration["plate_class_ids"] == [0]
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
            assert "/api/v1/models/artifacts/content" in schema["paths"]
            assert "/api/v1/models/conversions" in schema["paths"]
            assert "/api/v1/models/engine-exports" in schema["paths"]
            assert "/api/v1/faces/enroll" in schema["paths"]
            assert "/api/v1/faces/identities" in schema["paths"]
            assert "/api/v1/faces/quality-settings" in schema["paths"]
            assert "/api/v1/humans/logs" in schema["paths"]
            assert "/api/v1/humans/active" in schema["paths"]
            export_body = schema["paths"]["/api/v1/models/engine-exports"]["post"][
                "requestBody"
            ]["content"]
            assert "multipart/form-data" in export_body
            assert "/api/v1/sources/preview-config" in schema["paths"]
            assert any(tag["name"] == "plate-settings" for tag in schema["tags"])
            assert any(tag["name"] == "face-recognition" for tag in schema["tags"])
            assert any(tag["name"] == "human-tracking" for tag in schema["tags"])
            assert schema["tags"][0]["name"] == "system-diagnostics"

            quality = client.patch(
                "/api/v1/faces/quality-settings",
                json={
                    "quality_threshold": 0.73,
                    "min_face_width": 40,
                    "min_face_height": 48,
                    "max_abs_pitch": 70.0,
                },
            )
            assert quality.status_code == 200
            assert quality.json()["quality_threshold"] == 0.73
            assert quality.json()["min_face_width"] == 40
            assert quality.json()["min_face_height"] == 48
            assert quality.json()["max_abs_pitch"] == 70.0
            assert client.get("/api/v1/faces/quality-settings").json() == quality.json()
    finally:
        main_module.runtime = old_runtime


def test_general_model_settings_and_play_only_camera_api(
    tmp_path: Path,
    caplog,
) -> None:
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
            database_url=_test_database_url(),
            
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
        caplog.set_level(logging.INFO, logger="uvicorn.error")
        with TestClient(main_module.app) as client:
            selected_logs = [
                record.getMessage()
                for record in caplog.records
                if "MODEL_SELECTED" in record.getMessage()
            ]
            assert any("role=fire_smoke" in message for message in selected_logs)
            assert any("role=vehicle_detector" in message for message in selected_logs)
            assert any("role=plate_detector" in message for message in selected_logs)
            assert any("role=plate_recognizer" in message for message in selected_logs)
            assert any("role=face_human_detector" in message for message in selected_logs)
            assert any("role=face_detector" in message for message in selected_logs)
            assert any("role=face_embedding" in message for message in selected_logs)

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

            forbidden = client.patch(
                "/api/v1/settings/general",
                json={
                    "operational": {
                        "fire_confidence": 0.12,
                        "smoke_confidence": 0.34,
                    }
                },
            )
            assert forbidden.status_code == 422

            class UploadedEngineExporter:
                def __init__(self, source: Path) -> None:
                    self.source = source

                def export(self, **kwargs):
                    target = self.source.with_suffix(f".{kwargs['format']}")
                    target.write_bytes(kwargs["format"].encode("ascii"))
                    return str(target)

            test_runtime.model_conversions._exporter_factory = UploadedEngineExporter
            export = client.post(
                "/api/v1/models/engine-exports",
                data={
                    "role": "fire_smoke",
                    "output_name": "swagger_fire",
                    "create_onnx_fallback": "false",
                    "select_when_ready": "true",
                },
                files={"file": ("fire.pt", b"pt-weights", "application/octet-stream")},
            )
            assert export.status_code == 202
            export_job = export.json()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                export_job = client.get(export_job["status_url"]).json()
                if export_job["status"] not in {"queued", "running"}:
                    break
                time.sleep(0.01)
            assert export_job["status"] == "completed"
            assert export_job["artifacts"] == ["fire_smoke/swagger_fire.engine"]

            general = client.get("/api/v1/settings/general").json()
            assert "fire_confidence" not in general["operational"]
            assert "smoke_confidence" not in general["operational"]
            assert "fire_confidence" not in general["application_config"]["values"]
            assert "smoke_confidence" not in general["application_config"]["values"]
            fire_selection = general["models"]["resolved_models"]["fire_smoke"]
            assert fire_selection["selected"] == "fire_smoke/swagger_fire.engine"
            assert fire_selection["selected_url"] == export_job["artifact_urls"][0]
            downloaded = client.get(fire_selection["selected_url"])
            assert downloaded.status_code == 200
            assert downloaded.content == b"engine"

            source_uri = test_runtime.registry.list()[0].source_uri
            play_only = client.patch(
                f"/api/v1/sources/{source_uri}",
                json={"tasks": []},
            )
            assert play_only.status_code == 200
            assert play_only.json()["tasks"] == []

            renamed = client.patch(
                f"/api/v1/sources/{source_uri}",
                json={"name": "Updated camera", "source_uri": "rtsp://example.test/ignored"},
            )
            assert renamed.status_code == 200
            assert renamed.json()["name"] == "Updated camera"
            assert renamed.json()["source_uri"] == source_uri

            moved = client.patch(
                f"/api/v1/sources/{source_uri}",
                json={"source_uri": "data/4.mp4"},
            )
            assert moved.status_code == 200
            assert moved.json()["source_uri"] == "data/4.mp4"
            assert client.get(f"/api/v1/sources/data/4.mp4").status_code == 200
            source_uri = "data/4.mp4"

            numeric_id = str(test_runtime.registry.list()[0].id)
            numeric_patch = client.patch(
                f"/api/v1/sources/{numeric_id}",
                json={"name": "Updated by id"},
            )
            assert numeric_patch.status_code == 200
            assert numeric_patch.json()["name"] == "Updated by id"
            assert numeric_patch.json()["id"] == int(numeric_id)

            summary = test_runtime.router.submit_round(
                frames=[np.zeros((64, 64, 3), dtype=np.uint8)],
                source_ids=[source_uri],
                round_sequence=999,
            )
            assert summary["task_submissions"] == 0
            latest = test_runtime.broadcast.wait_next(source_uri, 0, timeout=1.0)
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
            database_url=_test_database_url(),
            
            saved_media_path=tmp_path / "media",
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        with TestClient(main_module.app) as client:
            source_uri = test_runtime.registry.list()[0].source_uri
            inherited = client.get(
                f"/api/v1/plate-settings/cameras/{source_uri}"
            )
            assert inherited.status_code == 200
            assert inherited.json()["overrides"] == {}

            overridden = client.patch(
                f"/api/v1/plate-settings/cameras/{source_uri}",
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
                f"/api/v1/plate-settings/cameras/{source_uri}"
            ).json()["effective"]
            assert effective["plate_confidence"] == 0.61
            assert effective["vehicle_confidence"] == 0.41
            assert effective["min_vehicle_width_pixels"] == 150

            cleared = client.patch(
                f"/api/v1/plate-settings/cameras/{source_uri}",
                json={"plate_confidence": None},
            )
            assert cleared.status_code == 200
            assert cleared.json()["effective"]["plate_confidence"] == 0.47
            assert "plate_confidence" in cleared.json()["inherited_fields"]
    finally:
        main_module.runtime = old_runtime


def test_get_all_sections_returns_all_groups(tmp_path: Path) -> None:
    """GET /api/v1/tests/all must return every section group."""
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            database_url=_test_database_url(),
            
            saved_media_path=tmp_path / "media",
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        with TestClient(main_module.app) as client:
            resp = client.get("/api/v1/tests/all")
            assert resp.status_code == 200
            body = resp.json()

            # Top-level groups (each must have a "tests" list)
            for group in ("models", "stores", "services", "data_stores", "settings",
                          "infrastructure", "model_management"):
                assert group in body, f"Missing group: {group}"
                assert "tests" in body[group], f"{group} missing tests list"
                assert isinstance(body[group]["tests"], list), f"{group} tests is not a list"

            assert "_summary" in body, "Missing _summary"

            # Every test must have name + status
            for group in ("models", "stores", "services", "data_stores", "settings",
                          "infrastructure", "model_management"):
                for t in body[group]["tests"]:
                    assert "name" in t, f"Test missing name in {group}: {t}"
                    assert t.get("status") in ("PASS", "FAIL", "WARN", "SKIP"), \
                        f"Unexpected status {t.get('status')} in {group}: {t['name']}"

            # Summary must have overall status
            assert body["_summary"]["status"] in ("PASS", "FAIL", "WARN")
            assert body["_summary"]["total_tests"] > 0

            # Check OpenAPI schema includes new endpoint
            schema = client.get("/openapi.json").json()
            assert "/api/v1/tests/all" in schema["paths"]
            assert "get" in schema["paths"]["/api/v1/tests/all"]
            assert "post" in schema["paths"]["/api/v1/tests/all"]
    finally:
        main_module.runtime = old_runtime


def test_post_all_sections_smoke_runs_tests(tmp_path: Path) -> None:
    """POST /api/v1/tests/all must run active smoke tests for every section."""
    import app.main as main_module

    test_runtime = build_runtime(
        replace(
            settings,
            processor_mode="mock",
            database_url=_test_database_url(),
            
            saved_media_path=tmp_path / "media",
            video_ingestion_enabled=False,
        )
    )
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime
    try:
        with TestClient(main_module.app) as client:
            resp = client.post("/api/v1/tests/all")
            assert resp.status_code == 200
            body = resp.json()

            # Every section must be present
            expected_sections = [
                "personnel", "locations", "shifts", "holidays",
                "requests", "infrastructure",
            ]
            for section in expected_sections:
                assert section in body, f"Missing section: {section}"
                if section == "infrastructure":
                    assert isinstance(body[section], dict)
                    assert "status" in body[section], "infrastructure missing overall status"
                    assert body[section]["status"] in ("PASS", "SKIP", "FAIL", "ERROR"), \
                        f"infrastructure has unexpected status: {body[section]['status']}"
                else:
                    assert body[section].get("status") in (
                        "PASS", "SKIP", "FAIL", "ERROR"
                    ), f"{section} has unexpected status"

            # Most should PASS with mock mode
            passed = body["_summary"]["passed"]
            failed = body["_summary"]["failed"]
            assert passed >= 6, f"Expected at least 6 passed sections, got {passed}"
            assert failed == 0, f"Expected 0 failed sections, got {failed}: {body}"

            assert body["_summary"]["total_sections"] == len(expected_sections)
    finally:
        main_module.runtime = old_runtime
