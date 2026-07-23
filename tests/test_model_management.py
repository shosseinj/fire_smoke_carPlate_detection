from __future__ import annotations

import time
import subprocess
from types import SimpleNamespace
from pathlib import Path

from app.core.model_management import (
    ModelConversionManager,
    ModelManager,
    ModelSelectionConfig,
)
from app.database import Database


def create_models(root: Path) -> ModelSelectionConfig:
    files = {
        "fire_smoke/fire_nano.pt": b"pt",
        "fire_smoke/fire_nano.onnx": b"onnx",
        "fire_smoke/fire_nano.engine": b"engine",
        "fire_smoke/fire_small.pt": b"pt-small",
        "vehicle_detector/yolo11n.pt": b"vehicle",
        "plate_detector/plate_medium.pt": b"plate",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return ModelSelectionConfig(
        fire_smoke_model="fire_smoke/fire_nano.pt",
        vehicle_detector_model="vehicle_detector/yolo11n.pt",
        plate_detector_model="plate_detector/plate_medium.pt",
    )


def test_catalog_selection_and_engine_onnx_pt_fallback(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    root = tmp_path / "weights"
    config = create_models(root)
    store = ModelManager(postgres_database, root, default_config=config)

    catalog = store.catalog(role="fire_smoke")
    assert {item["format"] for item in catalog} == {"pt", "onnx", "engine"}
    assert {item["variant"] for item in catalog} >= {"nano", "small"}
    assert [path.suffix for path in store.candidates("fire_smoke")] == [
        ".engine",
        ".onnx",
        ".pt",
    ]

    (root / "fire_smoke/fire_nano.engine").unlink()
    assert [path.suffix for path in store.candidates("fire_smoke")] == [
        ".onnx",
        ".pt",
    ]

    explicitly_onnx = store.update(
        {
            "fire_smoke_model": "fire_smoke/fire_nano.onnx",
            "preferred_format": "engine",
        }
    )
    assert [path.suffix for path in store.candidates("fire_smoke")] == [
        ".onnx",
        ".pt",
    ]
    assert explicitly_onnx["resolved_models"]["fire_smoke"]["selected_url"] == (
        "/api/v1/models/artifacts/content?path=fire_smoke%2Ffire_nano.onnx"
    )

    updated = store.update(
        {
            "fire_smoke_model": "fire_smoke/fire_small.pt",
            "preferred_format": "pt",
        }
    )
    assert updated["resolved_models"]["fire_smoke"]["active_choice"] == (
        "fire_smoke/fire_small.pt"
    )

    reopened = ModelManager(
        postgres_database,
        root,
        default_config=config,
    )
    assert reopened.snapshot()["fire_smoke_model"] == "fire_smoke/fire_small.pt"


class FailingEngineExporter:
    def __init__(self, source: Path) -> None:
        self.source = source

    def export(self, **kwargs):
        model_format = kwargs["format"]
        if model_format == "engine":
            raise RuntimeError("TensorRT is unavailable")
        target = self.source.with_suffix(f".{model_format}")
        target.write_bytes(b"exported")
        return str(target)


def test_conversion_job_falls_back_to_onnx_without_blocking_request(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    root = tmp_path / "weights"
    config = create_models(root)
    (root / "fire_smoke/fire_nano.onnx").unlink()
    store = ModelManager(postgres_database, root, default_config=config)
    manager = ModelConversionManager(
        store,
        exporter_factory=FailingEngineExporter,
    )
    try:
        queued = manager.submit(
            source_model="fire_smoke/fire_nano.pt",
            output_directory="fire_smoke/exports",
            imgsz=640,
            batch=8,
            workspace=4.0,
            half=True,
            dynamic=False,
            device="0",
            create_onnx_fallback=True,
            overwrite=False,
            timeout_seconds=300,
            select_when_ready=True,
        )
        assert queued["status"] == "queued"
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            job = manager.get(queued["job_id"])
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed_with_fallback"
        assert job["artifacts"] == ["fire_smoke/exports/fire_nano.onnx"]
        assert job["artifact_urls"] == [
            "/api/v1/models/artifacts/content?path="
            "fire_smoke%2Fexports%2Ffire_nano.onnx"
        ]
        assert job["errors"][0]["format"] == "engine"
        assert (root / job["artifacts"][0]).read_bytes() == b"exported"
        assert (root / "fire_smoke/exports/fire_nano.pt").is_file()
        assert store.snapshot()["fire_smoke_model"] == (
            "fire_smoke/exports/fire_nano.onnx"
        )
    finally:
        manager.close()

    reopened = ModelConversionManager(store, exporter_factory=FailingEngineExporter)
    try:
        assert reopened.get(queued["job_id"])["status"] == "completed_with_fallback"
    finally:
        reopened.close()


def test_stage_uploaded_pt_rejects_paths_and_preserves_file(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    from io import BytesIO

    root = tmp_path / "weights"
    config = create_models(root)
    store = ModelManager(postgres_database, root, default_config=config)
    manager = ModelConversionManager(store, exporter_factory=FailingEngineExporter)
    try:
        staged = manager.stage_uploaded_pt(
            BytesIO(b"uploaded-weights"),
            "model.pt",
            role="vehicle_detector",
            output_directory="vehicle_detector/uploads",
            output_name="uploaded_nano",
        )
        assert staged == "vehicle_detector/uploads/uploaded_nano.pt"
        assert (root / staged).read_bytes() == b"uploaded-weights"

        try:
            manager.stage_uploaded_pt(
                BytesIO(b"bad"),
                "model.pt",
                role="vehicle_detector",
                output_directory="plate_detector",
            )
        except ValueError as exc:
            assert "inside vehicle_detector/" in str(exc)
        else:
            raise AssertionError("cross-role upload path should be rejected")
    finally:
        manager.close()


def test_subprocess_engine_timeout_continues_with_onnx(
    tmp_path: Path,
    monkeypatch,
    postgres_database: Database,
) -> None:
    root = tmp_path / "weights"
    config = create_models(root)
    (root / "fire_smoke/fire_nano.engine").unlink()
    (root / "fire_smoke/fire_nano.onnx").unlink()
    store = ModelManager(postgres_database, root, default_config=config)

    def fake_run(command, **kwargs):
        model_format = command[command.index("--format") + 1]
        if model_format == "engine":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        target = Path(command[command.index("--target") + 1])
        target.write_bytes(b"onnx")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("app.core.model_management.subprocess.run", fake_run)
    manager = ModelConversionManager(store)
    try:
        queued = manager.submit(
            source_model="fire_smoke/fire_nano.pt",
            output_directory="fire_smoke/isolated",
            imgsz=640,
            batch=8,
            workspace=4.0,
            half=True,
            dynamic=True,
            device="0",
            create_onnx_fallback=True,
            overwrite=False,
            timeout_seconds=30,
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            job = manager.get(queued["job_id"])
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed_with_fallback"
        assert job["errors"][0]["format"] == "engine"
        assert "TimeoutError" in job["errors"][0]["error"]
        assert job["artifacts"] == ["fire_smoke/isolated/fire_nano.onnx"]
    finally:
        manager.close()
