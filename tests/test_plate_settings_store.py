from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.core.plate_settings_store import PlateDetectionPolicy, PlateSettingsStore


def test_camera_overrides_inherit_general_settings_and_persist(tmp_path: Path) -> None:
    database_path = tmp_path / "cameras.sqlite3"
    initial = PlateDetectionPolicy(
        vehicle_confidence=0.35,
        plate_confidence=0.30,
        ocr_confidence=0.50,
        min_vehicle_width_pixels=120,
        min_vehicle_height_pixels=80,
        min_vehicle_area_ratio=0.025,
        vehicle_crop_padding_ratio=0.05,
    )
    store = PlateSettingsStore(database_path, default_policy=initial)

    camera = store.update_camera(
        "camera-01",
        {"plate_confidence": 0.60, "min_vehicle_width_pixels": 180},
    )
    assert camera["effective"]["plate_confidence"] == 0.60
    assert camera["effective"]["vehicle_confidence"] == 0.35
    assert "vehicle_confidence" in camera["inherited_fields"]
    assert "plate_confidence" not in camera["inherited_fields"]

    store.update_general(
        replace(initial, vehicle_confidence=0.42, plate_confidence=0.44)
    )
    effective = store.resolve("camera-01")
    assert effective.vehicle_confidence == 0.42
    assert effective.plate_confidence == 0.60

    store.update_camera("camera-01", {"plate_confidence": None})
    assert store.resolve("camera-01").plate_confidence == 0.44

    reopened = PlateSettingsStore(
        database_path,
        default_policy=PlateDetectionPolicy(plate_confidence=0.01),
    )
    persisted = reopened.camera("camera-01")
    assert persisted["effective"]["vehicle_confidence"] == 0.42
    assert persisted["effective"]["plate_confidence"] == 0.44
    assert persisted["effective"]["min_vehicle_width_pixels"] == 180


def test_resetting_camera_settings_restores_all_general_values(tmp_path: Path) -> None:
    store = PlateSettingsStore(
        tmp_path / "cameras.sqlite3",
        default_policy=PlateDetectionPolicy(),
    )
    store.update_camera("camera-02", {"ocr_confidence": 0.80})

    assert store.delete_camera("camera-02") is True
    camera = store.camera("camera-02")
    assert camera["overrides"] == {}
    assert set(camera["inherited_fields"]) == set(store.FIELDS)
    assert camera["effective"]["ocr_confidence"] == 0.50
