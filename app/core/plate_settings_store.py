from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text

import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.source_registry import SourceChange


def _utc_now() -> str:
    return utc_now_text()


@dataclass(frozen=True, slots=True)
class PlateDetectionPolicy:
    vehicle_confidence: float = 0.35
    plate_confidence: float = 0.30
    ocr_confidence: float = 0.50
    min_vehicle_width_pixels: int = 120
    min_vehicle_height_pixels: int = 80
    min_vehicle_area_ratio: float = 0.025
    vehicle_crop_padding_ratio: float = 0.05

    def validated(self) -> "PlateDetectionPolicy":
        for name in ("vehicle_confidence", "plate_confidence", "ocr_confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 1 <= int(self.min_vehicle_width_pixels) <= 4096:
            raise ValueError("حداقل عرض خودرو باید بین 1 و 4096 باشد")
        if not 1 <= int(self.min_vehicle_height_pixels) <= 4096:
            raise ValueError("حداقل ارتفاع خودرو باید بین 1 و 4096 باشد")
        if not 0.0 <= float(self.min_vehicle_area_ratio) <= 1.0:
            raise ValueError("حداقل نسبت مساحت خودرو باید بین 0 و 1 باشد")
        if not 0.0 <= float(self.vehicle_crop_padding_ratio) <= 0.5:
            raise ValueError("نسبت حاشیه برش خودرو باید بین 0 و 0.5 باشد")
        return PlateDetectionPolicy(
            vehicle_confidence=float(self.vehicle_confidence),
            plate_confidence=float(self.plate_confidence),
            ocr_confidence=float(self.ocr_confidence),
            min_vehicle_width_pixels=int(self.min_vehicle_width_pixels),
            min_vehicle_height_pixels=int(self.min_vehicle_height_pixels),
            min_vehicle_area_ratio=float(self.min_vehicle_area_ratio),
            vehicle_crop_padding_ratio=float(self.vehicle_crop_padding_ratio),
        )


class PlateSettingsStore:
    """In-memory effective settings backed by general and per-camera PostgreSQL rows."""

    FIELDS = tuple(asdict(PlateDetectionPolicy()).keys())

    def __init__(
        self,
        database: Database | str,
        *,
        default_policy: PlateDetectionPolicy,
    ) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()
        self._revision = 0
        self._initialize(default_policy.validated())
        self._general, self._general_updated_at = self._load_general()
        self._camera_overrides = self._load_camera_overrides()

    def _connect(self) -> Connection:
        return self.database.connection()

    def _initialize(self, default: PlateDetectionPolicy) -> None:
        values = asdict(default)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO plate_general_settings (
                    id, vehicle_confidence, plate_confidence, ocr_confidence,
                    min_vehicle_width_pixels, min_vehicle_height_pixels,
                    min_vehicle_area_ratio, vehicle_crop_padding_ratio,
                    updated_at_utc
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (*values.values(), utc_now_text()),
            )

    def _load_general(self) -> tuple[PlateDetectionPolicy, str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM plate_general_settings WHERE id = 1"
            ).fetchone()
        assert row is not None
        return (
            PlateDetectionPolicy(**{field: row[field] for field in self.FIELDS}).validated(),
            str(row["updated_at_utc"]),
        )

    def _load_camera_overrides(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM plate_camera_settings").fetchall()
        return {
            str(row["camera_id"]): {
                field: row[field] for field in self.FIELDS if row[field] is not None
            }
            for row in rows
        }

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def general(self) -> dict[str, Any]:
        with self._lock:
            return {
                **asdict(self._general),
                "revision": self._revision,
                "updated_at_utc": self._general_updated_at,
            }

    def update_general(self, policy: PlateDetectionPolicy) -> dict[str, Any]:
        policy = policy.validated()
        values = asdict(policy)
        updated_at = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE plate_general_settings
                SET vehicle_confidence = ?, plate_confidence = ?,
                    ocr_confidence = ?, min_vehicle_width_pixels = ?,
                    min_vehicle_height_pixels = ?, min_vehicle_area_ratio = ?,
                    vehicle_crop_padding_ratio = ?, updated_at_utc = ?
                WHERE id = 1
                """,
                (*values.values(), updated_at),
            )
            connection.commit()
            self._general = policy
            self._general_updated_at = updated_at
            self._revision += 1
        return self.general()

    def resolve(self, camera_id: str) -> PlateDetectionPolicy:
        with self._lock:
            overrides = dict(self._camera_overrides.get(camera_id, {}))
            return replace(self._general, **overrides).validated()

    def camera(self, camera_id: str) -> dict[str, Any]:
        with self._lock:
            overrides = dict(self._camera_overrides.get(camera_id, {}))
            effective = asdict(replace(self._general, **overrides).validated())
            return {
                "camera_id": camera_id,
                "overrides": overrides,
                "effective": effective,
                "inherited_fields": [
                    field for field in self.FIELDS if field not in overrides
                ],
                "revision": self._revision,
            }

    def update_camera(self, camera_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        unknown = set(changes) - set(self.FIELDS)
        if unknown:
            raise ValueError(f"تنظیمات پلاک نامعتبر: {sorted(unknown)}")
        with self._lock:
            overrides = dict(self._camera_overrides.get(camera_id, {}))
            for field, value in changes.items():
                if value is None:
                    overrides.pop(field, None)
                else:
                    overrides[field] = value
            replace(self._general, **overrides).validated()
            updated_at = _utc_now()
            parameters = [camera_id]
            parameters.extend(overrides.get(field) for field in self.FIELDS)
            parameters.append(updated_at)
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO plate_camera_settings (
                        camera_id, vehicle_confidence, plate_confidence,
                        ocr_confidence, min_vehicle_width_pixels,
                        min_vehicle_height_pixels, min_vehicle_area_ratio,
                        vehicle_crop_padding_ratio, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(camera_id) DO UPDATE SET
                        vehicle_confidence = excluded.vehicle_confidence,
                        plate_confidence = excluded.plate_confidence,
                        ocr_confidence = excluded.ocr_confidence,
                        min_vehicle_width_pixels = excluded.min_vehicle_width_pixels,
                        min_vehicle_height_pixels = excluded.min_vehicle_height_pixels,
                        min_vehicle_area_ratio = excluded.min_vehicle_area_ratio,
                        vehicle_crop_padding_ratio = excluded.vehicle_crop_padding_ratio,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    parameters,
                )
                connection.commit()
            self._camera_overrides[camera_id] = overrides
            self._revision += 1
        return self.camera(camera_id)

    def delete_camera(self, camera_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM plate_camera_settings WHERE camera_id = ?",
                (camera_id,),
            )
            connection.commit()
            self._camera_overrides.pop(camera_id, None)
            if cursor.rowcount > 0:
                self._revision += 1
                return True
            return False

    def on_source_change(self, change: SourceChange) -> None:
        if change.action == "deleted":
            self.delete_camera(change.source_uri)
