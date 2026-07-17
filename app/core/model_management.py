from __future__ import annotations

import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MODEL_ROLE_DIRECTORIES = {
    "fire_smoke": "fire_smoke",
    "vehicle_detector": "vehicle_detector",
    "plate_detector": "plate_detector",
}
MODEL_FORMATS = ("engine", "onnx", "pt")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _variant_from_name(name: str) -> str | None:
    lowered = name.lower()
    for token, variant in (
        ("nano", "nano"),
        ("tiny", "tiny"),
        ("small", "small"),
        ("medium", "medium"),
        ("xlarge", "xlarge"),
        ("large", "large"),
    ):
        if token in lowered:
            return variant
    match = re.search(r"yolo(?:v)?\d+([nsmlx])(?:[._-]|$)", lowered.replace(" ", ""))
    if match:
        return {
            "n": "nano",
            "s": "small",
            "m": "medium",
            "l": "large",
            "x": "xlarge",
        }.get(match.group(1))
    return None


@dataclass(frozen=True, slots=True)
class ModelSelectionConfig:
    fire_smoke_model: str
    vehicle_detector_model: str
    plate_detector_model: str
    preferred_format: str = "engine"
    allow_onnx_fallback: bool = True
    allow_pt_fallback: bool = True
    export_imgsz: int = 640
    export_batch_size: int = 8
    export_workspace_gb: float = 4.0
    export_half: bool = True
    export_dynamic: bool = True
    export_timeout_seconds: int = 300

    def validated(self) -> "ModelSelectionConfig":
        if self.preferred_format not in MODEL_FORMATS:
            raise ValueError("preferred_format must be engine, onnx, or pt")
        if not 32 <= int(self.export_imgsz) <= 4096:
            raise ValueError("export_imgsz must be between 32 and 4096")
        if not 1 <= int(self.export_batch_size) <= 128:
            raise ValueError("export_batch_size must be between 1 and 128")
        if not 0.25 <= float(self.export_workspace_gb) <= 128.0:
            raise ValueError("export_workspace_gb must be between 0.25 and 128")
        if not 30 <= int(self.export_timeout_seconds) <= 7200:
            raise ValueError("export_timeout_seconds must be between 30 and 7200")
        return ModelSelectionConfig(
            fire_smoke_model=str(self.fire_smoke_model),
            vehicle_detector_model=str(self.vehicle_detector_model),
            plate_detector_model=str(self.plate_detector_model),
            preferred_format=str(self.preferred_format),
            allow_onnx_fallback=bool(self.allow_onnx_fallback),
            allow_pt_fallback=bool(self.allow_pt_fallback),
            export_imgsz=int(self.export_imgsz),
            export_batch_size=int(self.export_batch_size),
            export_workspace_gb=float(self.export_workspace_gb),
            export_half=bool(self.export_half),
            export_dynamic=bool(self.export_dynamic),
            export_timeout_seconds=int(self.export_timeout_seconds),
        )


class ModelManager:
    """Persistent model catalog, selection, and runtime fallback resolution."""

    ROLE_FIELDS = {
        "fire_smoke": "fire_smoke_model",
        "vehicle_detector": "vehicle_detector_model",
        "plate_detector": "plate_detector_model",
    }

    def __init__(
        self,
        database_path: Path,
        model_root: Path,
        *,
        default_config: ModelSelectionConfig,
    ) -> None:
        self.database_path = database_path.resolve()
        self.model_root = model_root.resolve()
        self.model_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._revision = 0
        self._initialize(default_config.validated())
        self._config, self._updated_at = self._load()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def relative_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.model_root).as_posix()
        except ValueError as exc:
            raise ValueError("Model path must stay inside MODEL_ROOT_PATH") from exc

    def resolve_path(self, value: str) -> Path:
        candidate = (self.model_root / value).resolve()
        try:
            candidate.relative_to(self.model_root)
        except ValueError as exc:
            raise ValueError("Model path must stay inside MODEL_ROOT_PATH") from exc
        return candidate

    def _initialize(self, default: ModelSelectionConfig) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS model_general_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    fire_smoke_model TEXT NOT NULL,
                    vehicle_detector_model TEXT NOT NULL,
                    plate_detector_model TEXT NOT NULL,
                    preferred_format TEXT NOT NULL,
                    allow_onnx_fallback INTEGER NOT NULL,
                    allow_pt_fallback INTEGER NOT NULL,
                    export_imgsz INTEGER NOT NULL,
                    export_batch_size INTEGER NOT NULL,
                    export_workspace_gb REAL NOT NULL,
                    export_half INTEGER NOT NULL,
                    export_dynamic INTEGER NOT NULL,
                    export_timeout_seconds INTEGER NOT NULL DEFAULT 300,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(model_general_settings)"
                )
            }
            if "export_timeout_seconds" not in columns:
                connection.execute(
                    "ALTER TABLE model_general_settings "
                    "ADD COLUMN export_timeout_seconds INTEGER NOT NULL DEFAULT 300"
                )
            values = asdict(default)
            connection.execute(
                """
                INSERT OR IGNORE INTO model_general_settings (
                    id, fire_smoke_model, vehicle_detector_model,
                    plate_detector_model, preferred_format,
                    allow_onnx_fallback, allow_pt_fallback, export_imgsz,
                    export_batch_size, export_workspace_gb, export_half,
                    export_dynamic, export_timeout_seconds, updated_at_utc
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["fire_smoke_model"],
                    values["vehicle_detector_model"],
                    values["plate_detector_model"],
                    values["preferred_format"],
                    int(values["allow_onnx_fallback"]),
                    int(values["allow_pt_fallback"]),
                    values["export_imgsz"],
                    values["export_batch_size"],
                    values["export_workspace_gb"],
                    int(values["export_half"]),
                    int(values["export_dynamic"]),
                    values["export_timeout_seconds"],
                    _utc_now(),
                ),
            )
            connection.commit()

    def _load(self) -> tuple[ModelSelectionConfig, str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_general_settings WHERE id = 1"
            ).fetchone()
        assert row is not None
        return (
            ModelSelectionConfig(
                fire_smoke_model=row["fire_smoke_model"],
                vehicle_detector_model=row["vehicle_detector_model"],
                plate_detector_model=row["plate_detector_model"],
                preferred_format=row["preferred_format"],
                allow_onnx_fallback=bool(row["allow_onnx_fallback"]),
                allow_pt_fallback=bool(row["allow_pt_fallback"]),
                export_imgsz=row["export_imgsz"],
                export_batch_size=row["export_batch_size"],
                export_workspace_gb=row["export_workspace_gb"],
                export_half=bool(row["export_half"]),
                export_dynamic=bool(row["export_dynamic"]),
                export_timeout_seconds=row["export_timeout_seconds"],
            ).validated(),
            str(row["updated_at_utc"]),
        )

    def _validate_role_path(self, role: str, value: str) -> Path:
        path = self.resolve_path(value)
        if path.suffix.lower().lstrip(".") not in MODEL_FORMATS:
            raise ValueError(f"{role} model must be a .pt, .onnx, or .engine file")
        expected = (self.model_root / MODEL_ROLE_DIRECTORIES[role]).resolve()
        try:
            path.relative_to(expected)
        except ValueError as exc:
            raise ValueError(f"{role} model must be inside {expected}") from exc
        if not path.is_file():
            raise ValueError(f"Selected model does not exist: {value}")
        return path

    def _validate_selected_models(self, config: ModelSelectionConfig) -> None:
        for role, field in self.ROLE_FIELDS.items():
            self._validate_role_path(role, getattr(config, field))

    def _format_order(self, config: ModelSelectionConfig) -> list[str]:
        if config.preferred_format == "engine":
            formats = ["engine"]
            if config.allow_onnx_fallback:
                formats.append("onnx")
            if config.allow_pt_fallback:
                formats.append("pt")
            return formats
        if config.preferred_format == "onnx":
            formats = ["onnx"]
            if config.allow_pt_fallback:
                formats.append("pt")
            return formats
        return ["pt"]

    def candidates(self, role: str) -> list[Path]:
        if role not in self.ROLE_FIELDS:
            raise KeyError(role)
        with self._lock:
            config = self._config
            selected = self.resolve_path(getattr(config, self.ROLE_FIELDS[role]))
            candidates = [
                selected.with_suffix(f".{model_format}")
                for model_format in self._format_order(config)
            ]
            if selected not in candidates:
                candidates.append(selected)
            return [path for path in candidates if path.is_file()]

    def provider(self, role: str) -> Callable[[], tuple[int, list[Path]]]:
        def snapshot() -> tuple[int, list[Path]]:
            with self._lock:
                return self._revision, self.candidates(role)

        return snapshot

    def catalog(
        self,
        *,
        role: str | None = None,
        model_format: str | None = None,
    ) -> list[dict[str, Any]]:
        if role is not None and role not in MODEL_ROLE_DIRECTORIES:
            raise ValueError(f"Unknown model role: {role}")
        if model_format is not None and model_format not in MODEL_FORMATS:
            raise ValueError(f"Unknown model format: {model_format}")
        items: list[dict[str, Any]] = []
        roles = [role] if role else list(MODEL_ROLE_DIRECTORIES)
        selected = asdict(self._config)
        for item_role in roles:
            directory = self.model_root / MODEL_ROLE_DIRECTORIES[item_role]
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                suffix = path.suffix.lower().lstrip(".")
                if not path.is_file() or suffix not in MODEL_FORMATS:
                    continue
                if model_format is not None and suffix != model_format:
                    continue
                relative = self.relative_path(path)
                items.append(
                    {
                        "role": item_role,
                        "path": relative,
                        "filename": path.name,
                        "format": suffix,
                        "variant": _variant_from_name(path.stem),
                        "size_bytes": path.stat().st_size,
                        "selected": selected[self.ROLE_FIELDS[item_role]] == relative,
                        "convertible": suffix == "pt",
                    }
                )
        return items

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            value = asdict(self._config)
            resolved = {
                role: {
                    "selected": value[field],
                    "candidates": [self.relative_path(path) for path in self.candidates(role)],
                    "active_choice": (
                        self.relative_path(self.candidates(role)[0])
                        if self.candidates(role)
                        else None
                    ),
                }
                for role, field in self.ROLE_FIELDS.items()
            }
            return {
                **value,
                "revision": self._revision,
                "updated_at_utc": self._updated_at,
                "resolved_models": resolved,
            }

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = set(asdict(self._config))
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown model setting(s): {sorted(unknown)}")
        with self._lock:
            candidate = replace(self._config, **changes).validated()
            for role, field in self.ROLE_FIELDS.items():
                if field in changes:
                    self._validate_role_path(role, getattr(candidate, field))
            updated_at = _utc_now()
            values = asdict(candidate)
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE model_general_settings SET
                        fire_smoke_model = ?, vehicle_detector_model = ?,
                        plate_detector_model = ?, preferred_format = ?,
                        allow_onnx_fallback = ?, allow_pt_fallback = ?,
                        export_imgsz = ?, export_batch_size = ?,
                        export_workspace_gb = ?, export_half = ?,
                        export_dynamic = ?, export_timeout_seconds = ?,
                        updated_at_utc = ?
                    WHERE id = 1
                    """,
                    (
                        values["fire_smoke_model"],
                        values["vehicle_detector_model"],
                        values["plate_detector_model"],
                        values["preferred_format"],
                        int(values["allow_onnx_fallback"]),
                        int(values["allow_pt_fallback"]),
                        values["export_imgsz"],
                        values["export_batch_size"],
                        values["export_workspace_gb"],
                        int(values["export_half"]),
                        int(values["export_dynamic"]),
                        values["export_timeout_seconds"],
                        updated_at,
                    ),
                )
                connection.commit()
            self._config = candidate
            self._updated_at = updated_at
            self._revision += 1
        return self.snapshot()


@dataclass(slots=True)
class ConversionJob:
    job_id: str
    source_model: str
    output_directory: str
    status: str
    created_at_utc: str
    updated_at_utc: str
    options: dict[str, Any]
    artifacts: list[str]
    errors: list[dict[str, str]]


class ModelConversionManager:
    """Serial background exporter so TensorRT work never blocks request threads."""

    def __init__(
        self,
        models: ModelManager,
        *,
        exporter_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.models = models
        self._exporter_factory = exporter_factory
        self._lock = threading.RLock()
        self._jobs: dict[str, ConversionJob] = {}
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=16)
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="model-conversion-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        *,
        source_model: str,
        output_directory: str | None = None,
        imgsz: int,
        batch: int,
        workspace: float,
        half: bool,
        dynamic: bool,
        device: str,
        create_onnx_fallback: bool,
        overwrite: bool,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        source = self.models.resolve_path(source_model)
        if not source.is_file() or source.suffix.lower() != ".pt":
            raise ValueError("source_model must be an existing .pt file in the model catalog")
        role = source.relative_to(self.models.model_root).parts[0]
        if role not in MODEL_ROLE_DIRECTORIES.values():
            raise ValueError("Only detector models can be exported")
        output = (
            self.models.resolve_path(output_directory)
            if output_directory
            else source.parent
        )
        output.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex
        now = _utc_now()
        job = ConversionJob(
            job_id=job_id,
            source_model=self.models.relative_path(source),
            output_directory=self.models.relative_path(output),
            status="queued",
            created_at_utc=now,
            updated_at_utc=now,
            options={
                "imgsz": imgsz,
                "batch": batch,
                "workspace": workspace,
                "half": half,
                "dynamic": dynamic,
                "device": device,
                "create_onnx_fallback": create_onnx_fallback,
                "overwrite": overwrite,
                "timeout_seconds": timeout_seconds,
            },
            artifacts=[],
            errors=[],
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("Model conversion manager is closed")
            self._jobs[job_id] = job
            try:
                self._queue.put_nowait(job_id)
            except queue.Full as exc:
                self._jobs.pop(job_id, None)
                raise RuntimeError("Model conversion queue is full") from exc
        return self.get(job_id)

    def _set_job(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for field, value in changes.items():
                setattr(job, field, value)
            job.updated_at_utc = _utc_now()

    def _export_format(self, job: ConversionJob, exporter: Any, model_format: str) -> str:
        source = self.models.resolve_path(job.source_model)
        output = self.models.resolve_path(job.output_directory)
        target = output / f"{source.stem}.{model_format}"
        if target.is_file() and not job.options["overwrite"]:
            return self.models.relative_path(target)
        kwargs: dict[str, Any] = {
            "format": model_format,
            "imgsz": job.options["imgsz"],
            "batch": job.options["batch"],
            "half": job.options["half"],
            "dynamic": job.options["dynamic"],
            "device": job.options["device"],
        }
        if model_format == "engine":
            kwargs["workspace"] = job.options["workspace"]
        exported = Path(exporter.export(**kwargs)).resolve()
        if exported != target.resolve():
            if target.exists():
                target.unlink()
            shutil.move(str(exported), str(target))
        if not target.is_file():
            raise RuntimeError(f"Exporter did not create {target}")
        return self.models.relative_path(target)

    def _export_format_subprocess(
        self,
        job: ConversionJob,
        model_format: str,
    ) -> str:
        source = self.models.resolve_path(job.source_model)
        output = self.models.resolve_path(job.output_directory)
        target = output / f"{source.stem}.{model_format}"
        if target.is_file() and not job.options["overwrite"]:
            return self.models.relative_path(target)
        command = [
            sys.executable,
            "-m",
            "app.model_export_worker",
            "--source",
            str(source),
            "--target",
            str(target),
            "--format",
            model_format,
            "--imgsz",
            str(job.options["imgsz"]),
            "--batch",
            str(job.options["batch"]),
            "--workspace",
            str(job.options["workspace"]),
            "--device",
            str(job.options["device"]),
        ]
        if job.options["half"]:
            command.append("--half")
        if job.options["dynamic"]:
            command.append("--dynamic")
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=job.options["timeout_seconds"],
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"{model_format} export exceeded {job.options['timeout_seconds']} seconds"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "export failed")[-2000:]
            raise RuntimeError(detail.strip())
        if not target.is_file():
            raise RuntimeError(f"Exporter did not create {target}")
        return self.models.relative_path(target)

    def _process(self, job_id: str) -> None:
        self._set_job(job_id, status="running")
        with self._lock:
            job = self._jobs[job_id]
        source = self.models.resolve_path(job.source_model)
        output = self.models.resolve_path(job.output_directory)
        family_pt = output / source.name
        if family_pt.resolve() != source.resolve() and (
            job.options["overwrite"] or not family_pt.exists()
        ):
            shutil.copy2(source, family_pt)
        exporter = None
        if self._exporter_factory is not None:
            try:
                exporter = self._exporter_factory(source)
            except Exception as exc:
                self._set_job(
                    job_id,
                    status="failed",
                    errors=[{"format": "load", "error": f"{type(exc).__name__}: {exc}"}],
                )
                return

        artifacts: list[str] = []
        errors: list[dict[str, str]] = []
        try:
            artifacts.append(
                self._export_format(job, exporter, "engine")
                if exporter is not None
                else self._export_format_subprocess(job, "engine")
            )
        except Exception as exc:
            errors.append({"format": "engine", "error": f"{type(exc).__name__}: {exc}"})
        if job.options["create_onnx_fallback"]:
            try:
                artifacts.append(
                    self._export_format(job, exporter, "onnx")
                    if exporter is not None
                    else self._export_format_subprocess(job, "onnx")
                )
            except Exception as exc:
                errors.append({"format": "onnx", "error": f"{type(exc).__name__}: {exc}"})
        if artifacts:
            status = "completed" if not errors else "completed_with_fallback"
        else:
            status = "failed"
        self._set_job(job_id, status=status, artifacts=artifacts, errors=errors)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._process(job_id)
            finally:
                self._queue.task_done()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return asdict(job)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(job) for job in reversed(list(self._jobs.values()))]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5.0)
