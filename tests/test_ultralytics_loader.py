from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from app.processors import ultralytics_loader


def test_load_yolo_class_serializes_and_caches_lazy_import(monkeypatch) -> None:
    expected = object()
    import_calls = 0
    active_imports = 0
    maximum_active_imports = 0
    counters_lock = threading.Lock()
    barrier = threading.Barrier(3)

    def fake_import_module(name: str):
        nonlocal import_calls, active_imports, maximum_active_imports
        assert name == "ultralytics"
        with counters_lock:
            import_calls += 1
            active_imports += 1
            maximum_active_imports = max(maximum_active_imports, active_imports)
        time.sleep(0.05)
        with counters_lock:
            active_imports -= 1
        return SimpleNamespace(YOLO=expected)

    monkeypatch.setattr(ultralytics_loader, "_yolo_class", None)
    monkeypatch.setattr(ultralytics_loader.importlib, "import_module", fake_import_module)

    resolved: list[object] = []

    def load() -> None:
        barrier.wait()
        resolved.append(ultralytics_loader.load_yolo_class())

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=1.0)

    assert resolved == [expected, expected]
    assert import_calls == 1
    assert maximum_active_imports == 1


def test_serialized_model_load_blocks_parallel_initialization() -> None:
    barrier = threading.Barrier(3)
    active_loads = 0
    maximum_active_loads = 0
    counters_lock = threading.Lock()

    def load() -> None:
        nonlocal active_loads, maximum_active_loads
        barrier.wait()
        with ultralytics_loader.serialized_model_load():
            with counters_lock:
                active_loads += 1
                maximum_active_loads = max(maximum_active_loads, active_loads)
            time.sleep(0.05)
            with counters_lock:
                active_loads -= 1

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=1.0)

    assert maximum_active_loads == 1


def test_preload_model_dependencies_resolves_both_lazy_exports(monkeypatch) -> None:
    yolo_class = object()
    hezar_model_class = object()
    imported: list[str] = []

    def fake_import_module(name: str):
        imported.append(name)
        if name == "ultralytics":
            return SimpleNamespace(YOLO=yolo_class)
        if name == "hezar.models":
            return SimpleNamespace(Model=hezar_model_class)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(ultralytics_loader, "_yolo_class", None)
    monkeypatch.setattr(ultralytics_loader, "_hezar_model_class", None)
    monkeypatch.setattr(ultralytics_loader.importlib, "import_module", fake_import_module)

    ultralytics_loader.preload_model_dependencies()

    assert imported == ["ultralytics", "hezar.models"]
    assert ultralytics_loader.load_yolo_class() is yolo_class
    assert ultralytics_loader.load_hezar_model_class() is hezar_model_class

import pytest

pytestmark = pytest.mark.streaming
