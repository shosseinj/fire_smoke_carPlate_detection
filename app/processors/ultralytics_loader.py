from __future__ import annotations

import importlib
import threading
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any


_model_load_lock = threading.RLock()
_yolo_class: Any | None = None
_hezar_model_class: Any | None = None


@contextmanager
def serialized_model_load() -> Iterator[None]:
    """Prevent parallel native-library imports and model initialization."""
    with _model_load_lock:
        yield


def load_yolo_class() -> Any:
    """Resolve Ultralytics' lazy YOLO export once for the whole process.

    Ultralytics resolves ``YOLO`` through module-level lazy imports. Loading it
    concurrently from multiple task-worker threads can initialize native
    dependencies in parallel and abort the interpreter.
    """
    global _yolo_class
    if _yolo_class is not None:
        return _yolo_class

    with _model_load_lock:
        if _yolo_class is None:
            ultralytics = importlib.import_module("ultralytics")
            _yolo_class = getattr(ultralytics, "YOLO")
        return _yolo_class


def load_hezar_model_class() -> Any:
    """Resolve Hezar's model class under the shared native-load lock."""
    global _hezar_model_class
    if _hezar_model_class is not None:
        return _hezar_model_class

    with _model_load_lock:
        if _hezar_model_class is None:
            hezar_models = importlib.import_module("hezar.models")
            _hezar_model_class = getattr(hezar_models, "Model")
        return _hezar_model_class


def preload_model_dependencies() -> None:
    """Complete lazy AI-library imports before native worker threads start."""
    with _model_load_lock:
        load_yolo_class()
        load_hezar_model_class()
