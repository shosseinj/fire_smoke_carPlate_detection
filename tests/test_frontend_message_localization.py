from __future__ import annotations

from pathlib import Path
import time

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
import numpy as np

from app.core.frontend_messages import (
    contains_persian,
    install_frontend_exception_handlers,
    LocalizedJSONRoute,
    localize_frontend_payload,
    public_error_message,
    validation_error_message,
)
from app.core.types import FramePacket, TaskName, TaskResult


def test_common_framework_errors_are_localized() -> None:
    assert public_error_message("Not Found", status_code=404) == "مسیر یا منبع درخواستی یافت نشد"
    assert public_error_message("Method Not Allowed", status_code=405) == "روش درخواست برای این مسیر مجاز نیست"
    assert contains_persian(public_error_message("database exploded", status_code=500))
    mixed_secret = public_error_message("خطای پردازش: password=secret")
    assert contains_persian(mixed_secret)
    assert "secret" not in mixed_secret


def test_validation_messages_are_persian_and_keep_machine_type_separate() -> None:
    assert validation_error_message("missing") == "این فیلد الزامی است"
    assert contains_persian(validation_error_message("int_parsing"))
    assert contains_persian(validation_error_message("literal_error"))


def test_main_app_localizes_not_found_and_validation_responses() -> None:
    class Payload(BaseModel):
        enabled: bool

    app = FastAPI()
    install_frontend_exception_handlers(app)

    @app.put("/state")
    def update_state(payload: Payload) -> Payload:
        return payload

    client = TestClient(app)
    missing = client.get("/route-that-does-not-exist")
    assert missing.status_code == 404
    assert contains_persian(missing.json()["detail"])

    invalid = client.put("/state", json={})
    assert invalid.status_code == 422
    error = invalid.json()["detail"][0]
    assert error["type"] == "missing"
    assert contains_persian(error["msg"])


def test_unhandled_server_error_is_safe_persian_json() -> None:
    app = FastAPI()
    install_frontend_exception_handlers(app)

    @app.get("/failure")
    def failure() -> None:
        raise RuntimeError("password=secret")

    response = TestClient(app, raise_server_exceptions=False).get("/failure")
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert contains_persian(response.json()["detail"])
    assert "secret" not in response.text


def test_localized_json_route_translates_message_fields_only() -> None:
    app = FastAPI()
    router = APIRouter(route_class=LocalizedJSONRoute)

    @router.get("/result")
    def result() -> dict:
        return {
            "status": "failed",
            "detail": "Record not found after creation",
            "failure_code": "record_not_found",
        }

    app.include_router(router)
    body = TestClient(app).get("/result").json()
    assert contains_persian(body["detail"])
    assert body["status"] == "failed"
    assert body["failure_code"] == "record_not_found"


def test_health_style_payload_localizes_guidance_and_redacts_errors() -> None:
    payload = localize_frontend_payload(
        {
            "status": "limited",
            "last_error": "password=secret",
            "guidance": ["A processor failed during the sample"],
            "causes": ["processor_failure"],
        }
    )
    assert contains_persian(payload["last_error"])
    assert "secret" not in payload["last_error"]
    assert contains_persian(payload["guidance"][0])
    assert payload["status"] == "limited"
    assert payload["causes"] == ["processor_failure"]


def test_dashboard_is_rtl_and_contains_no_known_english_ui_fallbacks() -> None:
    dashboard = (Path(__file__).parents[1] / "app" / "web" / "dashboard.html").read_text(encoding="utf-8")
    assert '<html lang="fa" dir="rtl">' in dashboard
    assert "دیوار عملیات هوش مصنوعی ویدیو" in dashboard
    for old_text in (
        "Video AI Operations Wall",
        "Recent Detections",
        "Waiting for source frames",
        "API unavailable",
        "Broadcast live",
        "Fullscreen source not found",
    ):
        assert old_text not in dashboard


def test_task_failure_exposes_persian_text_without_raw_exception() -> None:
    packet = FramePacket(
        source_id="camera-1",
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        round_sequence=1,
        frame_index=1,
        captured_monotonic=time.monotonic(),
        captured_at_utc="2026-07-29T00:00:00Z",
    )
    result = TaskResult.failed(
        task=TaskName.FIRE_SMOKE,
        packet=packet,
        processing_ms=1.0,
        error=RuntimeError("secret internal failure"),
    )
    assert result.error == "پردازش فریم با خطا مواجه شد"
    assert "secret" not in result.to_dict()["error"]

import pytest

pytestmark = pytest.mark.unit
