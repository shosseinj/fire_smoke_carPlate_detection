from __future__ import annotations

import inspect


def _methods(router: object, path: str) -> set[str]:
    return {
        method
        for route in router.routes
        if getattr(route, "path", None) == path
        for method in (getattr(route, "methods", set()) or set())
    }


def test_verified_legacy_aliases_are_registered() -> None:
    from app.api.cameras import router as cameras
    from app.api.legacy_model_exports import router as exports
    from app.api.legacy_websocket import root_router as websocket
    from app.api.locations import buildings_router, sections_router, rooms_router

    assert "GET" in _methods(cameras, "/api/v1/cameras/")
    assert "POST" in _methods(cameras, "/api/v1/cameras/")
    assert "PATCH" in _methods(buildings_router, "/buildings/{building_id}")
    assert "PATCH" in _methods(sections_router, "/sections/{section_id}")
    assert "PATCH" in _methods(
        sections_router, "/sections/{section_id}/assign-camera/{camera_id}"
    )
    assert "DELETE" in _methods(rooms_router, "/rooms/{room_id}/revoke/{personnel_id}")
    assert "DELETE" in _methods(exports, "/api/v1/model-exports/{job_id}")
    assert {route.path for route in websocket.routes} >= {
        "/ws/live/{camera_id}",
        "/ws/connections/status",
        "/ws/video-page",
    }


def test_legacy_filter_parameters_are_exposed() -> None:
    from app.api.car_plates import list_car_plates
    from app.api.fire_logs import list_fire_logs

    car_plate_parameters = inspect.signature(list_car_plates).parameters
    fire_log_parameters = inspect.signature(list_fire_logs).parameters
    assert {"usage_type", "vehicle_type", "owner_phone"} <= set(car_plate_parameters)
    assert {"hazard_type", "detected_from", "detected_to"} <= set(fire_log_parameters)
