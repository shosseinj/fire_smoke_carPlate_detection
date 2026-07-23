from __future__ import annotations

import inspect


def _methods(router: object, path: str) -> set[str]:
    return {
        method
        for route in router.routes
        if getattr(route, "path", None) == path
        for method in (getattr(route, "methods", set()) or set())
    }


def test_legacy_filter_parameters_are_exposed() -> None:
    from app.api.car_plates import list_car_plates
    from app.api.fire_logs import list_fire_logs

    car_plate_parameters = inspect.signature(list_car_plates).parameters
    fire_log_parameters = inspect.signature(list_fire_logs).parameters
    assert {"usage_type", "vehicle_type", "owner_phone"} <= set(car_plate_parameters)
    assert {"hazard_type", "detected_from", "detected_to"} <= set(fire_log_parameters)
