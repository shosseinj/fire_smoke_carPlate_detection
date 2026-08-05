from __future__ import annotations

from types import SimpleNamespace

from fastapi import HTTPException
import pytest

import app.api.personnel_requests as api


def test_bulk_route_is_registered_before_dynamic_request_id() -> None:
    paths = [route.path for route in api.router.routes]
    assert "/api/v1/personnel-requests/bulk" in paths
    assert paths.index("/api/v1/personnel-requests/bulk") < paths.index(
        "/api/v1/personnel-requests/{request_id}"
    )


def test_bulk_endpoint_uses_one_personnel_and_preserves_order(monkeypatch) -> None:
    personnel = SimpleNamespace(id=7, fname="Ali", lname="Ahmadi", shift_id=3)
    shift = SimpleNamespace(id=3)
    captured: dict = {}

    class Store:
        def create_many(self, *, personnel_id, requests):
            captured.update(personnel_id=personnel_id, requests=requests)
            return [SimpleNamespace(id=index) for index in (10, 11)]

    monkeypatch.setattr(api, "_get_personnel", lambda personnel_id: personnel)
    monkeypatch.setattr(api, "_get_shift", lambda value: shift)
    monkeypatch.setattr(
        api,
        "_prepare_bulk_request",
        lambda _personnel, item: {
            "request_type": item.request_type,
            "start_date": item.start_date,
            "end_date": item.end_date or item.start_date,
            "status": "approved",
        },
    )
    monkeypatch.setattr(api, "get_request_store", lambda: Store())
    monkeypatch.setattr(
        api,
        "legacy_request_response",
        lambda record, full_name: {"id": record.id, "full_name": full_name},
    )

    result = api.create_bulk_requests(
        api.BulkPersonnelRequestCreate(
            personnel_id=7,
            requests=[
                api.BulkPersonnelRequestItem(
                    request_type="earned_leave",
                    start_date="1405/06/01",
                ),
                api.BulkPersonnelRequestItem(
                    request_type="mission",
                    start_date="1405/06/02",
                ),
            ],
        ),
        _={},
    )

    assert captured["personnel_id"] == 7
    assert [item["request_type"] for item in captured["requests"]] == [
        "earned_leave",
        "mission",
    ]
    assert result == {
        "personnel_id": 7,
        "count": 2,
        "requests": [
            {"id": 10, "full_name": "Ali Ahmadi"},
            {"id": 11, "full_name": "Ali Ahmadi"},
        ],
    }


def test_bulk_endpoint_rejects_unknown_personnel(monkeypatch) -> None:
    monkeypatch.setattr(api, "_get_personnel", lambda _personnel_id: None)

    with pytest.raises(HTTPException) as exc_info:
        api.create_bulk_requests(
            api.BulkPersonnelRequestCreate(
                personnel_id=999,
                requests=[
                    api.BulkPersonnelRequestItem(start_date="1405/06/01")
                ],
            ),
            _={},
        )

    assert exc_info.value.status_code == 404


def test_create_request_remove_logs_parameter_defaults_to_false() -> None:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(api.router)
    parameters = app.openapi()["paths"]["/api/v1/personnel-requests/"]["post"][
        "parameters"
    ]
    parameter = next(
        item for item in parameters
        if item["name"] == "remove_logs_in_request_dates"
    )
    assert parameter["schema"]["default"] is False
    response_schema = app.openapi()["paths"][
        "/api/v1/personnel-requests/"
    ]["post"]["responses"]["201"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("/PersonnelRequestCreateResponse")
    create_schema = app.openapi()["components"]["schemas"][
        "PersonnelRequestCreateResponse"
    ]
    assert "removed_logs_count" in create_schema["properties"]
    request_schema = app.openapi()["paths"][
        "/api/v1/personnel-requests/"
    ]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema["$ref"].endswith("/PersonnelRequestCreate")
    request_model_schema = app.openapi()["components"]["schemas"][
        "PersonnelRequestCreate"
    ]
    assert set(request_model_schema["properties"]) == {
        "personnel_id",
        "request_type",
        "duration_type",
        "start_date",
        "end_date",
        "start_time",
        "end_time",
        "description",
    }


def test_personnel_request_read_routes_publish_shared_response_model() -> None:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(api.router)
    paths = app.openapi()["paths"]

    list_schema = paths["/api/v1/personnel-requests/"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    detail_schema = paths["/api/v1/personnel-requests/{request_id}"]["get"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]

    assert list_schema["items"]["$ref"].endswith("/PersonnelRequestResponse")
    assert detail_schema["$ref"].endswith("/PersonnelRequestResponse")


def test_create_daily_request_optionally_removes_personnel_logs(monkeypatch) -> None:
    personnel = SimpleNamespace(id=7, fname="Ali", lname="Ahmadi", shift_id=3)
    removed: list[tuple[int, object, object]] = []

    class Store:
        def create(self, **values):
            return SimpleNamespace(id=10, **values)

    monkeypatch.setattr(api, "_get_personnel", lambda _personnel_id: personnel)
    monkeypatch.setattr(api, "get_request_store", lambda: Store())
    monkeypatch.setattr(
        api,
        "_calculate_request_duration_for_assignments",
        lambda *_: {"duration_days": 2.0, "duration_minutes": None},
    )
    monkeypatch.setattr(
        api,
        "_remove_detection_logs_for_daily_request",
        lambda personnel_id, start, end: removed.append(
            (personnel_id, start, end)
        ) or 4,
    )
    monkeypatch.setattr(
        api,
        "legacy_request_response",
        lambda record, full_name: {"id": record.id, "full_name": full_name},
    )

    response = api.create_request(
        body=api.PersonnelRequestCreate(
            personnel_id=7,
            request_type="sick_leave",
            duration_type="daily",
            start_date="1405-01-18",
            end_date="1405-01-19",
        ),
        remove_logs_in_request_dates=True,
        _={},
    )

    assert len(removed) == 1
    assert removed[0][0] == 7
    assert response["removed_logs_count"] == 4


def test_create_hourly_request_never_removes_logs(monkeypatch) -> None:
    personnel = SimpleNamespace(id=7, fname="Ali", lname="Ahmadi", shift_id=3)

    class Store:
        def create(self, **values):
            return SimpleNamespace(id=10, **values)

    monkeypatch.setattr(api, "_get_personnel", lambda _personnel_id: personnel)
    monkeypatch.setattr(api, "get_request_store", lambda: Store())
    monkeypatch.setattr(
        api,
        "_calculate_request_duration_for_assignments",
        lambda *_: {"duration_days": None, "duration_minutes": 60},
    )
    monkeypatch.setattr(
        api,
        "_remove_detection_logs_for_daily_request",
        lambda *_: pytest.fail("hourly requests must not remove logs"),
    )
    monkeypatch.setattr(
        api,
        "legacy_request_response",
        lambda record, full_name: {"id": record.id, "full_name": full_name},
    )

    response = api.create_request(
        body=api.PersonnelRequestCreate(
            personnel_id=7,
            request_type="overtime",
            duration_type="hourly",
            start_date="1405-01-18",
            end_date="1405-01-18",
            start_time="09:00",
            end_time="10:00",
        ),
        remove_logs_in_request_dates=True,
        _={},
    )

    assert response["removed_logs_count"] == 0
