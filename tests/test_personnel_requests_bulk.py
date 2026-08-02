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
        lambda _personnel, _shift, item: {
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
