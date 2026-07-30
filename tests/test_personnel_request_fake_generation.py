from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException

from app.api import personnel_requests


class _CapturingRequestStore:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []

    def create(self, **values: object) -> SimpleNamespace:
        self.created.append(values)
        return SimpleNamespace(id=len(self.created), **values)


def test_generate_fake_requests_uses_all_personnel_pages_and_calculated_fields(
    monkeypatch,
) -> None:
    without_shift = SimpleNamespace(id=1, shift_id=None)
    first_page = [without_shift] * 1000
    eligible = SimpleNamespace(id=1001, shift_id=77)

    class PersonnelStore:
        def list(self, *, offset: int, limit: int):
            assert limit == 1000
            if offset == 0:
                return first_page, 1001
            assert offset == 1000
            return [eligible], 1001

    request_store = _CapturingRequestStore()
    shift = SimpleNamespace(id=77)
    calculation = {"duration_days": 0.0, "duration_minutes": 135}

    monkeypatch.setattr(personnel_requests, "get_request_store", lambda: request_store)
    monkeypatch.setattr(
        personnel_requests,
        "get_personnel_store",
        lambda: PersonnelStore(),
    )
    monkeypatch.setattr(personnel_requests, "_get_shift", lambda person: shift)
    monkeypatch.setattr(personnel_requests, "get_holiday_store", lambda: object())
    monkeypatch.setattr(
        personnel_requests,
        "calculate_request_duration",
        lambda *args: calculation,
    )
    monkeypatch.setattr(personnel_requests, "utc_now_text", lambda: "2026-07-30T10:00:00Z")
    monkeypatch.setattr(personnel_requests._random, "choice", lambda values: values[-1])
    monkeypatch.setattr(personnel_requests._random, "randint", lambda low, high: low)

    response = personnel_requests.generate_fake_requests(
        count=1,
        from_date=None,
        to_date=None,
        admin_user=object(),
    )

    assert response["count"] == 1
    assert len(request_store.created) == 1
    created = request_store.created[0]
    assert created["personnel_id"] == 1001
    assert created["request_type"] == "overtime"
    assert created["duration_type"] == "hourly"
    assert created["start_date"] == created["end_date"]
    assert created["start_time"] == "08:45"
    assert created["end_time"] == "09:45"
    assert created["duration_days"] is None
    assert created["duration_minutes"] == 135
    assert created["status"] == "rejected"
    assert created["reviewed_at"] == "2026-07-30T10:00:00Z"
    assert created["rejection_reason"] == "Generated test rejection"


def test_generate_fake_requests_review_metadata_matches_each_status(monkeypatch) -> None:
    person = SimpleNamespace(id=8, shift_id=3)

    class PersonnelStore:
        def list(self, *, offset: int, limit: int):
            return ([person], 1) if offset == 0 else ([], 1)

    request_store = _CapturingRequestStore()
    statuses = iter(("pending", "approved", "rejected"))

    def choose(values):
        if values == ("pending", "approved", "rejected"):
            return next(statuses)
        return values[0]

    monkeypatch.setattr(personnel_requests, "get_request_store", lambda: request_store)
    monkeypatch.setattr(
        personnel_requests,
        "get_personnel_store",
        lambda: PersonnelStore(),
    )
    monkeypatch.setattr(personnel_requests, "_get_shift", lambda value: object())
    monkeypatch.setattr(personnel_requests, "get_holiday_store", lambda: object())
    monkeypatch.setattr(
        personnel_requests,
        "calculate_request_duration",
        lambda *args: {"duration_days": 2.0, "duration_minutes": None},
    )
    monkeypatch.setattr(personnel_requests, "utc_now_text", lambda: "reviewed-now")
    monkeypatch.setattr(personnel_requests._random, "choice", choose)
    monkeypatch.setattr(personnel_requests._random, "randint", lambda low, high: low)

    response = personnel_requests.generate_fake_requests(
        count=3,
        from_date=None,
        to_date=None,
        admin_user=object(),
    )

    assert response["count"] == 3
    assert [item["status"] for item in request_store.created] == [
        "pending",
        "approved",
        "rejected",
    ]
    assert [item["reviewed_at"] for item in request_store.created] == [
        None,
        "reviewed-now",
        "reviewed-now",
    ]
    assert [item["rejection_reason"] for item in request_store.created] == [
        None,
        None,
        "Generated test rejection",
    ]
    assert all(item["duration_days"] == 2.0 for item in request_store.created)
    assert all(item["duration_minutes"] is None for item in request_store.created)


def _configure_single_eligible_person(monkeypatch) -> _CapturingRequestStore:
    person = SimpleNamespace(id=8, shift_id=3)

    class PersonnelStore:
        def list(self, *, offset: int, limit: int):
            return ([person], 1) if offset == 0 else ([], 1)

    request_store = _CapturingRequestStore()
    monkeypatch.setattr(personnel_requests, "get_request_store", lambda: request_store)
    monkeypatch.setattr(
        personnel_requests,
        "get_personnel_store",
        lambda: PersonnelStore(),
    )
    monkeypatch.setattr(personnel_requests, "_get_shift", lambda value: object())
    monkeypatch.setattr(personnel_requests, "get_holiday_store", lambda: object())
    monkeypatch.setattr(
        personnel_requests,
        "calculate_request_duration",
        lambda *args: {"duration_days": 1.0, "duration_minutes": None},
    )
    monkeypatch.setattr(personnel_requests._random, "choice", lambda values: values[0])
    monkeypatch.setattr(personnel_requests._random, "randint", lambda low, high: low)
    return request_store


def test_generate_fake_requests_has_no_count_upper_bound(monkeypatch) -> None:
    request_store = _configure_single_eligible_person(monkeypatch)

    app = FastAPI()
    app.include_router(personnel_requests.router)
    operation = app.openapi()["paths"][
        "/api/v1/personnel-requests/generate-fake"
    ]["post"]
    count_parameter = next(
        parameter for parameter in operation["parameters"] if parameter["name"] == "count"
    )
    assert "maximum" not in count_parameter["schema"]

    response = personnel_requests.generate_fake_requests(
        count=51,
        from_date="1405-05-01",
        to_date="1405-05-01",
        admin_user=object(),
    )

    assert response["count"] == 51
    assert len(request_store.created) == 51


@pytest.mark.parametrize(
    ("from_date", "to_date"),
    [
        ("1405-05-01", None),
        (None, "1405-05-01"),
        ("invalid", "1405-05-01"),
        ("1405-05-02", "1405-05-01"),
    ],
)
def test_generate_fake_requests_validates_paired_jalali_range(
    monkeypatch,
    from_date: str | None,
    to_date: str | None,
) -> None:
    _configure_single_eligible_person(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        personnel_requests.generate_fake_requests(
            count=1,
            from_date=from_date,
            to_date=to_date,
            admin_user=object(),
        )

    assert exc_info.value.status_code == 400


def test_generate_fake_requests_uses_inclusive_jalali_range_and_clamps_daily_end(
    monkeypatch,
) -> None:
    request_store = _configure_single_eligible_person(monkeypatch)
    randint_values = iter((1, 0, 0, 1))
    monkeypatch.setattr(
        personnel_requests._random,
        "randint",
        lambda low, high: next(randint_values),
    )

    response = personnel_requests.generate_fake_requests(
        count=2,
        from_date="1405-05-01",
        to_date="1405-05-02",
        admin_user=object(),
    )

    expected_start = personnel_requests.validate_jalali_date("1405-05-01")
    expected_end = personnel_requests.validate_jalali_date("1405-05-02")
    assert response["count"] == 2
    assert [date.fromisoformat(item["start_date"]) for item in request_store.created] == [
        expected_end,
        expected_start,
    ]
    assert all(
        date.fromisoformat(item["end_date"]) <= expected_end
        for item in request_store.created
    )


def test_generate_fake_requests_default_range_remains_last_180_days(monkeypatch) -> None:
    request_store = _configure_single_eligible_person(monkeypatch)
    randint_values = iter((0, 0, 180, 0))
    monkeypatch.setattr(
        personnel_requests._random,
        "randint",
        lambda low, high: next(randint_values),
    )

    personnel_requests.generate_fake_requests(
        count=2,
        from_date=None,
        to_date=None,
        admin_user=object(),
    )

    assert [date.fromisoformat(item["start_date"]) for item in request_store.created] == [
        date.today() - personnel_requests.timedelta(days=180),
        date.today(),
    ]
