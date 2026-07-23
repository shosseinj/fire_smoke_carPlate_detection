"""Tests for RequestStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.personnel_store import PersonnelStore
from app.core.request_store import (
    RequestStore,
    VALID_REQUEST_STATUSES,
    VALID_REQUEST_TYPES,
)
from app.database import Database


@pytest.fixture
def stores(
    postgres_database: Database,
    tmp_path: Path,
) -> tuple[RequestStore, PersonnelStore, Path]:
    with postgres_database.connection() as connection:
        connection.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("request-reviewer", "unused-in-store-tests", "admin"),
        )
    return (
        RequestStore(postgres_database),
        PersonnelStore(postgres_database, tmp_path / "media"),
        tmp_path,
    )


@pytest.fixture
def person_id(stores: tuple[RequestStore, PersonnelStore, Path]) -> int:
    _, ps, _ = stores
    p = ps.create(fname="Test", lname="User", national_code="1234567891", employee_type="employee")
    return p.id


class TestRequestStore:
    def test_create(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req = rs.create(personnel_id=person_id, request_type="leave",
                        start_date="2026-08-01", end_date="2026-08-03")
        assert req.id > 0
        assert req.status == "pending"
        assert req.request_type == "leave"

    def test_create_jalali_date(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req = rs.create(personnel_id=person_id, request_type="leave",
                        start_date="1405-05-10", end_date="1405-05-12")
        assert req.id > 0

    def test_create_invalid_type(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        with pytest.raises(ValueError, match="Invalid request type"):
            rs.create(personnel_id=person_id, request_type="invalid",
                      start_date="2026-08-01", end_date="2026-08-03")

    def test_create_missing_dates(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        with pytest.raises(ValueError, match="required"):
            rs.create(personnel_id=person_id, request_type="leave",
                      start_date="", end_date="")

    def test_create_start_after_end(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        with pytest.raises(ValueError, match="start_date must not be after end_date"):
            rs.create(personnel_id=person_id, request_type="leave",
                      start_date="2026-08-10", end_date="2026-08-05")

    def test_create_nonexistent_personnel(self, stores: tuple[RequestStore, PersonnelStore, Path]) -> None:
        rs, _, _ = stores
        with pytest.raises(ValueError, match="Personnel not found"):
            rs.create(personnel_id=99999, request_type="leave",
                      start_date="2026-08-01", end_date="2026-08-03")

    def test_get(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req = rs.create(personnel_id=person_id, request_type="leave",
                        start_date="2026-08-01", end_date="2026-08-03")
        fetched = rs.get(req.id)
        assert fetched is not None
        assert fetched.id == req.id

    def test_get_nonexistent(self, stores: tuple[RequestStore, PersonnelStore, Path]) -> None:
        rs, _, _ = stores
        assert rs.get(99999) is None

    def test_approve(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req = rs.create(personnel_id=person_id, request_type="leave",
                        start_date="2026-09-01", end_date="2026-09-02")
        approved = rs.approve(req.id, approved_by=1)
        assert approved is not None
        assert approved.status == "approved"
        assert approved.approved_by == 1

    def test_reject(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req = rs.create(personnel_id=person_id, request_type="leave",
                        start_date="2026-09-05", end_date="2026-09-06")
        rejected = rs.reject(req.id, approved_by=1, rejection_reason="Not approved")
        assert rejected is not None
        assert rejected.status == "rejected"
        assert rejected.rejection_reason == "Not approved"

    def test_approve_already_processed(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req = rs.create(personnel_id=person_id, request_type="leave",
                        start_date="2026-10-01", end_date="2026-10-02")
        rs.approve(req.id, approved_by=1)
        with pytest.raises(ValueError, match="already"):
            rs.approve(req.id, approved_by=1)

    def test_cancel(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req = rs.create(personnel_id=person_id, request_type="leave",
                        start_date="2026-11-01", end_date="2026-11-02")
        cancelled = rs.cancel(req.id)
        assert cancelled is not None
        assert cancelled.status == "cancelled"

    def test_cancel_approved_fails(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req = rs.create(personnel_id=person_id, request_type="leave",
                        start_date="2026-12-01", end_date="2026-12-02")
        rs.approve(req.id, approved_by=1)
        with pytest.raises(ValueError, match="Cannot cancel"):
            rs.cancel(req.id)

    def test_delete(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req = rs.create(personnel_id=person_id, request_type="leave",
                        start_date="2026-07-01", end_date="2026-07-02")
        assert rs.delete(req.id) is True
        assert rs.get(req.id) is None

    def test_delete_nonexistent(self, stores: tuple[RequestStore, PersonnelStore, Path]) -> None:
        rs, _, _ = stores
        assert rs.delete(99999) is False

    def test_list(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        rs.create(personnel_id=person_id, request_type="leave",
                  start_date="2026-08-01", end_date="2026-08-03")
        rs.create(personnel_id=person_id, request_type="sick_leave",
                  start_date="2026-09-01", end_date="2026-09-02")
        records, total = rs.list(limit=100)
        assert total >= 2

    def test_list_filter_personnel(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        rs.create(personnel_id=person_id, request_type="leave",
                  start_date="2026-08-01", end_date="2026-08-03")
        records, total = rs.list(personnel_id=person_id)
        assert total >= 1

    def test_list_filter_status(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req = rs.create(personnel_id=person_id, request_type="leave",
                        start_date="2026-08-01", end_date="2026-08-03")
        rs.approve(req.id, approved_by=1)
        records, total = rs.list(status="approved")
        assert total >= 1

    def test_overlap_prevention(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req1 = rs.create(personnel_id=person_id, request_type="leave",
                         start_date="2026-08-10", end_date="2026-08-15")
        rs.approve(req1.id, approved_by=1)
        # Pending requests may overlap; approval enforces the conflict.
        req2 = rs.create(personnel_id=person_id, request_type="leave",
                         start_date="2026-08-12", end_date="2026-08-14")
        with pytest.raises(ValueError, match="Overlapping approved"):
            rs.approve(req2.id, approved_by=1)

    def test_overlap_different_type_allowed(self, stores: tuple[RequestStore, PersonnelStore, Path], person_id: int) -> None:
        rs, _, _ = stores
        req1 = rs.create(personnel_id=person_id, request_type="leave",
                         start_date="2026-08-10", end_date="2026-08-15")
        rs.approve(req1.id, approved_by=1)
        # Different type — allowed
        req2 = rs.create(personnel_id=person_id, request_type="sick_leave",
                         start_date="2026-08-12", end_date="2026-08-14")
        assert req2.id > 0
        assert rs.approve(req2.id, approved_by=1).status == "approved"

    def test_valid_request_types(self) -> None:
        assert "leave" in VALID_REQUEST_TYPES
        assert "sick_leave" in VALID_REQUEST_TYPES

    def test_valid_statuses(self) -> None:
        assert "pending" in VALID_REQUEST_STATUSES
        assert "approved" in VALID_REQUEST_STATUSES
