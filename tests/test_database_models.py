from __future__ import annotations


def test_personnel_table_in_metadata() -> None:
    from app.database import metadata
    assert "personnel" in metadata.tables


def test_sources_table_owns_nullable_fps() -> None:
    from app.database import metadata

    assert "sources" in metadata.tables
    assert metadata.tables["sources"].c.fps.nullable is True


def test_detection_logs_table_in_metadata() -> None:
    from app.database import metadata
    assert "detection_logs" in metadata.tables


def test_employee_types_table_uses_name_without_code() -> None:
    from app.database import metadata

    employee_types = metadata.tables["employee_types"]
    assert "name" in employee_types.c
    assert "code" not in employee_types.c
    assert "include_in_attendance_reports" in employee_types.c
    assert employee_types.c.include_in_attendance_reports.nullable is False
    assert "employee_type_id" in metadata.tables["personnel"].c
