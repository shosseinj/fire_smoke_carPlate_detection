from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError as SAIntegrityError

from app.core.employee_type_store import EmployeeTypeStore
from app.core.personnel_store import PersonnelStore


@pytest.mark.postgresql
def test_employee_type_crud_and_personnel_delete_protection(postgres_database, tmp_path) -> None:
    employee_types = EmployeeTypeStore(postgres_database)
    personnel = PersonnelStore(postgres_database, tmp_path)

    custom = employee_types.create(name="consultant")
    assert custom.name == "consultant"
    assert custom.include_in_attendance_reports is False
    assert employee_types.personnel_count(custom.id) == 0

    person = personnel.create(
        fname="Reza",
        lname="Test",
        national_code="0311344119",
        employee_type=None,
        employee_type_id=custom.id,
    )
    assert person.employee_type_id == custom.id
    assert person.employee_type == "consultant"
    assert employee_types.personnel_count(custom.id) == 1

    with pytest.raises(ValueError, match="امکان حذف نوع استخدام"):
        employee_types.delete(custom.id)

    personnel.update(person.id, employee_type="کارمند")
    assert employee_types.delete(custom.id) is True
    assert employee_types.get(custom.id) is None


@pytest.mark.postgresql
def test_database_restricts_deleting_assigned_employee_type(postgres_database, tmp_path) -> None:
    employee_types = EmployeeTypeStore(postgres_database)
    personnel = PersonnelStore(postgres_database, tmp_path)
    custom = employee_types.create(name="temporary")
    personnel.create(
        fname="Reza",
        lname="Test",
        national_code="0311344119",
        employee_type=None,
        employee_type_id=custom.id,
    )

    with pytest.raises(SAIntegrityError):
        with postgres_database.engine.begin() as connection:
            connection.exec_driver_sql(
                "DELETE FROM employee_types WHERE id = %s", (custom.id,)
            )


@pytest.mark.postgresql
def test_personnel_can_filter_by_employee_type_id_and_name(postgres_database, tmp_path) -> None:
    employee_types = EmployeeTypeStore(postgres_database)
    personnel = PersonnelStore(postgres_database, tmp_path)
    custom = employee_types.create(name="vendor")
    personnel.create(
        fname="Reza",
        lname="Test",
        national_code="0311344119",
        employee_type=None,
        employee_type_id=custom.id,
    )

    by_id, count_by_id = personnel.list(employee_type_id=custom.id)
    by_name, count_by_name = personnel.list(employee_type="vendor")
    assert count_by_id == count_by_name == 1
    assert by_id[0].id == by_name[0].id


@pytest.mark.postgresql
def test_employee_type_attendance_report_flag_can_be_updated(postgres_database) -> None:
    employee_types = EmployeeTypeStore(postgres_database)
    custom = employee_types.create(name="office_worker")

    updated = employee_types.update(
        custom.id,
        include_in_attendance_reports=True,
    )

    assert updated is not None
    assert updated.include_in_attendance_reports is True
    assert employee_types.get(custom.id).include_in_attendance_reports is True
