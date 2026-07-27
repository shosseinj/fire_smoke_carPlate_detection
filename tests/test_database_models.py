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
