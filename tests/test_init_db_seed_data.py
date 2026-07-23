from __future__ import annotations


def test_personnel_table_metadata_registered() -> None:
    from app.database import metadata
    assert "personnel" in metadata.tables


def test_detection_logs_table_metadata_registered() -> None:
    from app.database import metadata
    assert "detection_logs" in metadata.tables