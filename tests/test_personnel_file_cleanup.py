from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.core.personnel_store import PersonnelStore


class _Result:
    def __init__(
        self,
        *,
        rows: list[dict[str, str]] | None = None,
        row: dict[str, int] | None = None,
        rowcount: int = 0,
    ) -> None:
        self._rows = rows or []
        self._row = row
        self.rowcount = rowcount

    def fetchall(self) -> list[dict[str, str]]:
        return self._rows

    def fetchone(self) -> dict[str, int] | None:
        return self._row


class _Connection:
    def __init__(self, snapshot_key: str, *, personnel_exists: bool = True) -> None:
        self._snapshot_key = snapshot_key
        self._personnel_exists = personnel_exists

    def execute(self, statement: str, _: tuple[int]) -> _Result:
        if statement.startswith("SELECT id FROM personnel"):
            row = {"id": 7} if self._personnel_exists else None
            return _Result(row=row)
        if statement.startswith("SELECT") and "FROM personnel WHERE id" in statement:
            row = {"id": 7} if self._personnel_exists else None
            return _Result(row=row)
        if statement.startswith("SELECT storage_key FROM personnel_images"):
            return _Result(rows=[{"storage_key": self._snapshot_key}])
        if statement.startswith("SELECT embedding_id FROM personnel_images"):
            return _Result(rows=[])
        return _Result(rowcount=1)


class _TestPersonnelStore(PersonnelStore):
    def __init__(
        self,
        media_root: Path,
        snapshot_key: str,
        *,
        personnel_exists: bool = True,
    ) -> None:
        self._media_root = media_root
        self._snapshot_dir = media_root / "personnel_snapshots"
        self._cropped_face_dir = media_root / "personnel_cropped_faces"
        self._snapshot_dir.mkdir(parents=True)
        self._cropped_face_dir.mkdir(parents=True)
        self._connection_value = _Connection(
            snapshot_key,
            personnel_exists=personnel_exists,
        )
        self._lock = threading.RLock()

    @contextmanager
    def _connection(self) -> Iterator[_Connection]:
        yield self._connection_value


def test_delete_removes_personnel_snapshot_and_cropped_files(tmp_path: Path) -> None:
    snapshot_key = "personnel_snapshots/personnel_7_snapshot.jpg"
    store = _TestPersonnelStore(tmp_path, snapshot_key)
    snapshot = tmp_path / snapshot_key
    cropped = store._cropped_face_dir / "personnel_7_face.jpg"
    other_snapshot = store._snapshot_dir / "personnel_70_snapshot.jpg"
    other_cropped = store._cropped_face_dir / "personnel_8_face.jpg"

    for path in (snapshot, cropped, other_snapshot, other_cropped):
        path.write_bytes(b"image")

    assert store.delete(7) is True
    assert not snapshot.exists()
    assert not cropped.exists()
    assert other_snapshot.is_file()
    assert other_cropped.is_file()


def test_delete_missing_personnel_does_not_remove_orphan_files(tmp_path: Path) -> None:
    snapshot_key = "personnel_snapshots/personnel_7_snapshot.jpg"
    store = _TestPersonnelStore(tmp_path, snapshot_key, personnel_exists=False)
    snapshot = tmp_path / snapshot_key
    cropped = store._cropped_face_dir / "personnel_7_face.jpg"

    snapshot.write_bytes(b"image")
    cropped.write_bytes(b"image")

    assert store.delete(7) is False
    assert snapshot.is_file()
    assert cropped.is_file()
