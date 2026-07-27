from __future__ import annotations

from pydantic import BaseModel

from app.database import Connection, Row


class UserBrief(BaseModel):
    """Minimal user identity embedded in audit fields of response models."""

    id: int
    full_name: str


def resolve_user_brief(user_id: int | None, db: Connection) -> UserBrief | None:
    """Resolve a ``created_by`` / ``updated_by`` user ID to a ``UserBrief``.

    Returns ``None`` when the input is ``None`` or the user is not found.
    """
    if user_id is None:
        return None
    row: Row | None = db.execute(
        "SELECT id, full_name FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if row is None:
        return None
    return UserBrief(id=int(row["id"]), full_name=str(row["full_name"] or ""))
