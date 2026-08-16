from __future__ import annotations

from dataclasses import dataclass

from fastapi import WebSocket

from app.config import settings
from app.core.auth import _auth_disabled, get_auth_store, has_scoped_permission
from app.core.auth_store import UserRecord, WebSocketTicketRecord


@dataclass(frozen=True, slots=True)
class WebSocketAccess:
    user_id: int
    application: str
    scope_type: str
    scope_id: int


async def authenticate_websocket(
    websocket: WebSocket,
    application: str,
    ticket: str | None,
    *,
    runtime=None,
    fullscreen_source: str | None = None,
) -> WebSocketAccess | None:
    if _auth_disabled():
        return WebSocketAccess(0, application, "global", 0)
    if not ticket:
        await websocket.close(code=1008, reason="بلیط اتصال ارسال نشده است")
        return None
    store = get_auth_store()
    consumed = store.consume_websocket_ticket(ticket, application)
    if consumed is None:
        await websocket.close(code=1008, reason="بلیط اتصال نامعتبر، منقضی یا استفاده‌شده است")
        return None
    user = store.get_user_by_id(consumed.user_id)
    if user is None or not user.is_active or not _permission_still_valid(user, consumed):
        await websocket.close(code=1008, reason="دسترسی شما به این جریان معتبر نیست")
        return None
    if consumed.scope_type != "global":
        if runtime is None or not fullscreen_source:
            await websocket.close(code=1008, reason="دسترسی محدود نیازمند انتخاب یک منبع است")
            return None
        source = runtime.registry.get(fullscreen_source)
        if source is None or source.room_id is None or not _scope_covers_room(runtime, consumed, source.room_id):
            await websocket.close(code=1008, reason="منبع انتخاب‌شده خارج از محدوده دسترسی شما است")
            return None
    return WebSocketAccess(consumed.user_id, consumed.application, consumed.scope_type, consumed.scope_id)


def websocket_access_still_valid(access: WebSocketAccess) -> bool:
    if _auth_disabled():
        return True
    store = get_auth_store()
    user = store.get_user_by_id(access.user_id)
    if user is None or not user.is_active:
        return False
    return _permission_still_valid(
        user,
        WebSocketTicketRecord(access.user_id, access.application, access.scope_type, access.scope_id),
    )


def _permission_still_valid(user: UserRecord, ticket: WebSocketTicketRecord) -> bool:
    if user.username == settings.auth_default_admin_username:
        return True
    return has_scoped_permission(user, f"{ticket.application}.read", ticket.scope_type, ticket.scope_id)


def _scope_covers_room(runtime, ticket: WebSocketTicketRecord, room_id: int) -> bool:
    if ticket.scope_type == "global":
        return True
    with runtime.database.connection() as conn:
        row = conn.execute(
            "SELECT r.cam_id, c.section_id, s.building_id FROM rooms r "
            "LEFT JOIN cam c ON c.id = r.cam_id "
            "LEFT JOIN sections s ON s.id = c.section_id WHERE r.id = ?",
            (room_id,),
        ).fetchone()
    if row is None:
        return False
    if ticket.scope_type == "room":
        return room_id == ticket.scope_id
    if ticket.scope_type == "camera":
        return row["cam_id"] is not None and int(row["cam_id"]) == ticket.scope_id
    if ticket.scope_type == "section":
        return row["section_id"] is not None and int(row["section_id"]) == ticket.scope_id
    return row["building_id"] is not None and int(row["building_id"]) == ticket.scope_id
