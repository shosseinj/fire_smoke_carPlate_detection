from __future__ import annotations

from app.database import Connection, Database, Row, ensure_database
from app.time_utils import utc_now_text

import threading
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.core.access_matrix import ACCESS_REGISTRY, permission_key


_USER_COLUMNS = (
    "id, username, password_hash, is_active, created_at_utc, email, "
    "full_name, last_login_utc, login_attempts, locked_until_utc"
    ", auth_version"
)


@dataclass(frozen=True, slots=True)
class UserRecord:
    id: int
    username: str
    password_hash: str
    is_active: bool
    created_at_utc: str
    email: str | None = None
    full_name: str | None = None
    last_login_utc: str | None = None
    login_attempts: int = 0
    locked_until_utc: str | None = None
    auth_version: int = 0


@dataclass(frozen=True, slots=True)
class UserPermissionGrantRecord:
    application: str
    action: str
    scope_type: str
    scope_id: int

    @property
    def permission(self) -> str:
        return f"{self.application}.{self.action}"


@dataclass(frozen=True, slots=True)
class WebSocketTicketRecord:
    user_id: int
    application: str
    scope_type: str
    scope_id: int


class AuthStore:
    """PostgreSQL-backed user and refresh-token revocation storage."""

    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)
        self._lock = threading.Lock()
        self._init_db()

    @property
    def database_url(self) -> str:
        return self.database.url

    def _connection(self) -> Connection:
        return self.database.connection()

    def _init_db(self) -> None:
        pass

    @staticmethod
    def _row_to_record(row: Row) -> UserRecord:
        return UserRecord(
            id=int(row["id"]),
            username=str(row["username"]),
            password_hash=str(row["password_hash"]),
            is_active=bool(row["is_active"]),
            created_at_utc=str(row["created_at_utc"]),
            email=row["email"],
            full_name=row["full_name"],
            last_login_utc=row["last_login_utc"],
            login_attempts=int(row["login_attempts"] or 0),
            locked_until_utc=row["locked_until_utc"],
            auth_version=int(row["auth_version"] or 0),
        )

    def _select_user(self, conn: Connection, where: str, value: object) -> UserRecord | None:
        row = conn.execute(
            f"SELECT {_USER_COLUMNS} FROM users WHERE {where}", (value,)
        ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def seed_default_admin(
        self,
        username: str,
        password: str,
        email: str | None = None,
        full_name: str | None = None,
    ) -> UserRecord | None:
        """Create the default administrator only when the users table is empty."""
        with self._lock, self._connection() as conn:
            if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
                return None
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, email, full_name) "
                "VALUES (?, ?, ?, ?)",
                (username, _hash(password), email, full_name),
            )
            created = self._select_user(conn, "id = ?", cursor.lastrowid)
            return created

    def ensure_default_admin(
        self,
        username: str,
        password: str,
        email: str | None = None,
        full_name: str | None = None,
    ) -> UserRecord:
        """Ensure the configured administrator exists without resetting its password."""
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, email, full_name) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (username) DO NOTHING",
                (username, _hash(password), email, full_name),
            )
            if cursor.lastrowid is not None:
                created = self._select_user(conn, "id = ?", cursor.lastrowid)
                if created is None:
                    raise RuntimeError("Default administrator could not be reloaded")
                return created

            existing = self._select_user(conn, "username = ?", username)
            if existing is None:
                raise RuntimeError("Default administrator could not be ensured")
            return existing

    def verify_credentials(self, username: str, password: str) -> UserRecord | None:
        """Return an active user when the supplied password is valid."""
        user = self.get_user_by_username(username)
        if user is None or not user.is_active:
            return None
        try:
            valid = _checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8"))
        except ValueError:
            return None
        return user if valid else None

    def get_user_by_id(self, user_id: int) -> UserRecord | None:
        with self._lock, self._connection() as conn:
            return self._select_user(conn, "id = ?", user_id)

    def get_user_by_username(self, username: str) -> UserRecord | None:
        with self._lock, self._connection() as conn:
            return self._select_user(conn, "username = ?", username)

    def get_user_by_email(self, email: str) -> UserRecord | None:
        with self._lock, self._connection() as conn:
            return self._select_user(conn, "email = ?", email)

    def create_user(
        self,
        username: str,
        password_hash: str,
        email: str | None = None,
        is_active: bool = True,
        full_name: str | None = None,
    ) -> UserRecord:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users "
                "(username, password_hash, email, is_active, full_name) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, email, int(is_active), full_name),
            )
            user = self._select_user(conn, "id = ?", cursor.lastrowid)
            if user is None:
                raise RuntimeError("Created user could not be reloaded")
            return user

    def list_users(self, offset: int = 0, limit: int = 50) -> tuple[list[UserRecord], int]:
        with self._lock, self._connection() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
            rows = conn.execute(
                f"SELECT {_USER_COLUMNS} FROM users ORDER BY id ASC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_record(row) for row in rows], total

    def update_user(
        self,
        user_id: int,
        username: str | None = None,
        email: str | None = None,
        is_active: bool | None = None,
        full_name: str | None = None,
        update_full_name: bool = False,
    ) -> UserRecord | None:
        with self._lock, self._connection() as conn:
            if self._select_user(conn, "id = ?", user_id) is None:
                return None
            if username is not None:
                conn.execute("UPDATE users SET username = ? WHERE id = ?", (username, user_id))
            if email is not None:
                conn.execute("UPDATE users SET email = ? WHERE id = ?", (email, user_id))
            if is_active is not None:
                conn.execute(
                    "UPDATE users SET is_active = ? WHERE id = ?",
                    (int(is_active), user_id),
                )
            if update_full_name:
                conn.execute(
                    "UPDATE users SET full_name = ? WHERE id = ?", (full_name, user_id)
                )
            return self._select_user(conn, "id = ?", user_id)

    def has_scoped_permission(self, user_id: int, permission: str, scope_type: str, scope_id: int = 0) -> bool:
        if scope_type not in {"global", "building", "section", "camera"}:
            raise ValueError("نوع محدوده انتخاب‌شده معتبر نیست")
        application, action = permission_key(*permission.strip().lower().split(".", 1)).split(".", 1)
        with self._lock, self._connection() as conn:
            params: list[object] = [user_id, application, action]
            targets = ["(scope_type = 'global' AND scope_id = 0)"]
            if scope_type == "building":
                targets.append("(scope_type = 'building' AND scope_id = ?)")
                params.append(scope_id)
            elif scope_type == "section":
                targets.extend((
                    "(scope_type = 'section' AND scope_id = ?)",
                    "(scope_type = 'building' AND scope_id = (SELECT building_id FROM sections WHERE id = ?))",
                ))
                params.extend((scope_id, scope_id))
            elif scope_type == "camera":
                targets.extend((
                    "(scope_type = 'camera' AND scope_id = ?)",
                    "(scope_type = 'section' AND scope_id = (SELECT section_id FROM cam WHERE id = ?))",
                    "(scope_type = 'building' AND scope_id = (SELECT se.building_id FROM cam c JOIN sections se ON se.id = c.section_id WHERE c.id = ?))",
                ))
                params.extend((scope_id, scope_id, scope_id))
            row = conn.execute(
                "SELECT 1 FROM user_permission_grants WHERE user_id = ? AND application = ? "
                "AND action = ? AND (" + " OR ".join(targets) + ") LIMIT 1",
                params,
            ).fetchone()
            return row is not None

    def accessible_scope_ids(self, user_id: int, permission: str, target_type: str) -> set[int] | None:
        """Return allowed target IDs, or None when a global grant allows every target."""
        if target_type not in {"building", "section", "camera"}:
            raise ValueError("نوع منبع انتخاب‌شده معتبر نیست")
        application, action = permission_key(*permission.strip().lower().split(".", 1)).split(".", 1)
        with self._lock, self._connection() as conn:
            grants = conn.execute(
                "SELECT scope_type, scope_id FROM user_permission_grants "
                "WHERE user_id = ? AND application = ? AND action = ?",
                (user_id, application, action),
            ).fetchall()
            if any(row["scope_type"] == "global" for row in grants):
                return None
            building_ids = [int(row["scope_id"]) for row in grants if row["scope_type"] == "building"]
            section_ids = [int(row["scope_id"]) for row in grants if row["scope_type"] == "section"]
            camera_ids = {int(row["scope_id"]) for row in grants if row["scope_type"] == "camera"}
            if target_type == "building":
                return set(building_ids)
            if target_type == "section":
                result = set(section_ids)
                if building_ids:
                    marks = ", ".join("?" for _ in building_ids)
                    result.update(int(row["id"]) for row in conn.execute(f"SELECT id FROM sections WHERE building_id IN ({marks})", building_ids).fetchall())
                return result
            if section_ids:
                marks = ", ".join("?" for _ in section_ids)
                camera_ids.update(int(row["id"]) for row in conn.execute(f"SELECT id FROM cam WHERE section_id IN ({marks})", section_ids).fetchall())
            if building_ids:
                marks = ", ".join("?" for _ in building_ids)
                camera_ids.update(int(row["id"]) for row in conn.execute(f"SELECT c.id FROM cam c JOIN sections s ON s.id = c.section_id WHERE s.building_id IN ({marks})", building_ids).fetchall())
            return camera_ids

    def get_effective_permissions(self, user_id: int) -> frozenset[str]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT application, action FROM user_permission_grants WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return frozenset(f"{row['application']}.{row['action']}" for row in rows)

    def list_user_grants(self, user_id: int) -> list[UserPermissionGrantRecord] | None:
        with self._lock, self._connection() as conn:
            if self._select_user(conn, "id = ?", user_id) is None:
                return None
            rows = conn.execute(
                "SELECT application, action, scope_type, scope_id FROM user_permission_grants "
                "WHERE user_id = ? ORDER BY application, action, scope_type, scope_id", (user_id,)
            ).fetchall()
            return [UserPermissionGrantRecord(str(row["application"]), str(row["action"]), str(row["scope_type"]), int(row["scope_id"])) for row in rows]

    def replace_user_grants(self, user_id: int, grants: tuple[UserPermissionGrantRecord, ...], assigned_by: int | None) -> list[UserPermissionGrantRecord] | None:
        with self._lock, self._connection() as conn:
            if self._select_user(conn, "id = ?", user_id) is None:
                return None
            for grant in grants:
                permission_key(grant.application, grant.action)
                definition = ACCESS_REGISTRY[(grant.application, grant.action)]
                if grant.scope_type not in definition.scope_types:
                    raise ValueError("محدوده برای این دسترسی معتبر نیست")
                if (grant.scope_type == "global") != (grant.scope_id == 0):
                    raise ValueError("شناسه محدوده معتبر نیست")
                table = {"building": "buildings", "section": "sections", "camera": "cam"}.get(grant.scope_type)
                if table and conn.execute(f"SELECT id FROM {table} WHERE id = ?", (grant.scope_id,)).fetchone() is None:
                    raise ValueError("منبع انتخاب‌شده یافت نشد")
            conn.execute("DELETE FROM user_permission_grants WHERE user_id = ?", (user_id,))
            for grant in dict.fromkeys(grants):
                conn.execute(
                    "INSERT INTO user_permission_grants (user_id, application, action, scope_type, scope_id, assigned_by) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, grant.application, grant.action, grant.scope_type, grant.scope_id, assigned_by),
                )
        return self.list_user_grants(user_id)

    def record_audit_event(
        self,
        actor_user_id: int | None,
        action: str,
        target_type: str,
        target_id: str,
        details: str | None = None,
    ) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO auth_audit_log "
                "(actor_user_id, action, target_type, target_id, details) "
                "VALUES (?, ?, ?, ?, ?)",
                (actor_user_id, action, target_type, target_id, details),
            )

    def issue_websocket_ticket(
        self, user_id: int, application: str, scope_type: str, scope_id: int, ttl_seconds: int = 30
    ) -> tuple[str, int]:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO websocket_tickets (token_hash, user_id, application, scope_type, scope_id, expires_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (token_hash, user_id, application, scope_type, scope_id, expires_at),
            )
            conn.execute("DELETE FROM websocket_tickets WHERE expires_at_utc < CURRENT_TIMESTAMP")
        return token, ttl_seconds

    def consume_websocket_ticket(self, token: str, application: str) -> WebSocketTicketRecord | None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "UPDATE websocket_tickets SET consumed_at_utc = CURRENT_TIMESTAMP "
                "WHERE token_hash = ? AND application = ? AND consumed_at_utc IS NULL "
                "AND expires_at_utc > CURRENT_TIMESTAMP "
                "RETURNING user_id, application, scope_type, scope_id",
                (token_hash, application),
            ).fetchone()
            if row is None:
                return None
            return WebSocketTicketRecord(int(row["user_id"]), str(row["application"]), str(row["scope_type"]), int(row["scope_id"]))

    def update_password(self, user_id: int, new_password_hash: str) -> UserRecord | None:
        with self._lock, self._connection() as conn:
            if self._select_user(conn, "id = ?", user_id) is None:
                return None
            conn.execute(
                "UPDATE users SET password_hash = ?, auth_version = auth_version + 1 WHERE id = ?",
                (new_password_hash, user_id),
            )
            return self._select_user(conn, "id = ?", user_id)

    def record_failed_login(
        self,
        user_id: int,
        max_attempts: int,
        locked_until_utc: str,
    ) -> UserRecord | None:
        """Atomically increment failures and lock once the configured threshold is reached."""
        with self._lock, self._connection() as conn:
            if self._select_user(conn, "id = ?", user_id) is None:
                return None
            conn.execute(
                "UPDATE users SET login_attempts = login_attempts + 1 WHERE id = ?",
                (user_id,),
            )
            attempts = int(
                conn.execute(
                    "SELECT login_attempts FROM users WHERE id = ?", (user_id,)
                ).fetchone()[0]
            )
            if attempts >= max_attempts:
                conn.execute(
                    "UPDATE users SET locked_until_utc = ? WHERE id = ?",
                    (locked_until_utc, user_id),
                )
            return self._select_user(conn, "id = ?", user_id)

    def record_successful_login(self, user_id: int, last_login_utc: str) -> UserRecord | None:
        with self._lock, self._connection() as conn:
            if self._select_user(conn, "id = ?", user_id) is None:
                return None
            conn.execute(
                "UPDATE users SET login_attempts = 0, locked_until_utc = NULL, "
                "last_login_utc = ? WHERE id = ?",
                (last_login_utc, user_id),
            )
            return self._select_user(conn, "id = ?", user_id)

    def delete_user(self, user_id: int) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cursor.rowcount > 0

    def count_users(self) -> int:
        with self._lock, self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def revoke_token(self, jti: str, expires_at_utc: str) -> None:
        now = _utc_now_text()
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO revoked_tokens "
                "(jti, revoked_at_utc, expires_at_utc) VALUES (?, ?, ?) "
                "ON CONFLICT(jti) DO NOTHING",
                (jti, now, expires_at_utc),
            )

    def is_token_revoked(self, jti: str) -> bool:
        with self._lock, self._connection() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,)
                ).fetchone()
                is not None
            )

    def purge_expired_revoked(self) -> int:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM revoked_tokens WHERE expires_at_utc < ?", (_utc_now_text(),)
            )
            return cursor.rowcount

    def close(self) -> None:
        return None


def _utc_now_text() -> str:
    return utc_now_text()


def _hash(password: str) -> str:
    import bcrypt

    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _checkpw(plain: bytes, stored: bytes) -> bool:
    import bcrypt

    return bcrypt.checkpw(plain, stored)
