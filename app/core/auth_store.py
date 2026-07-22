from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_USER_COLUMNS = (
    "id, username, password_hash, role, is_active, created_at_utc, email, "
    "full_name, last_login_utc, login_attempts, locked_until_utc"
)


@dataclass(frozen=True, slots=True)
class UserRecord:
    id: int
    username: str
    password_hash: str
    role: str
    is_active: bool
    created_at_utc: str
    email: str | None = None
    full_name: str | None = None
    last_login_utc: str | None = None
    login_attempts: int = 0
    locked_until_utc: str | None = None


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
        with self._lock, self._connection() as conn:
            conn.execute("UPDATE users SET role = 'admin' WHERE role = 'superuser'")
            conn.execute("UPDATE users SET role = 'viewer' WHERE role = 'user'")


    @staticmethod
    def _row_to_record(row: Row) -> UserRecord:
        return UserRecord(
            id=int(row["id"]),
            username=str(row["username"]),
            password_hash=str(row["password_hash"]),
            role=str(row["role"]),
            is_active=bool(row["is_active"]),
            created_at_utc=str(row["created_at_utc"]),
            email=row["email"],
            full_name=row["full_name"],
            last_login_utc=row["last_login_utc"],
            login_attempts=int(row["login_attempts"] or 0),
            locked_until_utc=row["locked_until_utc"],
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
        role: str = "admin",
        email: str | None = None,
        full_name: str | None = None,
    ) -> UserRecord | None:
        """Create the default administrator only when the users table is empty."""
        with self._lock, self._connection() as conn:
            if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
                return None
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, role, email, full_name) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, _hash(password), role, email, full_name),
            )
            return self._select_user(conn, "id = ?", cursor.lastrowid)

    def ensure_default_admin(
        self,
        username: str,
        password: str,
        role: str = "admin",
        email: str | None = None,
        full_name: str | None = None,
    ) -> UserRecord:
        """Ensure the configured administrator exists without resetting its password."""
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, role, email, full_name) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (username) DO NOTHING",
                (username, _hash(password), role, email, full_name),
            )
            if cursor.lastrowid is not None:
                created = self._select_user(conn, "id = ?", cursor.lastrowid)
                if created is None:
                    raise RuntimeError("Default administrator could not be reloaded")
                return created

            existing = self._select_user(conn, "username = ?", username)
            if existing is None:
                raise RuntimeError("Default administrator could not be ensured")
            if existing.role != role:
                conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, existing.id))
                existing = self._select_user(conn, "id = ?", existing.id)
                if existing is None:
                    raise RuntimeError("Default administrator could not be reloaded")
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
        role: str,
        email: str | None = None,
        is_active: bool = True,
        full_name: str | None = None,
    ) -> UserRecord:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users "
                "(username, password_hash, role, email, is_active, full_name) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (username, password_hash, role, email, int(is_active), full_name),
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

    def set_user_role(self, user_id: int, role: str) -> UserRecord | None:
        with self._lock, self._connection() as conn:
            if self._select_user(conn, "id = ?", user_id) is None:
                return None
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
            return self._select_user(conn, "id = ?", user_id)

    def update_password(self, user_id: int, new_password_hash: str) -> UserRecord | None:
        with self._lock, self._connection() as conn:
            if self._select_user(conn, "id = ?", user_id) is None:
                return None
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
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

    def count_active_admins(self) -> int:
        with self._lock, self._connection() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1"
                ).fetchone()[0]
            )

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
