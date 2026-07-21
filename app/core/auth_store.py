from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class UserRecord:
    id: int
    username: str
    password_hash: str
    role: str
    is_active: bool
    created_at_utc: str
    email: str | None = None


class AuthStore:
    """SQLite-backed user storage for JWT authentication."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    username        TEXT    NOT NULL UNIQUE,
                    password_hash   TEXT    NOT NULL,
                    role            TEXT    NOT NULL DEFAULT 'viewer',
                    is_active       INTEGER NOT NULL DEFAULT 1,
                    created_at_utc  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    jti             TEXT    PRIMARY KEY,
                    revoked_at_utc  TEXT    NOT NULL,
                    expires_at_utc  TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_revoked_expires ON revoked_tokens(expires_at_utc);
            """)
            self._migrate_add_email(conn)
            # Create email index after potential migration
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            except Exception:
                pass

    def _migrate_add_email(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "email" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")

    def _row_to_record(self, row: sqlite3.Row) -> UserRecord:
        email: str | None = None
        if "email" in row.keys():
            email = row["email"]
        return UserRecord(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            created_at_utc=row["created_at_utc"],
            email=email,
        )

    def seed_default_admin(
        self, username: str, password: str, role: str = "admin"
    ) -> UserRecord | None:
        """Create the default admin user if no users exist yet."""
        with self._lock, self._connection() as conn:
            existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if existing > 0:
                return None
            password_hash = _hash(password)
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash, role),
            )
            row = conn.execute(
                "SELECT id, username, password_hash, role, is_active, created_at_utc, email "
                "FROM users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def verify_credentials(self, username: str, password: str) -> UserRecord | None:
        """Return the user record if credentials are valid, else None."""
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, role, is_active, created_at_utc, email "
                "FROM users WHERE username = ? AND is_active = 1",
                (username,),
            ).fetchone()
        if row is None:
            return None
        stored_hash = row["password_hash"].encode("utf-8")
        if not _checkpw(password.encode("utf-8"), stored_hash):
            return None
        return self._row_to_record(row)

    def get_user_by_id(self, user_id: int) -> UserRecord | None:
        """Return user record by primary key, or None."""
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, role, is_active, created_at_utc, email "
                "FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_user_by_username(self, username: str) -> UserRecord | None:
        """Return user record by username, or None."""
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, role, is_active, created_at_utc, email "
                "FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_user_by_email(self, email: str) -> UserRecord | None:
        """Return user record by email, or None."""
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, role, is_active, created_at_utc, email "
                "FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def create_user(
        self,
        username: str,
        password_hash: str,
        role: str,
        email: str | None = None,
        is_active: bool = True,
    ) -> UserRecord:
        """Insert a new user and return the record. Raises sqlite3.IntegrityError on duplicate."""
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, role, email, is_active) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, role, email, int(is_active)),
            )
            row = conn.execute(
                "SELECT id, username, password_hash, role, is_active, created_at_utc, email "
                "FROM users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            return self._row_to_record(row)

    def list_users(
        self, offset: int = 0, limit: int = 50
    ) -> tuple[list[UserRecord], int]:
        """Return paginated user list and total count."""
        with self._lock, self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            rows = conn.execute(
                "SELECT id, username, password_hash, role, is_active, created_at_utc, email "
                "FROM users ORDER BY id ASC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            users = [self._row_to_record(r) for r in rows]
            return users, total

    def update_user(
        self,
        user_id: int,
        username: str | None = None,
        email: str | None = None,
        is_active: bool | None = None,
    ) -> UserRecord | None:
        """Update user fields. Returns updated record or None if not found."""
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if existing is None:
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
            row = conn.execute(
                "SELECT id, username, password_hash, role, is_active, created_at_utc, email "
                "FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            return self._row_to_record(row) if row else None

    def set_user_role(self, user_id: int, role: str) -> UserRecord | None:
        """Change a user's role. Returns updated record or None if not found."""
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if existing is None:
                return None
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
            row = conn.execute(
                "SELECT id, username, password_hash, role, is_active, created_at_utc, email "
                "FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            return self._row_to_record(row) if row else None

    def update_password(
        self, user_id: int, new_password_hash: str
    ) -> UserRecord | None:
        """Update a user's password hash. Returns updated record or None."""
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if existing is None:
                return None
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_password_hash, user_id),
            )
            row = conn.execute(
                "SELECT id, username, password_hash, role, is_active, created_at_utc, email "
                "FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            return self._row_to_record(row) if row else None

    def delete_user(self, user_id: int) -> bool:
        """Delete a user by id. Returns True if deleted, False if not found."""
        with self._lock, self._connection() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cursor.rowcount > 0

    def count_users(self) -> int:
        with self._lock, self._connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def count_active_admins(self) -> int:
        """Count users with admin role and is_active=1."""
        with self._lock, self._connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1"
            ).fetchone()[0]

    # ── Token blacklist ──────────────────────────────────────────────

    def revoke_token(self, jti: str, expires_at_utc: str) -> None:
        """Add a token to the revoked set."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO revoked_tokens (jti, revoked_at_utc, expires_at_utc) "
                "VALUES (?, ?, ?)",
                (jti, now, expires_at_utc),
            )

    def is_token_revoked(self, jti: str) -> bool:
        """Check if a token has been revoked."""
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,)
            ).fetchone()
            return row is not None

    def purge_expired_revoked(self) -> int:
        """Remove expired revoked tokens. Returns count purged."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM revoked_tokens WHERE expires_at_utc < ?", (now,)
            )
            return cursor.rowcount

    def close(self) -> None:
        pass


def _hash(password: str) -> str:
    import bcrypt
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _checkpw(plain: bytes, stored: bytes) -> bool:
    import bcrypt
    return bcrypt.checkpw(plain, stored)
