from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.core.auth_store import AuthStore, UserRecord

LOGGER = logging.getLogger("uvicorn.error")

_bearer_scheme = HTTPBearer(auto_error=False)

# Global auth store instance — initialised once at runtime build time.
_auth_store: AuthStore | None = None


def get_auth_store() -> AuthStore:
    global _auth_store
    if _auth_store is None:
        _auth_store = AuthStore(settings.auth_db_path)
        _auth_store.seed_default_admin(
            username=settings.auth_default_admin_username,
            password=settings.auth_default_admin_password,
        )
    return _auth_store


# ── Password utilities ──────────────────────────────────────────────


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), stored_hash.encode("utf-8"))


# ── JWT utilities ───────────────────────────────────────────────────


def _make_jwt_payload(
    user_id: int,
    username: str,
    role: str,
    token_type: str,
    expires_minutes: int | None = None,
) -> dict[str, Any]:
    if token_type == "access":
        if expires_minutes is None:
            expires_minutes = settings.jwt_expiry_minutes
    else:
        if expires_minutes is None:
            expires_minutes = settings.jwt_refresh_expiry_minutes
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
        "type": token_type,
        "jti": _generate_jti(),
    }
    return payload


def _generate_jti() -> str:
    import uuid
    return str(uuid.uuid4())


def create_access_token(
    user_id: int,
    username: str,
    role: str,
    expires_minutes: int | None = None,
) -> str:
    """Create a signed JWT access token."""
    payload = _make_jwt_payload(user_id, username, role, "access", expires_minutes)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(
    user_id: int,
    username: str,
    role: str,
    expires_minutes: int | None = None,
) -> str:
    """Create a signed JWT refresh token."""
    payload = _make_jwt_payload(user_id, username, role, "refresh", expires_minutes)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT access token. Returns payload or None."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        if payload.get("type") != "access":
            return None
        return payload
    except jwt.ExpiredSignatureError:
        LOGGER.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError as exc:
        LOGGER.warning("Invalid JWT token: %s", exc)
        return None


def decode_refresh_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT refresh token. Returns payload or None."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        if payload.get("type") != "refresh":
            return None
        store = get_auth_store()
        if store.is_token_revoked(payload.get("jti", "")):
            LOGGER.warning("Refresh token has been revoked")
            return None
        return payload
    except jwt.ExpiredSignatureError:
        LOGGER.warning("JWT refresh token expired")
        return None
    except jwt.InvalidTokenError as exc:
        LOGGER.warning("Invalid JWT refresh token: %s", exc)
        return None


def revoke_refresh_token(token: str) -> bool:
    """Revoke a refresh token so it cannot be used again."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        if payload.get("type") != "refresh":
            return False
        store = get_auth_store()
        exp_dt = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        store.revoke_token(
            jti=payload.get("jti", ""),
            expires_at_utc=exp_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        return True
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return False


# ── FastAPI dependencies ────────────────────────────────────────────


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> UserRecord:
    """FastAPI dependency: extract and validate Bearer token, return the user.

    Raises 401 if the token is missing, invalid, or expired.
    Raises 403 if the user account is disabled.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id_str: str | None = payload.get("sub")
    if user_id_str is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
            headers={"WWW-Authenticate": "Bearer"},
        )
    store = get_auth_store()
    user = store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )
    return user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> UserRecord | None:
    """FastAPI dependency: return the authenticated user or None if no valid token."""
    if credentials is None:
        return None
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        return None
    user_id_str = payload.get("sub")
    if user_id_str is None:
        return None
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        return None
    store = get_auth_store()
    user = store.get_user_by_id(user_id)
    if user is None or not user.is_active:
        return None
    return user


def require_role(required_role: str):
    """Factory: return a dependency that enforces a minimum role.

    Role hierarchy: admin > operator > viewer.
    """
    _ROLE_HIERARCHY = {"admin": 3, "operator": 2, "viewer": 1}

    def _role_checker(current_user: UserRecord = Depends(get_current_user)) -> UserRecord:
        user_level = _ROLE_HIERARCHY.get(current_user.role, 0)
        required_level = _ROLE_HIERARCHY.get(required_role, 0)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' or higher required",
            )
        return current_user

    return _role_checker
