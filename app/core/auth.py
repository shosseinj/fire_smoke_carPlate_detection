from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.core.auth_store import AuthStore, UserRecord
from app.database import get_database

LOGGER = logging.getLogger("uvicorn.error")

_bearer_scheme = HTTPBearer(auto_error=False)
_auth_store: AuthStore | None = None

ROLE_ALIASES = {
    "superuser": "admin",
    "admin": "admin",
    "operator": "operator",
    "user": "viewer",
    "viewer": "viewer",
}
CANONICAL_ROLES = frozenset({"admin", "operator", "viewer"})


def normalize_role(role: str | None) -> str:
    """Map legacy role vocabulary to the canonical target roles."""
    normalized = ROLE_ALIASES.get((role or "").strip().lower())
    return normalized or (role or "").strip().lower()


def get_auth_store() -> AuthStore:
    global _auth_store
    if _auth_store is None:
        _auth_store = AuthStore(get_database(
            settings.database_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
        ))
        _auth_store.seed_default_admin(
            username=settings.auth_default_admin_username,
            password=settings.auth_default_admin_password,
            role="admin",
            email=settings.auth_default_admin_email,
        )
    return _auth_store


# ── Password utilities ──────────────────────────────────────────────


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) > 72:
        raise ValueError("Password exceeds bcrypt's 72-byte limit")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, stored_hash: str) -> bool:
    try:
        encoded = plain.encode("utf-8")
        if len(encoded) > 72:
            return False
        return bcrypt.checkpw(encoded, stored_hash.encode("utf-8"))
    except (TypeError, ValueError):
        return False


# ── JWT utilities ───────────────────────────────────────────────────


def _make_jwt_payload(
    user_id: int,
    username: str,
    role: str,
    token_type: str,
    expires_minutes: int | None = None,
) -> dict[str, Any]:
    if expires_minutes is None:
        expires_minutes = (
            settings.jwt_expiry_minutes
            if token_type == "access"
            else settings.jwt_refresh_expiry_minutes
        )
    now = datetime.now(timezone.utc)
    return {
        "sub": str(user_id),
        "user_id": user_id,
        "username": username,
        "role": normalize_role(role),
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
        "type": token_type,
        "jti": str(uuid.uuid4()),
    }


def create_access_token(
    user_id: int,
    username: str,
    role: str,
    expires_minutes: int | None = None,
) -> str:
    payload = _make_jwt_payload(user_id, username, role, "access", expires_minutes)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(
    user_id: int,
    username: str,
    role: str,
    expires_minutes: int | None = None,
) -> str:
    payload = _make_jwt_payload(user_id, username, role, "refresh", expires_minutes)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def _decode_token(token: str, expected_type: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        LOGGER.warning("JWT %s token expired", expected_type)
        return None
    except jwt.InvalidTokenError as exc:
        LOGGER.warning("Invalid JWT %s token: %s", expected_type, exc)
        return None

    if payload.get("type") != expected_type:
        return None
    payload["role"] = normalize_role(payload.get("role"))
    return payload


def decode_access_token(token: str) -> dict[str, Any] | None:
    return _decode_token(token, "access")


def token_revocation_id(token: str, payload: dict[str, Any]) -> str:
    """Use jti when present and a stable non-secret fingerprint for legacy tokens."""
    jti = payload.get("jti")
    if isinstance(jti, str) and jti.strip():
        return jti
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def decode_refresh_token(token: str) -> dict[str, Any] | None:
    payload = _decode_token(token, "refresh")
    if payload is None:
        return None
    if get_auth_store().is_token_revoked(token_revocation_id(token, payload)):
        LOGGER.warning("Refresh token has been revoked")
        return None
    return payload


def revoke_refresh_token(token: str) -> bool:
    """Persistently revoke a valid refresh token; repeated calls remain idempotent."""
    payload = _decode_token(token, "refresh")
    if payload is None:
        return False
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    get_auth_store().revoke_token(token_revocation_id(token, payload), expires_at)
    return True


def resolve_user_from_payload(payload: dict[str, Any]) -> UserRecord | None:
    """Resolve current storage identity from current or legacy JWT claim shapes."""
    store = get_auth_store()

    user_id_claim = payload.get("user_id")
    try:
        if user_id_claim not in (None, ""):
            user = store.get_user_by_id(int(user_id_claim))
            if user is not None:
                return user
    except (TypeError, ValueError):
        pass

    subject = payload.get("sub")
    try:
        if subject not in (None, ""):
            user = store.get_user_by_id(int(subject))
            if user is not None:
                return user
    except (TypeError, ValueError):
        pass

    if isinstance(subject, str) and subject:
        return store.get_user_by_username(subject)
    return None


# ── FastAPI dependencies ────────────────────────────────────────────


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> UserRecord:
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
    user = resolve_user_from_payload(payload)
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
    if credentials is None:
        return None
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        return None
    user = resolve_user_from_payload(payload)
    if user is None or not user.is_active:
        return None
    return user


def require_role(required_role: str):
    """Return a dependency enforcing the canonical admin > operator > viewer hierarchy."""
    hierarchy = {"admin": 3, "operator": 2, "viewer": 1}
    canonical_required = normalize_role(required_role)

    def _role_checker(current_user: UserRecord = Depends(get_current_user)) -> UserRecord:
        user_level = hierarchy.get(normalize_role(current_user.role), 0)
        required_level = hierarchy.get(canonical_required, 0)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{canonical_required}' or higher required",
            )
        return current_user

    return _role_checker
