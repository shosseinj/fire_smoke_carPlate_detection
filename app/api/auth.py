from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Query, status

from app.config import settings
from app.api.auth_schemas import (
    CreateUserRequest,
    CreateUserResponse,
    LoginRequest,
    LogoutRequest,
    LogoutResponse,
    PaginatedUserResponse,
    PasswordChangeRequest,
    PasswordChangeResponse,
    RefreshRequest,
    RefreshResponse,
    RoleChangeRequest,
    TokenExchangeResponse,
    TokenRequest,
    TokenResponse,
    UserResponse,
    UserUpdateRequest,
    VALID_ROLES,
)
from app.core.auth import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    get_auth_store,
    get_current_user,
    hash_password,
    revoke_refresh_token,
    verify_password,
)
from app.core.auth_store import UserRecord

LOGGER = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


def _user_to_response(user: UserRecord) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        created_at_utc=user.created_at_utc,
        email=user.email,
    )


def _build_token_response(
    user_id: int,
    username: str,
    role: str,
    include_refresh: bool = False,
) -> TokenExchangeResponse:
    access = create_access_token(user_id=user_id, username=username, role=role)
    refresh = None
    if include_refresh:
        refresh = create_refresh_token(user_id=user_id, username=username, role=role)
    return TokenExchangeResponse(
        access_token=access,
        token_type="bearer",
        expires_in=settings.jwt_expiry_minutes * 60,
        refresh_token=refresh,
        role=role,
    )


# ── Step 1: Login (existing, extended with refresh) ─────────────────


@router.post(
    "/login",
    summary="Authenticate and receive a JWT token",
    description=(
        "Accepts username and password credentials. Returns a Bearer JWT access token "
        "on success. The token includes user_id, username, and role for downstream "
        "authorisation decisions. Use the token in the Authorization header for "
        "protected endpoints."
    ),
    responses={
        200: {"description": "Login successful", "model": TokenResponse},
        401: {"description": "Invalid credentials"},
        422: {"description": "Validation error"},
    },
)
def login(payload: LoginRequest) -> TokenResponse:
    store = get_auth_store()
    user = store.verify_credentials(payload.username, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
    )
    return TokenResponse(access_token=token, token_type="bearer", role=user.role)


# ── Step 2: Token lifecycle ─────────────────────────────────────────


@router.post(
    "/token",
    summary="OAuth2-compatible token endpoint",
    description=(
        "Accepts username and password as form data (OAuth2 password grant). "
        "Returns access_token, refresh_token, token_type, expires_in, and role. "
        "This endpoint is functionally equivalent to /login but returns a refresh "
        "token and follows the OAuth2 password grant specification."
    ),
    responses={
        200: {"description": "Token issued", "model": TokenExchangeResponse},
        401: {"description": "Invalid credentials"},
        422: {"description": "Validation error"},
    },
)
def token_endpoint(
    grant_type: str | None = Form(default="password"),
    username: str = Form(min_length=1),
    password: str = Form(min_length=1),
    scope: str = Form(default=""),
    client_id: str | None = Form(default=None),
    client_secret: str | None = Form(default=None),
) -> TokenExchangeResponse:
    store = get_auth_store()
    user = store.verify_credentials(username, password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    return _build_token_response(
        user_id=user.id,
        username=user.username,
        role=user.role,
        include_refresh=True,
    )


@router.post(
    "/refresh",
    summary="Refresh an access token",
    description=(
        "Accepts a valid refresh token and returns a new access token. "
        "Refresh tokens are single-use: the provided refresh token is revoked "
        "and a new one is issued alongside the access token."
    ),
    responses={
        200: {"description": "Token refreshed", "model": RefreshResponse},
        401: {"description": "Invalid, expired, or revoked refresh token"},
        422: {"description": "Validation error"},
    },
)
def refresh(payload: RefreshRequest) -> RefreshResponse:
    refresh_payload = decode_refresh_token(payload.refresh_token)
    if refresh_payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked refresh token",
        )

    store = get_auth_store()
    user_id = int(refresh_payload["sub"])
    user = store.get_user_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )

    # Revoke the old refresh token (single-use)
    revoke_refresh_token(payload.refresh_token)

    # Issue new access and refresh tokens
    access = create_access_token(
        user_id=user.id, username=user.username, role=user.role,
    )
    new_refresh = create_refresh_token(
        user_id=user.id, username=user.username, role=user.role,
    )
    return RefreshResponse(
        access_token=access,
        token_type="bearer",
        expires_in=settings.jwt_expiry_minutes * 60,
        role=user.role,
    )


@router.post(
    "/logout",
    summary="Logout and revoke refresh token",
    description=(
        "Revokes the provided refresh token so it cannot be used again. "
        "Access tokens are short-lived and not stored server-side; they expire "
        "naturally and do not need explicit revocation."
    ),
    responses={
        200: {"description": "Logged out", "model": LogoutResponse},
        401: {"description": "Invalid refresh token"},
        422: {"description": "Validation error"},
    },
)
def logout(payload: LogoutRequest) -> LogoutResponse:
    revoked = revoke_refresh_token(payload.refresh_token)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    return LogoutResponse(message="Logged out successfully")


# ── Step 3: User creation ───────────────────────────────────────────


@router.post(
    "/create-admin",
    summary="Create a new administrator user",
    description=(
        "Creates a new user with the admin role. Requires authentication as an "
        "existing admin. The request body must include username, password, "
        "and optional email."
    ),
    responses={
        201: {"description": "Admin created", "model": CreateUserResponse},
        400: {"description": "Duplicate username or email"},
        401: {"description": "Not authenticated"},
        403: {"description": "Requires admin role"},
        422: {"description": "Validation error"},
    },
    status_code=status.HTTP_201_CREATED,
)
def create_admin(
    payload: CreateUserRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> CreateUserResponse:
    _require_admin(current_user)
    return _create_user(payload, "admin")


@router.post(
    "/create-user",
    summary="Create a new user",
    description=(
        "Creates a new user with the specified role. Requires authentication as "
        "an admin or operator. Operators can only create users with viewer or "
        "operator roles (cannot create other operators or admins)."
    ),
    responses={
        201: {"description": "User created", "model": CreateUserResponse},
        400: {"description": "Duplicate username, email, or invalid role"},
        401: {"description": "Not authenticated"},
        403: {"description": "Requires admin or operator role"},
        422: {"description": "Validation error"},
    },
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    payload: CreateUserRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> CreateUserResponse:
    if current_user.role == "operator" and payload.role in ("admin", "operator"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operators cannot create admin or operator users",
        )
    return _create_user(payload, payload.role)


def _require_admin(user: UserRecord) -> None:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )


def _create_user(payload: CreateUserRequest, role: str) -> CreateUserResponse:
    store = get_auth_store()

    if store.get_user_by_username(payload.username) is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{payload.username}' already exists",
        )

    if payload.email is not None and store.get_user_by_email(payload.email) is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email '{payload.email}' already exists",
        )

    password_hash = hash_password(payload.password)
    try:
        user = store.create_user(
            username=payload.username,
            password_hash=password_hash,
            role=role,
            email=payload.email,
            is_active=payload.is_active,
        )
    except Exception as exc:
        LOGGER.error("Failed to create user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user",
        )

    return CreateUserResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        created_at_utc=user.created_at_utc,
        email=user.email,
    )


# ── Step 4: User management ─────────────────────────────────────────


@router.get(
    "/users",
    summary="List all users",
    description=(
        "Returns a paginated list of all users. Requires admin role. "
        "Supports offset and limit query parameters for pagination."
    ),
    responses={
        200: {"description": "User list", "model": PaginatedUserResponse},
        401: {"description": "Not authenticated"},
        403: {"description": "Requires admin role"},
    },
)
def list_users(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: UserRecord = Depends(get_current_user),
) -> PaginatedUserResponse:
    _require_admin(current_user)
    store = get_auth_store()
    users, total = store.list_users(offset=offset, limit=limit)
    return PaginatedUserResponse(
        users=[_user_to_response(u) for u in users],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/users/{user_id}",
    summary="Get user details",
    description="Returns the public profile of a specific user. Requires admin role.",
    responses={
        200: {"description": "User details", "model": UserResponse},
        401: {"description": "Not authenticated"},
        403: {"description": "Requires admin role"},
        404: {"description": "User not found"},
    },
)
def get_user(
    user_id: int,
    current_user: UserRecord = Depends(get_current_user),
) -> UserResponse:
    _require_admin(current_user)
    store = get_auth_store()
    user = store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return _user_to_response(user)


@router.put(
    "/users/{user_id}",
    summary="Update a user",
    description=(
        "Updates user fields (username, email, is_active). Requires admin role. "
        "Cannot demote the last active admin."
    ),
    responses={
        200: {"description": "User updated", "model": UserResponse},
        400: {"description": "Duplicate username or last-admin protection"},
        401: {"description": "Not authenticated"},
        403: {"description": "Requires admin role"},
        404: {"description": "User not found"},
        422: {"description": "Validation error"},
    },
)
def update_user(
    user_id: int,
    payload: UserUpdateRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> UserResponse:
    _require_admin(current_user)
    store = get_auth_store()

    user = store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if payload.username is not None and payload.username != user.username:
        existing = store.get_user_by_username(payload.username)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Username '{payload.username}' already exists",
            )

    if payload.email is not None and payload.email != user.email:
        if store.get_user_by_email(payload.email) is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email '{payload.email}' already exists",
            )

    # Prevent deactivating the last active admin
    if payload.is_active is False and user.role == "admin" and user.is_active:
        if store.count_active_admins() <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate the last active admin",
            )

    updated = store.update_user(
        user_id=user_id,
        username=payload.username,
        email=payload.email,
        is_active=payload.is_active,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return _user_to_response(updated)


@router.patch(
    "/users/{user_id}/role",
    summary="Change a user's role",
    description=(
        "Changes the role of a specific user. Requires admin role. "
        "Cannot demote the last active admin to a non-admin role."
    ),
    responses={
        200: {"description": "Role changed", "model": UserResponse},
        400: {"description": "Last-admin protection or invalid role"},
        401: {"description": "Not authenticated"},
        403: {"description": "Requires admin role"},
        404: {"description": "User not found"},
        422: {"description": "Validation error"},
    },
)
def change_role(
    user_id: int,
    payload: RoleChangeRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> UserResponse:
    _require_admin(current_user)
    store = get_auth_store()

    if payload.role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid role. Must be one of: {', '.join(sorted(VALID_ROLES))}",
        )

    user = store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Prevent removing the last admin
    if user.role == "admin" and payload.role != "admin":
        if store.count_active_admins() <= 1 and user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change the role of the last active admin",
            )

    updated = store.set_user_role(user_id=user_id, role=payload.role)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return _user_to_response(updated)


# ── Step 5: Password change ─────────────────────────────────────────


@router.post(
    "/me/password",
    summary="Change current user's password",
    description=(
        "Changes the password for the currently authenticated user. "
        "Requires the current password for verification. The new password "
        "must be at least 8 characters and match the confirmation field. "
        "Previously issued access and refresh tokens remain valid until "
        "they expire naturally; they are not revoked."
    ),
    responses={
        200: {"description": "Password changed", "model": PasswordChangeResponse},
        400: {"description": "Current password is incorrect"},
        401: {"description": "Not authenticated"},
        403: {"description": "Account is disabled"},
        422: {"description": "Validation error"},
    },
)
def change_password(
    payload: PasswordChangeRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> PasswordChangeResponse:
    store = get_auth_store()

    # Verify current password
    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Don't allow reusing the same password
    if verify_password(payload.new_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password",
        )

    new_hash = hash_password(payload.new_password)
    updated = store.update_password(current_user.id, new_hash)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password",
        )

    LOGGER.info("Password changed for user %s (id=%d)", current_user.username, current_user.id)
    return PasswordChangeResponse(message="Password changed successfully")


# ── Existing /me endpoint ───────────────────────────────────────────


@router.get(
    "/me",
    summary="Get current authenticated user profile",
    description=(
        "Returns the profile of the currently authenticated user. "
        "Requires a valid Bearer token in the Authorization header."
    ),
    responses={
        200: {"description": "User profile", "model": UserResponse},
        401: {"description": "Not authenticated or invalid token"},
        403: {"description": "Account disabled"},
    },
)
def me(current_user: UserRecord = Depends(get_current_user)) -> UserResponse:
    return _user_to_response(current_user)
