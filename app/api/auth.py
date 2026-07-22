from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Query, Response, status

from app.api.auth_schemas import (
    CreateUserRequest,
    CreateUserResponse,
    LegacyPasswordChangeRequest,
    LegacyRoleChangeResponse,
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
    TokenResponse,
    UserResponse,
    UserUpdateRequest,
    VALID_ROLES,
    validate_legacy_password_strength,
)
from app.config import settings
from app.database import IntegrityError
from app.core.auth import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    get_auth_store,
    get_current_user,
    get_optional_user,
    hash_password,
    normalize_role,
    resolve_user_from_payload,
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
        role=normalize_role(user.role),
        is_active=user.is_active,
        created_at_utc=user.created_at_utc,
        created_at=user.created_at_utc,
        email=user.email,
        full_name=user.full_name,
        last_login_utc=user.last_login_utc,
        last_login=user.last_login_utc,
    )


def _create_user_response(user: UserRecord, message: str | None = None) -> CreateUserResponse:
    return CreateUserResponse(
        id=user.id,
        user_id=user.id,
        username=user.username,
        role=normalize_role(user.role),
        is_active=user.is_active,
        created_at_utc=user.created_at_utc,
        email=user.email,
        full_name=user.full_name,
        message=message,
    )


def _build_token_response(user: UserRecord) -> TokenExchangeResponse:
    role = normalize_role(user.role)
    return TokenExchangeResponse(
        access_token=create_access_token(user.id, user.username, role),
        refresh_token=create_refresh_token(user.id, user.username, role),
        token_type="bearer",
        expires_in=settings.jwt_expiry_minutes * 60,
        user_id=user.id,
        username=user.username,
        role=role,
    )


def _require_admin(user: UserRecord) -> None:
    if normalize_role(user.role) != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="نام کاربری یا رمز عبور نادرست است",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _extract_refresh_token(
    body: RefreshRequest | LogoutRequest | None,
    query_token: str | None,
) -> str:
    body_token = body.refresh_token if body is not None else None
    if body_token and query_token and body_token != query_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conflicting refresh tokens were supplied",
        )
    token = query_token or body_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="refresh_token is required in the query or JSON body",
        )
    return token


def _is_legacy_creation(payload: CreateUserRequest) -> bool:
    return bool({"confirm_password", "full_name"} & payload.model_fields_set)


def _validate_legacy_creation(payload: CreateUserRequest) -> None:
    if not payload.email:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ایمیل برای درخواست قدیمی الزامی است",
        )
    if payload.confirm_password is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="تایید رمز عبور الزامی است",
        )
    if payload.confirm_password != payload.password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="رمز عبور و تایید رمز عبور یکسان نیست",
        )
    errors = validate_legacy_password_strength(payload.password)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="; ".join(errors),
        )


def _create_user_record(
    payload: CreateUserRequest,
    role: str,
    *,
    is_active: bool,
) -> UserRecord:
    store = get_auth_store()
    if store.get_user_by_username(payload.username) is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{payload.username}' already exists",
        )
    if payload.email and store.get_user_by_email(payload.email) is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email '{payload.email}' already exists",
        )

    try:
        return store.create_user(
            username=payload.username,
            password_hash=hash_password(payload.password),
            role=normalize_role(role),
            email=payload.email,
            is_active=is_active,
            full_name=payload.full_name,
        )
    except IntegrityError as exc:
        LOGGER.info("Duplicate user creation rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="نام کاربری یا ایمیل موجود است!",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOGGER.exception("Failed to create user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user",
        ) from exc


# ── Login and token lifecycle ───────────────────────────────────────


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive access and refresh tokens",
    responses={401: {"description": "Invalid credentials or locked account"}},
)
def login(payload: LoginRequest) -> TokenResponse:
    store = get_auth_store()
    user = store.get_user_by_username(payload.username)
    if user is None or not user.is_active:
        raise _invalid_credentials()

    now = datetime.now(timezone.utc)
    locked_until = _parse_utc(user.locked_until_utc)
    if locked_until is not None and locked_until > now:
        remaining = max(1, math.ceil((locked_until - now).total_seconds() / 60))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"حساب کاربری قفل شده است. {remaining} دقیقه دیگر تلاش کنید",
        )

    if not verify_password(payload.password, user.password_hash):
        lock_until = now + timedelta(minutes=settings.auth_login_lockout_minutes)
        store.record_failed_login(
            user.id,
            max_attempts=settings.auth_login_max_attempts,
            locked_until_utc=_utc_text(lock_until),
        )
        raise _invalid_credentials()

    updated = store.record_successful_login(user.id, _utc_text(now))
    if updated is None:
        raise _invalid_credentials()
    return TokenResponse(**_build_token_response(updated).model_dump())


@router.post(
    "/token",
    response_model=TokenExchangeResponse,
    summary="OAuth2-compatible token endpoint",
    description=(
        "OAuth2 password-form authentication. For compatibility, this endpoint does "
        "not mutate /login lockout counters or last_login_utc."
    ),
)
def token_endpoint(
    grant_type: str | None = Form(default="password"),
    username: str = Form(min_length=1),
    password: str = Form(min_length=1),
    scope: str = Form(default=""),
    client_id: str | None = Form(default=None),
    client_secret: str | None = Form(default=None),
) -> TokenExchangeResponse:
    del grant_type, scope, client_id, client_secret
    user = get_auth_store().verify_credentials(username, password)
    if user is None:
        raise _invalid_credentials()
    return _build_token_response(user)


@router.post(
    "/refresh",
    response_model=RefreshResponse,
    summary="Rotate a refresh token",
    description="Accepts refresh_token in either the JSON body or query string.",
)
def refresh(
    payload: RefreshRequest | None = Body(default=None),
    refresh_token: str | None = Query(default=None),
) -> RefreshResponse:
    token = _extract_refresh_token(payload, refresh_token)
    refresh_payload = decode_refresh_token(token)
    if refresh_payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked refresh token",
        )

    user = resolve_user_from_payload(refresh_payload)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )

    if not revoke_refresh_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    return RefreshResponse(**_build_token_response(user).model_dump())


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Revoke a refresh token",
    description=(
        "Current clients may send JSON without an access token. Legacy query mode "
        "requires a valid Bearer access token."
    ),
)
def logout(
    payload: LogoutRequest | None = Body(default=None),
    refresh_token: str | None = Query(default=None),
    current_user: UserRecord | None = Depends(get_optional_user),
) -> LogoutResponse:
    token = _extract_refresh_token(payload, refresh_token)
    legacy_mode = refresh_token is not None
    if legacy_mode and current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not revoke_refresh_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    return LogoutResponse(
        message="خروج با موفقیت انجام شد" if legacy_mode else "Logged out successfully"
    )


# ── User creation ───────────────────────────────────────────────────


@router.post(
    "/create-admin",
    response_model=CreateUserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "Legacy-compatible admin creation"}},
)
def create_admin(
    payload: CreateUserRequest,
    response: Response,
    current_user: UserRecord = Depends(get_current_user),
) -> CreateUserResponse:
    _require_admin(current_user)
    legacy_mode = _is_legacy_creation(payload)
    if legacy_mode:
        _validate_legacy_creation(payload)
        response.status_code = status.HTTP_200_OK
    user = _create_user_record(
        payload,
        role="admin",
        is_active=True if legacy_mode else payload.is_active,
    )
    return _create_user_response(user, "کاربر ادمین با موفقیت ساخته شد.")


@router.post(
    "/create-user",
    response_model=CreateUserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "Legacy-compatible regular-user creation"}},
)
def create_user(
    payload: CreateUserRequest,
    response: Response,
    current_user: UserRecord = Depends(get_current_user),
) -> CreateUserResponse:
    caller_role = normalize_role(current_user.role)
    if caller_role not in {"admin", "operator"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or operator role required",
        )

    legacy_mode = _is_legacy_creation(payload)
    if legacy_mode:
        _require_admin(current_user)
        _validate_legacy_creation(payload)
        response.status_code = status.HTTP_200_OK
        requested_role = "viewer"
        is_active = True
    else:
        requested_role = normalize_role(payload.role)
        if requested_role not in VALID_ROLES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid role. Must be one of: {', '.join(sorted(VALID_ROLES))}",
            )
        if caller_role == "operator" and requested_role in {"admin", "operator"}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operators cannot create admin or operator users",
            )
        is_active = payload.is_active

    user = _create_user_record(
        payload,
        role=requested_role,
        is_active=is_active,
    )
    return _create_user_response(user, "کاربر با موفقیت ساخته شد.")


# ── Current user and password changes ───────────────────────────────


@router.get("/me", response_model=UserResponse)
def me(current_user: UserRecord = Depends(get_current_user)) -> UserResponse:
    return _user_to_response(current_user)


def _change_password(
    current_user: UserRecord,
    current_password: str,
    new_password: str,
    *,
    reject_reuse: bool,
) -> None:
    if not verify_password(current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    if reject_reuse and verify_password(new_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password",
        )
    try:
        new_hash = hash_password(new_password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if get_auth_store().update_password(current_user.id, new_hash) is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password",
        )


@router.post("/me/password", response_model=PasswordChangeResponse)
def change_password(
    payload: PasswordChangeRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> PasswordChangeResponse:
    _change_password(
        current_user,
        payload.current_password,
        payload.new_password,
        reject_reuse=True,
    )
    LOGGER.info("Password changed for user %s (id=%d)", current_user.username, current_user.id)
    return PasswordChangeResponse(message="Password changed successfully")


@router.put(
    "/me/password",
    response_model=PasswordChangeResponse,
    deprecated=True,
    summary="Legacy-compatible password change",
)
def change_password_legacy(
    payload: LegacyPasswordChangeRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> PasswordChangeResponse:
    _change_password(
        current_user,
        payload.old_password,
        payload.new_password,
        reject_reuse=False,
    )
    return PasswordChangeResponse(message="رمز عبور با موفقیت تغییر کرد")


# ── User management ─────────────────────────────────────────────────


@router.get(
    "/users",
    response_model=PaginatedUserResponse | list[UserResponse],
    summary="List users in current or legacy pagination format",
)
def list_users(
    offset: int | None = Query(default=None, ge=0),
    skip: int | None = Query(default=None, ge=0),
    limit: int | None = Query(default=None, ge=1, le=200),
    current_user: UserRecord = Depends(get_current_user),
) -> PaginatedUserResponse | list[UserResponse]:
    _require_admin(current_user)
    if offset is not None and skip is not None and offset != skip:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="skip and offset must match when both are supplied",
        )

    current_mode = offset is not None
    effective_offset = offset if current_mode else (skip or 0)
    effective_limit = limit if limit is not None else (50 if current_mode else 100)
    users, total = get_auth_store().list_users(effective_offset, effective_limit)
    serialized = [_user_to_response(user) for user in users]
    if not current_mode:
        return serialized
    return PaginatedUserResponse(
        users=serialized,
        total=total,
        offset=effective_offset,
        limit=effective_limit,
    )


@router.get("/users/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    current_user: UserRecord = Depends(get_current_user),
) -> UserResponse:
    _require_admin(current_user)
    user = get_auth_store().get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_to_response(user)


@router.put("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    payload: UserUpdateRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> UserResponse:
    _require_admin(current_user)
    store = get_auth_store()
    user = store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.username is not None and payload.username != user.username:
        if store.get_user_by_username(payload.username) is not None:
            raise HTTPException(status_code=400, detail="Username already exists")
    if payload.email is not None and payload.email != user.email:
        if store.get_user_by_email(payload.email) is not None:
            raise HTTPException(status_code=400, detail="Email already exists")
    if payload.is_active is False and normalize_role(user.role) == "admin" and user.is_active:
        if store.count_active_admins() <= 1:
            raise HTTPException(status_code=400, detail="Cannot deactivate the last active admin")

    updated = store.update_user(
        user_id=user_id,
        username=payload.username,
        email=payload.email,
        is_active=payload.is_active,
        full_name=payload.full_name,
        update_full_name="full_name" in payload.model_fields_set,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_to_response(updated)


def _change_user_role(user_id: int, requested_role: str) -> UserRecord:
    store = get_auth_store()
    role = normalize_role(requested_role)
    if role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid role. Must be one of: {', '.join(sorted(VALID_ROLES))}",
        )
    user = store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if normalize_role(user.role) == "admin" and role != "admin" and user.is_active:
        if store.count_active_admins() <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot change the role of the last active admin",
            )
    updated = store.set_user_role(user_id, role)
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found")
    return updated


@router.patch("/users/{user_id}/role", response_model=UserResponse)
def change_role(
    user_id: int,
    payload: RoleChangeRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> UserResponse:
    _require_admin(current_user)
    return _user_to_response(_change_user_role(user_id, payload.role))


@router.put(
    "/users/{user_id}/role",
    response_model=LegacyRoleChangeResponse,
    deprecated=True,
    summary="Legacy-compatible role change",
)
def change_role_legacy(
    user_id: int,
    role: str = Query(...),
    current_user: UserRecord = Depends(get_current_user),
) -> LegacyRoleChangeResponse:
    _require_admin(current_user)
    if role not in {"user", "admin"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="نقش کاربر باید یکی از این موارد باشد: ['user', 'admin']",
        )
    updated = _change_user_role(user_id, "viewer" if role == "user" else "admin")
    public = _user_to_response(updated)
    return LegacyRoleChangeResponse(
        **public.model_dump(),
        user_id=updated.id,
        message=f"نقش کاربر به {normalize_role(updated.role)} تغییر کرد",
    )


@router.delete("/users/{user_id}", response_model=LogoutResponse)
def delete_user(
    user_id: int,
    current_user: UserRecord = Depends(get_current_user),
) -> LogoutResponse:
    _require_admin(current_user)
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="امکان حذف حساب کاربری خودتان وجود ندارد",
        )
    store = get_auth_store()
    user = store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="کاربر یافت نشد")
    if normalize_role(user.role) == "admin" and user.is_active:
        if store.count_active_admins() <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete the last active admin",
            )
    if not store.delete_user(user_id):
        raise HTTPException(status_code=404, detail="کاربر یافت نشد")
    return LogoutResponse(message="کاربر با موفقیت حذف شد")
