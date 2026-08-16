from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials

from app.api.auth_schemas import (
    AccessDefinitionResponse,
    ChangePasswordRequest,
    CreateUserRequest,
    CreateUserResponse,
    LoginRequest,
    LogoutRequest,
    LogoutResponse,
    PasswordChangeResponse,
    RefreshRequest,
    SetUserAccessRequest,
    TokenResponse,
    UserPermissionGrant,
    WebSocketTicketRequest,
    WebSocketTicketResponse,
    UserAccessResponse,
    UserResponse,
    validate_password_strength,
)
from app.config import settings
from app.database import IntegrityError
from app.core.auth import (
    _bearer_scheme,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    effective_permissions,
    get_auth_store,
    get_current_user,
    has_scoped_permission,
    hash_password,
    require_permission,
    resolve_user_from_payload,
    revoke_access_token,
    revoke_refresh_token,
    token_matches_user_version,
    verify_password,
)
from app.core.access_matrix import ACCESS_DEFINITIONS
from app.core.auth_store import UserPermissionGrantRecord, UserRecord

LOGGER = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


def _user_to_response(user: UserRecord) -> UserResponse:
    from app.core.jalali_utils import utc_iso_to_jalali_datetime
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        created_at_jalali=utc_iso_to_jalali_datetime(user.created_at_utc) or "",
        last_login_jalali=utc_iso_to_jalali_datetime(user.last_login_utc),
        permissions=sorted(effective_permissions(user)),
    )


def _create_user_response(user: UserRecord, message: str) -> CreateUserResponse:
    return CreateUserResponse(
        message=message,
        user_id=user.id,
        username=user.username,
    )


def _build_token_response(user: UserRecord) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id, user.username, auth_version=user.auth_version),
        refresh_token=create_refresh_token(user.id, user.username, auth_version=user.auth_version),
        token_type="bearer",
        user_id=user.id,
        username=user.username,
        permissions=sorted(effective_permissions(user)),
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
            detail="توکن‌های نوسازی ارسال‌شده با یکدیگر مغایرت دارند",
        )
    token = query_token or body_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="توکن نوسازی باید در پرس‌وجو یا بدنه JSON ارسال شود",
        )
    return token


def _validate_user_creation(payload: CreateUserRequest) -> None:
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
    errors = validate_password_strength(payload.password)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="; ".join(errors),
        )


def _create_user_record(
    payload: CreateUserRequest,
    *,
    is_active: bool,
) -> UserRecord:
    store = get_auth_store()
    if store.get_user_by_username(payload.username) is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="این نام کاربری قبلاً استفاده شده است",
        )
    if payload.email and store.get_user_by_email(payload.email) is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="این ایمیل قبلاً استفاده شده است",
        )

    try:
        return store.create_user(
            username=payload.username,
            password_hash=hash_password(payload.password),
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
            detail="ایجاد کاربر با خطا مواجه شد",
        ) from exc


# ── Login and token lifecycle ───────────────────────────────────────


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive access and refresh tokens",
    responses={401: {"description": "Invalid credentials or locked account"}},
)
def login(payload: LoginRequest= Body(
        example={
            "username": "superadmin",
            "password": "SuperAdmin123!"
        }
    ),) -> TokenResponse:
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
    return _build_token_response(updated)


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="OAuth2-compatible token endpoint",
    description="OAuth2 password-form authentication.",
)
def token_endpoint(
    grant_type: str | None = Form(default="password"),
    username: str = Form(min_length=1),
    password: str = Form(min_length=1),
    scope: str = Form(default=""),
    client_id: str | None = Form(default=None),
    client_secret: str | None = Form(default=None),
) -> TokenResponse:
    del grant_type, scope, client_id, client_secret
    user = get_auth_store().verify_credentials(username, password)
    if user is None:
        raise _invalid_credentials()
    return _build_token_response(user)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate a refresh token",
    description="Accepts refresh_token in either the JSON body or query string.",
)
def refresh(
    payload: RefreshRequest | None = Body(default=None),
    refresh_token: str | None = Query(default=None),
) -> TokenResponse:
    token = _extract_refresh_token(payload, refresh_token)
    refresh_payload = decode_refresh_token(token)
    if refresh_payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="توکن نوسازی نامعتبر، منقضی یا لغوشده است",
        )

    user = resolve_user_from_payload(refresh_payload)
    if user is None or not user.is_active or not token_matches_user_version(refresh_payload, user):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="کاربر یافت نشد یا حساب کاربری غیرفعال است",
        )

    if not revoke_refresh_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="توکن نوسازی نامعتبر است",
        )
    return _build_token_response(user)


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Logout and revoke the current access token",
    description="No request body needed. Sends the Bearer token (Authorization header) to revoke it.",
)
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> LogoutResponse:
    if credentials is not None:
        revoke_access_token(credentials.credentials)
    return LogoutResponse(message="خروج با موفقیت انجام شد")


# ── User creation ───────────────────────────────────────────────────


@router.post(
    "/users",
    response_model=CreateUserResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    payload: CreateUserRequest,
    current_user: UserRecord = Depends(require_permission("auth.manage")),
) -> CreateUserResponse:
    _validate_user_creation(payload)
    user = _create_user_record(
        payload,
        is_active=True,
    )
    return _create_user_response(user, "کاربر با موفقیت ساخته شد.")


@router.get("/access/definitions", response_model=list[AccessDefinitionResponse])
def access_definitions(_: UserRecord = Depends(require_permission("auth.manage"))) -> list[AccessDefinitionResponse]:
    return [AccessDefinitionResponse(application=item.application, title=item.title, actions=list(item.actions), scope_types=list(item.scope_types)) for item in ACCESS_DEFINITIONS]


@router.post("/websocket-ticket", response_model=WebSocketTicketResponse)
def create_websocket_ticket(
    payload: WebSocketTicketRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> WebSocketTicketResponse:
    permission = f"{payload.application}.read"
    if not has_scoped_permission(current_user, permission, payload.scope_type, payload.scope_id):
        raise HTTPException(status_code=403, detail="برای اتصال به این جریان زنده دسترسی ندارید")
    ticket, expires_in = get_auth_store().issue_websocket_ticket(
        current_user.id, payload.application, payload.scope_type, payload.scope_id
    )
    return WebSocketTicketResponse(ticket=ticket, expires_in=expires_in)


@router.get("/users/{user_id}/access", response_model=UserAccessResponse)
def get_user_access(
    user_id: int,
    _: UserRecord = Depends(require_permission("auth.manage")),
) -> UserAccessResponse:
    store = get_auth_store()
    user = store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="کاربر یافت نشد")
    grants = store.list_user_grants(user.id) or []
    return UserAccessResponse(user_id=user.id, grants=[UserPermissionGrant(application=g.application, action=g.action, scope_type=g.scope_type, scope_id=g.scope_id) for g in grants])


@router.put("/users/{user_id}/access", response_model=UserAccessResponse)
def replace_user_access(
    user_id: int,
    payload: SetUserAccessRequest,
    current_user: UserRecord = Depends(require_permission("auth.manage")),
) -> UserAccessResponse:
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="امکان تغییر دسترسی‌های حساب خودتان وجود ندارد")
    store = get_auth_store()
    target = store.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="کاربر یافت نشد")
    if target.username == settings.auth_default_admin_username:
        raise HTTPException(status_code=403, detail="دسترسی‌های مدیر اصلی قابل تغییر نیست")
    actor_permissions = effective_permissions(current_user)
    requested = {f"{grant.application}.{grant.action}" for grant in payload.grants}
    if "*" not in actor_permissions and not requested.issubset(actor_permissions):
        raise HTTPException(status_code=403, detail="امکان اعطای دسترسی‌ای که خودتان ندارید وجود ندارد")
    if "*" not in actor_permissions and any(
        not store.has_scoped_permission(
            current_user.id,
            f"{grant.application}.{grant.action}",
            grant.scope_type,
            grant.scope_id,
        )
        for grant in payload.grants
    ):
        raise HTTPException(status_code=403, detail="امکان اعطای دسترسی در محدوده‌ای که خودتان ندارید وجود ندارد")
    records = tuple(UserPermissionGrantRecord(g.application, g.action, g.scope_type, g.scope_id) for g in payload.grants)
    try:
        saved = store.replace_user_grants(user_id, records, current_user.id) or []
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    store.record_audit_event(current_user.id, "user.access_replaced", "user", str(user_id), f"{len(saved)} grants")
    return UserAccessResponse(user_id=user_id, grants=[UserPermissionGrant(application=g.application, action=g.action, scope_type=g.scope_type, scope_id=g.scope_id) for g in saved])


# ── Current user and password changes ───────────────────────────────


@router.get("/me", response_model=UserResponse)
def me(current_user: UserRecord = Depends(get_current_user)) -> UserResponse:
    return _user_to_response(current_user)


def _change_password(
    current_user: UserRecord,
    current_password: str,
    new_password: str,
) -> None:
    if not verify_password(current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="رمز عبور فعلی نادرست است",
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
            detail="تغییر رمز عبور با خطا مواجه شد",
        )


@router.post(
    "/me/password",
    response_model=PasswordChangeResponse,
    summary="Change current user password",
)
def change_password(
    payload: ChangePasswordRequest,
    current_user: UserRecord = Depends(get_current_user),
) -> PasswordChangeResponse:
    _change_password(
        current_user,
        payload.current_password,
        payload.new_password,
    )
    return PasswordChangeResponse(message="رمز عبور با موفقیت تغییر کرد")


# ── User management ─────────────────────────────────────────────────


@router.get(
    "/users",
    response_model=list[UserResponse],
    summary="List all users",
)
def list_users(
    skip: int | None = Query(default=None, ge=0),
    limit: int | None = Query(default=None, ge=1, le=200),
    current_user: UserRecord = Depends(require_permission("auth.manage")),
) -> list[UserResponse]:
    effective_skip = skip or 0
    effective_limit = limit or 100
    users, _ = get_auth_store().list_users(effective_skip, effective_limit)
    return [_user_to_response(user) for user in users]


@router.delete("/users/{user_id}", response_model=LogoutResponse)
def delete_user(
    user_id: int,
    current_user: UserRecord = Depends(require_permission("auth.manage")),
) -> LogoutResponse:
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="امکان حذف حساب کاربری خودتان وجود ندارد",
        )
    store = get_auth_store()
    user = store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="کاربر یافت نشد")
    if user.username == settings.auth_default_admin_username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="حذف حساب سوپرادمین مجاز نیست",
        )
    if not store.delete_user(user_id):
        raise HTTPException(status_code=404, detail="کاربر یافت نشد")
    return LogoutResponse(message="کاربر با موفقیت حذف شد")
