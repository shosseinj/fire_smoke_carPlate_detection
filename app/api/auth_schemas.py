from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


VALID_ROLES = frozenset({"admin", "operator", "viewer"})
LEGACY_ROLE_VALUES = frozenset({"superuser", "admin", "user"})


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=500)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    role: str


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class LogoutRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class LogoutResponse(BaseModel):
    message: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class AuthError(BaseModel):
    detail: str


class CreateUserRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=200,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    password: str = Field(min_length=8, max_length=72)
    email: str
    confirm_password: str
    full_name: Optional[str] = None


class CreateUserResponse(BaseModel):
    message: str
    user_id: int
    username: str
    role: str


class DuplicateError(BaseModel):
    detail: str


class LegacyRoleChangeResponse(UserResponse):
    user_id: int
    message: str


class LegacyPasswordChangeRequest(BaseModel):
    old_password: str = Field(min_length=1, max_length=500)
    new_password: str = Field(min_length=8, max_length=72)
    confirm_new_password: str = Field(min_length=1, max_length=72)

    @field_validator("new_password")
    @classmethod
    def strong_password(cls, value: str) -> str:
        errors = validate_legacy_password_strength(value)
        if errors:
            raise ValueError("; ".join(errors))
        return value

    @field_validator("confirm_new_password")
    @classmethod
    def passwords_match(cls, value: str, info) -> str:
        if info.data.get("new_password") is not None and value != info.data["new_password"]:
            raise ValueError("رمز عبور جدید و تایید آن یکسان نیست")
        return value


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=500)
    new_password: str = Field(min_length=8, max_length=72)
    confirm_password: str = Field(min_length=1, max_length=72)

    @field_validator("new_password")
    @classmethod
    def strong_password(cls, value: str) -> str:
        errors = validate_legacy_password_strength(value)
        if errors:
            raise ValueError("; ".join(errors))
        return value

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, value: str, info) -> str:
        if info.data.get("new_password") is not None and value != info.data["new_password"]:
            raise ValueError("رمز عبور جدید و تایید آن یکسان نیست")
        return value


class PasswordChangeResponse(BaseModel):
    message: str


def validate_legacy_password_strength(password: str) -> list[str]:
    errors: list[str] = []
    if len(password) < 8:
        errors.append("رمز عبور باید حداقل ۸ کاراکتر باشد")
    if len(password.encode("utf-8")) > 72:
        errors.append("رمز عبور باید حداکثر ۷۲ بایت باشد")
    if not re.search(r"[A-Z]", password):
        errors.append("رمز عبور باید حداقل یک حرف بزرگ داشته باشد")
    if not re.search(r"[a-z]", password):
        errors.append("رمز عبور باید حداقل یک حرف کوچک داشته باشد")
    if not re.search(r"\d", password):
        errors.append("رمز عبور باید حداقل یک عدد داشته باشد")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append("رمز عبور باید حداقل یک کاراکتر ویژه داشته باشد")

    lowered = password.lower()
    for pattern in (
        "password",
        "123456",
        "qwerty",
        "abc123",
        "admin",
        "welcome",
        "letmein",
        "monkey",
    ):
        if pattern in lowered:
            errors.append(f"رمز عبور شامل الگوی ضعیف '{pattern}' است")
            break

    if re.search(r"(.)\1{2,}", password):
        errors.append("رمز عبور شامل کاراکترهای تکراری است")
    return errors