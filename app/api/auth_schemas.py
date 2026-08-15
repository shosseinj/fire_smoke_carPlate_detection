from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=500)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    permissions: list[str] = Field(default_factory=list)


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
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    created_at_jalali: str = ""
    last_login_jalali: str | None = None
    permissions: list[str] = Field(default_factory=list)


class AuthError(BaseModel):
    detail: str


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(
        min_length=3,
        max_length=200,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    password: str = Field(min_length=8, max_length=72)
    email: Optional[str] = None
    confirm_password: str
    full_name: Optional[str] = None


class CreateUserResponse(BaseModel):
    message: str
    user_id: int
    username: str


class DuplicateError(BaseModel):
    detail: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=500)
    new_password: str = Field(min_length=8, max_length=72)
    confirm_password: str = Field(min_length=1, max_length=72)

    @field_validator("new_password")
    @classmethod
    def strong_password(cls, value: str) -> str:
        errors = validate_password_strength(value)
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


class UserPermissionGrant(BaseModel):
    application: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_-]*$")
    action: str = Field(min_length=1, max_length=50, pattern=r"^[a-z][a-z0-9_-]*$")
    scope_type: Literal["global", "building", "section", "camera", "room"]
    scope_id: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_scope_id(self):
        if self.scope_type == "global" and self.scope_id != 0:
            raise ValueError("مجوز سراسری باید شناسه محدوده صفر داشته باشد")
        if self.scope_type != "global" and self.scope_id < 1:
            raise ValueError("شناسه محدوده منبع باید عددی مثبت باشد")
        return self


class SetUserAccessRequest(BaseModel):
    grants: list[UserPermissionGrant]


class UserAccessResponse(BaseModel):
    user_id: int
    grants: list[UserPermissionGrant]


class AccessDefinitionResponse(BaseModel):
    application: str
    title: str
    actions: list[str]
    scope_types: list[Literal["global", "building", "section", "camera", "room"]]


class WebSocketTicketRequest(BaseModel):
    application: Literal["broadcast", "video_wall", "results"]
    scope_type: Literal["global", "building", "section", "camera", "room"] = "global"
    scope_id: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_scope(self):
        if self.scope_type == "global" and self.scope_id != 0:
            raise ValueError("محدوده سراسری باید شناسه صفر داشته باشد")
        if self.scope_type != "global" and self.scope_id < 1:
            raise ValueError("محدوده منبع باید شناسه مثبت داشته باشد")
        return self


class WebSocketTicketResponse(BaseModel):
    ticket: str
    expires_in: int


def validate_password_strength(password: str) -> list[str]:
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
