from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# ── Existing schemas ────────────────────────────────────────────────


class LoginRequest(BaseModel):
    """Credentials for login."""

    username: str = Field(min_length=1, max_length=200, description="User login name")
    password: str = Field(min_length=1, max_length=500, description="User password")


class TokenResponse(BaseModel):
    """Successful login response carrying a JWT access token."""

    access_token: str
    token_type: str = "bearer"
    role: str


class UserResponse(BaseModel):
    """Public user profile returned by /api/v1/auth/me."""

    id: int
    username: str
    role: str
    is_active: bool
    created_at_utc: str
    email: str | None = None


class AuthError(BaseModel):
    """Standard error detail returned by auth endpoints."""

    detail: str


# ── Step 2: Token lifecycle schemas ─────────────────────────────────


class TokenRequest(BaseModel):
    """OAuth2-compatible form-login request. Used by /api/v1/auth/token."""

    grant_type: str | None = Field(default=None, pattern="password")
    username: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=500)
    scope: str = ""
    client_id: str | None = None
    client_secret: str | None = None


class TokenExchangeResponse(BaseModel):
    """OAuth2-compatible token response with access and refresh tokens."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str | None = None
    role: str


class RefreshRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str = Field(min_length=1, description="Valid refresh token")


class RefreshResponse(BaseModel):
    """Response carrying a new access token."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str


class LogoutRequest(BaseModel):
    """Logout request carrying a refresh token to revoke."""

    refresh_token: str = Field(min_length=1, description="Refresh token to revoke")


class LogoutResponse(BaseModel):
    """Confirmation of successful logout."""

    message: str = "Logged out successfully"


# ── Step 3: User-creation schemas ───────────────────────────────────


class CreateUserRequest(BaseModel):
    """Create a new user."""

    username: str = Field(
        min_length=3, max_length=200, pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique username (letters, digits, underscores, hyphens)",
    )
    password: str = Field(
        min_length=8, max_length=200,
        description="Strong password with at least 8 characters",
    )
    email: str | None = Field(
        default=None, max_length=300, pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        description="Optional email address",
    )
    role: str = Field(
        default="viewer", pattern=r"^(admin|operator|viewer)$",
        description="User role: admin, operator, or viewer",
    )
    is_active: bool = True


class CreateUserResponse(BaseModel):
    """Public user record returned after creation (no password hash)."""

    id: int
    username: str
    role: str
    is_active: bool
    created_at_utc: str
    email: str | None = None


class DuplicateError(BaseModel):
    """Error response for duplicate username or email."""

    detail: str


# ── Step 4: User-management schemas ─────────────────────────────────


class UserUpdateRequest(BaseModel):
    """Fields that can be updated on a user."""

    username: str | None = Field(
        default=None, min_length=3, max_length=200, pattern=r"^[a-zA-Z0-9_-]+$",
    )
    email: str | None = Field(
        default=None, max_length=300, pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    )
    is_active: bool | None = None


class RoleChangeRequest(BaseModel):
    """Request to change a user's role."""

    role: str = Field(
        pattern=r"^(admin|operator|viewer)$",
        description="New role: admin, operator, or viewer",
    )


class PaginatedUserResponse(BaseModel):
    """Paginated list of users."""

    users: list[UserResponse]
    total: int
    offset: int
    limit: int


# ── Step 5: Password-change schemas ─────────────────────────────────


class PasswordChangeRequest(BaseModel):
    """Change the current user's password."""

    current_password: str = Field(
        min_length=1, max_length=500,
        description="Current password for verification",
    )
    new_password: str = Field(
        min_length=8, max_length=200,
        description="New password (at least 8 characters)",
    )
    confirm_password: str = Field(
        min_length=1, max_length=200,
        description="Confirm new password (must match new_password)",
    )

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("Passwords do not match")
        return v


class PasswordChangeResponse(BaseModel):
    """Confirmation of successful password change."""

    message: str = "Password changed successfully"


# ── Valid roles ─────────────────────────────────────────────────────

VALID_ROLES = frozenset({"admin", "operator", "viewer"})
