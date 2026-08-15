from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AccessDefinition:
    application: str
    title: str
    actions: tuple[str, ...]
    scope_types: tuple[str, ...]


ACCESS_DEFINITIONS = (
    AccessDefinition("application", "سامانه", ("read", "manage", "system"), ("global",)),
    AccessDefinition("auth", "کاربران و دسترسی‌ها", ("manage",), ("global",)),
    AccessDefinition("buildings", "ساختمان‌ها", ("create", "read", "edit", "delete"), ("global", "building")),
    AccessDefinition("sections", "بخش‌ها", ("create", "read", "edit", "delete"), ("global", "building", "section")),
    AccessDefinition("cameras", "دوربین‌ها", ("create", "read", "edit", "delete"), ("global", "building", "section", "camera")),
    AccessDefinition("personnel", "پرسنل", ("create", "read", "edit", "delete"), ("global", "building", "section")),
    AccessDefinition("broadcast", "پخش زنده", ("read",), ("global", "building", "section", "camera")),
    AccessDefinition("video_wall", "دیوار ویدیویی", ("read",), ("global", "building", "section", "camera")),
    AccessDefinition("results", "نتایج زنده تشخیص", ("read",), ("global",)),
)

ACCESS_REGISTRY = {
    (definition.application, action): definition
    for definition in ACCESS_DEFINITIONS
    for action in definition.actions
}


def split_permission(permission: str) -> tuple[str, str]:
    try:
        application, action = permission.strip().lower().split(".", 1)
    except ValueError as exc:
        raise ValueError("مجوز انتخاب‌شده معتبر نیست") from exc
    if (application, action) not in ACCESS_REGISTRY:
        raise ValueError("مجوز انتخاب‌شده معتبر نیست")
    return application, action


def permission_key(application: str, action: str) -> str:
    key = (application.strip().lower(), action.strip().lower())
    if key not in ACCESS_REGISTRY:
        raise ValueError("برنامه یا عملیات انتخاب‌شده معتبر نیست")
    return f"{key[0]}.{key[1]}"
