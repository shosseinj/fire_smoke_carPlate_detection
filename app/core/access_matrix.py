from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class AccessDefinition:
    application: str
    title: str
    actions: tuple[str, ...]
    scope_types: tuple[str, ...]

GLOBAL = ("global",)
LOCATION = ("global", "building", "section", "camera", "room")

ACCESS_DEFINITIONS = (
    AccessDefinition("auth", "کاربران و دسترسی‌ها", ("manage",), GLOBAL),
    AccessDefinition("buildings", "ساختمان‌ها", ("create", "read", "edit", "delete"), ("global", "building")),
    AccessDefinition("sections", "بخش‌ها", ("create", "read", "edit", "delete"), ("global", "building", "section")),
    AccessDefinition("cameras", "دوربین‌ها", ("create", "read", "edit", "delete"), ("global", "building", "section", "camera")),
    AccessDefinition("rooms", "اتاق‌ها", ("create", "read", "edit", "delete", "assign"), LOCATION),
    AccessDefinition("sources", "منابع", ("create", "read", "edit", "delete", "enable", "assign"), LOCATION),
    AccessDefinition("personnel", "پرسنل", ("create", "read", "edit", "delete", "import", "export"), ("global", "building", "section")),
    AccessDefinition("personnel_images", "تصاویر پرسنل", ("create", "read", "delete"), ("global", "building", "section")),
    AccessDefinition("employee_types", "انواع پرسنل", ("create", "read", "edit", "delete"), GLOBAL),
    AccessDefinition("shifts", "شیفت‌ها", ("create", "read", "edit", "delete", "assign"), GLOBAL),
    AccessDefinition("holidays", "تعطیلات", ("create", "read", "edit", "delete", "import", "export"), GLOBAL),
    AccessDefinition("requests", "درخواست‌ها", ("create", "read", "delete", "approve", "cancel"), GLOBAL),
    AccessDefinition("personnel_requests", "درخواست‌های پرسنلی", ("create", "read", "edit", "delete", "approve", "reject", "generate"), GLOBAL),
    AccessDefinition("car_plates", "پلاک‌ها", ("create", "read", "edit", "delete"), GLOBAL),
    AccessDefinition("plate_settings", "تنظیمات پلاک", ("read", "edit", "delete"), GLOBAL),
    AccessDefinition("plate_logs", "گزارش‌های پلاک", ("create", "read", "edit", "delete", "download"), LOCATION),
    AccessDefinition("fire_smoke", "تشخیص حریق و دود", ("read", "configure"), GLOBAL),
    AccessDefinition("fire_logs", "گزارش‌های حریق", ("create", "read", "edit", "delete", "download"), LOCATION),
    AccessDefinition("detection_logs", "گزارش‌های تشخیص", ("create", "read", "edit", "delete", "import", "export", "download"), LOCATION),
    AccessDefinition("humans", "رهگیری انسان", ("read",), LOCATION),
    AccessDefinition("faces", "تشخیص چهره", ("read", "configure", "enroll", "delete"), GLOBAL),
    AccessDefinition("recordings", "ضبط‌ها", ("create", "read", "cancel", "download", "delete"), LOCATION),
    AccessDefinition("static_videos", "ویدئوهای ثابت", ("create", "read", "edit", "delete", "retry"), GLOBAL),
    AccessDefinition("models", "مدل‌ها", ("read", "configure", "upload", "convert"), GLOBAL),
    AccessDefinition("general_settings", "تنظیمات عمومی", ("read", "edit", "reset"), GLOBAL),
    AccessDefinition("diagnostics", "عیب‌یابی", ("read", "execute"), GLOBAL),
    AccessDefinition("processor_tests", "آزمون پردازنده", ("read", "execute"), GLOBAL),
    AccessDefinition("frames", "پردازش فریم", ("execute",), LOCATION),
    AccessDefinition("extract_frames", "استخراج فریم", ("execute",), LOCATION),
    AccessDefinition("broadcast", "پخش زنده", ("read", "control"), LOCATION),
    AccessDefinition("video_wall", "دیوار ویدیویی", ("read",), LOCATION),
    AccessDefinition("results", "نتایج زنده تشخیص", ("read",), LOCATION),
    AccessDefinition("broadcast_gpu", "پردازش پخش زنده", ("read",), GLOBAL),
    AccessDefinition("import_progress", "وضعیت ورود اطلاعات", ("create", "read", "delete"), GLOBAL),
    AccessDefinition("live_branch", "شاخه پردازش زنده", ("execute",), GLOBAL),
    AccessDefinition("developer", "ابزارهای توسعه", ("read", "execute"), GLOBAL),
)

ACCESS_REGISTRY = {(d.application, action): d for d in ACCESS_DEFINITIONS for action in d.actions}

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
