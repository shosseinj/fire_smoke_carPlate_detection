from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class PlateFormat(str, Enum):
    STANDARD = "standard"
    MOTORCYCLE = "motorcycle"
    DIPLOMATIC = "diplomatic"
    TEMPORARY = "temporary"
    FREE_ZONE = "free_zone"
    HISTORICAL = "historical"


class PlateUsageType(str, Enum):
    PERSONAL = "personal"
    TAXI = "taxi"
    PUBLIC_TRANSPORT = "public_transport"
    GOVERNMENT = "government"
    POLICE = "police"
    MILITARY = "military"
    DIPLOMATIC = "diplomatic"
    TEMPORARY = "temporary"
    FREE_ZONE = "free_zone"
    HISTORICAL = "historical"
    AGRICULTURAL = "agricultural"
    OTHER = "other"


class VehicleType(str, Enum):
    SEDAN = "sedan"
    HATCHBACK = "hatchback"
    SUV = "suv"
    PICKUP = "pickup"
    VAN = "van"
    MINIBUS = "minibus"
    BUS = "bus"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    AGRICULTURAL = "agricultural"
    CONSTRUCTION = "construction"
    OTHER = "other"


class PlateLogDirection(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    UNKNOWN = "unknown"


class PlateLogSourceType(str, Enum):
    CAMERA = "camera"
    MANUAL = "manual"


PERSIAN_PLATE_ALPHABETS = {
    "الف", "ب", "پ", "ت", "ث", "ج", "د", "س", "ص", "ط",
    "ع", "ف", "ق", "ک", "ل", "م", "ن", "و", "ه", "ی", "ز", "ش",
}

PLATE_ALPHABETS_BY_USAGE: dict[str, set[str]] = {
    "personal": {"ب", "ج", "د", "س", "ص", "ط", "ق", "ک", "ل", "م", "ن", "و", "ه", "ی"},
    "taxi": {"ت"},
    "public_transport": {"ع"},
    "government": {"الف"},
    "police": {"پ"},
    "military": {"ث", "ز", "ش", "ف"},
}


_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_ASCII_DIGITS = "0123456789"
_DIGIT_TRANSLATION = str.maketrans(
    _PERSIAN_DIGITS + _ARABIC_DIGITS,
    _ASCII_DIGITS + _ASCII_DIGITS,
)


def normalize_persian_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.strip().replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")


def normalize_owner_phone(value: str) -> str:
    normalized = re.sub(r"[\s\-()]", "", str(value or ""))
    if normalized.startswith("0098"):
        normalized = "+98" + normalized[4:]
    elif normalized.startswith("98"):
        normalized = "+" + normalized
    elif normalized.startswith("0"):
        normalized = "+98" + normalized[1:]
    if not re.fullmatch(r"\+98\d{10}", normalized):
        raise ValueError("شماره تلفن مالک باید یک شماره معتبر ایرانی باشد")
    return normalized


def normalize_plate_full_number(value: str) -> str:
    normalized = normalize_persian_text(str(value or "")) or ""
    normalized = normalized.translate(_DIGIT_TRANSLATION)
    normalized = re.sub(r"[\s\-_/|]+", "", normalized)
    normalized = normalized.replace("ایران", "")
    if not normalized:
        raise ValueError("شماره کامل پلاک الزامی است")
    if len(normalized) > 32:
        raise ValueError("شماره کامل پلاک نمی‌تواند بیشتر از ۳۲ کاراکتر باشد")
    if not re.fullmatch(r"[0-9A-Za-zآ-ی]+", normalized):
        raise ValueError("شماره کامل پلاک شامل کاراکتر نامعتبر است")
    return normalized
