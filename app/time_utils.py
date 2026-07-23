from __future__ import annotations
from datetime import datetime, timezone

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_now_text() -> str:
    return utc_now().isoformat().replace('+00:00', 'Z')

def ensure_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
