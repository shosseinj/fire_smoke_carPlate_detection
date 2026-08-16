from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from app.core.common_schemas import UserBrief


HolidayType = str


class HolidayCreate(BaseModel):
    name: str = Field(min_length=2, max_length=255)
    date: date
    description: Optional[str] = None
    holiday_type: HolidayType = "national"
    every_year: bool = False
    is_active: bool = True


class HolidayUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=255)
    date: Optional[date] = None
    description: Optional[str] = None
    holiday_type: Optional[HolidayType] = None
    every_year: Optional[bool] = None
    is_active: Optional[bool] = None


class HolidayResponse(BaseModel):
    id: int
    name: str
    date: date
    date_jalali: str = ""
    description: Optional[str] = None
    holiday_type: str
    every_year: bool
    is_active: bool
    created_at_jalali: str = ""
    updated_at_jalali: str | None = None
    created_by: UserBrief | None = None
    updated_by: UserBrief | None = None


class HolidayRangeResponse(BaseModel):
    count: int
    unique_day_count: int
    holidays: list[HolidayResponse]