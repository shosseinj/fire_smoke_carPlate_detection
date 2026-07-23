from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


HolidayType = str


class HolidayCreate(BaseModel):
    name: str = Field(min_length=2, max_length=255)
    date: date
    description: str | None = None
    holiday_type: HolidayType = "national"
    every_year: bool = False
    is_active: bool = True


class HolidayUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=255)
    date: date | None = None
    description: str | None = None
    holiday_type: HolidayType | None = None
    every_year: bool | None = None
    is_active: bool | None = None


class HolidayResponse(BaseModel):
    id: int
    name: str
    date: date
    description: str | None = None
    holiday_type: str
    every_year: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: int | None = None
    updated_by: int | None = None
    created_by_username: str | None = None
    updated_by_username: str | None = None