from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field

from app.core.auth import get_current_user, require_permission
from app.core.auth_store import UserRecord
from app.core.plate_constants import (
    PLATE_ALPHABETS_BY_USAGE,
    PERSIAN_PLATE_ALPHABETS,
    PlateFormat,
    PlateUsageType,
    VehicleType,
    normalize_owner_phone,
    normalize_persian_text,
)
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/car-plates", tags=["car-plates"])


class CarPlateBase(BaseModel):
    left_digits: str = Field(..., pattern=r"^\d{2}$")
    plate_alphabet: str = Field(..., min_length=1, max_length=1)
    right_digits: str = Field(..., pattern=r"^\d{3}$")
    iran_code: str = Field(..., pattern=r"^\d{2}$")
    plate_format: PlateFormat = PlateFormat.STANDARD
    usage_type: PlateUsageType
    vehicle_type: VehicleType
    owner_name: str = Field(..., min_length=2, max_length=200)
    owner_phone: str
    color: Optional[str] = Field(default=None, max_length=50)
    brand: Optional[str] = Field(default=None, max_length=80)
    model: Optional[str] = Field(default=None, max_length=80)
    manufacture_year: Optional[int] = Field(default=None, ge=1300, le=1600)
    description: Optional[str] = None
    is_active: bool = True

    @field_validator("plate_alphabet", mode="before")
    @classmethod
    def normalize_plate_alphabet(cls, value: object) -> object:
        normalized = normalize_persian_text(value)
        if normalized not in PERSIAN_PLATE_ALPHABETS:
            raise ValueError("حرف پلاک باید یکی از حروف فارسی مجاز باشد")
        return normalized

    @field_validator("owner_name", "color", "brand", "model", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: object) -> object:
        return normalize_persian_text(value)

    @field_validator("owner_phone", mode="before")
    @classmethod
    def normalize_phone(cls, value: object) -> object:
        return normalize_owner_phone(value)

    @model_validator(mode="after")
    def validate_alphabet_for_usage(self) -> CarPlateBase:
        allowed = PLATE_ALPHABETS_BY_USAGE.get(self.usage_type.value)
        if allowed and self.plate_alphabet not in allowed:
            raise ValueError(
                f"حرف پلاک «{self.plate_alphabet}» برای نوع کاربری «{self.usage_type.value}» مجاز نیست"
            )
        return self


class CarPlateCreate(CarPlateBase):
    pass


class CarPlateUpdate(BaseModel):
    left_digits: Optional[str] = Field(default=None, pattern=r"^\d{2}$")
    plate_alphabet: Optional[str] = Field(default=None, min_length=1, max_length=1)
    right_digits: Optional[str] = Field(default=None, pattern=r"^\d{3}$")
    iran_code: Optional[str] = Field(default=None, pattern=r"^\d{2}$")
    plate_format: Optional[PlateFormat] = None
    usage_type: Optional[PlateUsageType] = None
    vehicle_type: Optional[VehicleType] = None
    owner_name: Optional[str] = Field(default=None, min_length=2, max_length=200)
    owner_phone: Optional[str] = None
    color: Optional[str] = Field(default=None, max_length=50)
    brand: Optional[str] = Field(default=None, max_length=80)
    model: Optional[str] = Field(default=None, max_length=80)
    manufacture_year: Optional[int] = Field(default=None, ge=1300, le=1600)
    description: Optional[str] = None
    is_active: Optional[bool] = None

    @field_validator("plate_alphabet", mode="before")
    @classmethod
    def normalize_plate_alphabet(cls, value: object) -> object:
        if value is None:
            return None
        normalized = normalize_persian_text(value)
        if normalized not in PERSIAN_PLATE_ALPHABETS:
            raise ValueError("حرف پلاک باید یکی از حروف فارسی مجاز باشد")
        return normalized

    @field_validator("owner_name", "color", "brand", "model", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: object) -> object:
        return normalize_persian_text(value)

    @field_validator("owner_phone", mode="before")
    @classmethod
    def normalize_phone(cls, value: object) -> object:
        if value is None:
            return None
        return normalize_owner_phone(value)

    @model_validator(mode="after")
    def validate_alphabet_for_usage(self) -> CarPlateUpdate:
        if self.usage_type is not None and self.plate_alphabet is not None:
            allowed = PLATE_ALPHABETS_BY_USAGE.get(self.usage_type.value)
            if allowed and self.plate_alphabet not in allowed:
                raise ValueError(
                    f"حرف پلاک «{self.plate_alphabet}» برای نوع کاربری «{self.usage_type.value}» مجاز نیست"
                )
        return self


class CarPlateResponse(CarPlateBase):
    id: int
    created_at_utc: Optional[datetime] = None
    updated_at_utc: Optional[datetime] = None
    deleted_at_utc: Optional[datetime] = None
    created_by: Optional[int] = None
    updated_by: Optional[int] = None
    created_at_jalali: str = ""
    updated_at_jalali: str | None = None

    @model_validator(mode="after")
    def _populate_jalali(self) -> CarPlateResponse:
        from app.core.jalali_utils import utc_iso_to_jalali_datetime
        if self.created_at_utc is not None:
            iso_str = self.created_at_utc.isoformat() if hasattr(self.created_at_utc, "isoformat") else str(self.created_at_utc)
            self.created_at_jalali = utc_iso_to_jalali_datetime(iso_str) or ""
        if self.updated_at_utc is not None:
            iso_str = self.updated_at_utc.isoformat() if hasattr(self.updated_at_utc, "isoformat") else str(self.updated_at_utc)
            self.updated_at_jalali = utc_iso_to_jalali_datetime(iso_str)
        return self

    @computed_field
    @property
    def formatted_plate(self) -> str:
        return f"{self.left_digits} {self.plate_alphabet} {self.right_digits} ایران {self.iran_code}"

    @computed_field
    @property
    def normalized_plate(self) -> str:
        return f"{self.left_digits}{self.plate_alphabet}{self.right_digits}{self.iran_code}"


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _get_plate_or_404(plate_id: int, runtime: Runtime) -> dict[str, Any]:
    value = runtime.car_plates.get(plate_id)
    if value is None:
        raise HTTPException(status_code=404, detail="پلاک خودرو یافت نشد")
    return value


@router.get("/", response_model=list[CarPlateResponse])
def list_car_plates(
    active_only: bool = True,
    search: str | None = Query(default=None, max_length=200),
    usage_type: PlateUsageType | None = None,
    vehicle_type: VehicleType | None = None,
    owner_phone: str | None = Query(default=None, max_length=32),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    return runtime.car_plates.list(
        active_only=active_only,
        search=search,
        usage_type=usage_type.value if usage_type else None,
        vehicle_type=vehicle_type.value if vehicle_type else None,
        owner_phone=owner_phone,
        skip=skip,
        limit=limit,
    )


@router.get("/{plate_id}", response_model=CarPlateResponse)
def get_car_plate(
    plate_id: int,
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    return _get_plate_or_404(plate_id, runtime)


# @router.post("", response_model=CarPlateResponse, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=CarPlateResponse, status_code=status.HTTP_201_CREATED)
def create_car_plate(
    payload: CarPlateCreate,
    admin_user: UserRecord = Depends(require_permission("car_plates.create")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    data = payload.model_dump(mode="json")
    data["created_by"] = admin_user.id
    data["updated_by"] = admin_user.id
    try:
        return runtime.car_plates.create(data)
    except Exception as exc:
        if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
            raise HTTPException(status_code=409, detail="این پلاک قبلاً ثبت شده است") from exc
        raise


@router.patch("/{plate_id}", response_model=CarPlateResponse)
def update_car_plate(
    plate_id: int,
    payload: CarPlateUpdate,
    admin_user: UserRecord = Depends(require_permission("car_plates.edit")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    plate = _get_plate_or_404(plate_id, runtime)
    updates = payload.model_dump(exclude_unset=True, mode="json")
    updates["updated_by"] = admin_user.id
    result = runtime.car_plates.update(plate_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="پلاک خودرو یافت نشد")
    return result


@router.delete("/{plate_id}")
def delete_car_plate(
    plate_id: int,
    superuser: UserRecord = Depends(require_permission("car_plates.delete")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, str]:
    _get_plate_or_404(plate_id, runtime)
    if not runtime.car_plates.delete(plate_id):
        raise HTTPException(status_code=404, detail="پلاک خودرو یافت نشد")
    return {"message": "پلاک خودرو با موفقیت حذف شد"}
