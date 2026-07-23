from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.auth import get_current_user, require_role
from app.core.auth_store import UserRecord
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/car-plates", tags=["legacy-car-plates"])


class CarPlateCreate(BaseModel):
    left_digits: str = Field(pattern=r"^\d{2}$")
    plate_alphabet: str = Field(min_length=1, max_length=1)
    right_digits: str = Field(pattern=r"^\d{3}$")
    iran_code: str = Field(pattern=r"^\d{2}$")
    plate_format: str = "standard"
    usage_type: str
    vehicle_type: str
    owner_name: str = Field(min_length=2, max_length=200)
    owner_phone: str = Field(min_length=1, max_length=32)
    color: str | None = None
    brand: str | None = None
    model: str | None = None
    manufacture_year: int | None = Field(default=None, ge=1300, le=1600)
    description: str | None = None
    is_active: bool = True


class CarPlateUpdate(BaseModel):
    left_digits: str | None = Field(default=None, pattern=r"^\d{2}$")
    plate_alphabet: str | None = Field(default=None, min_length=1, max_length=1)
    right_digits: str | None = Field(default=None, pattern=r"^\d{3}$")
    iran_code: str | None = Field(default=None, pattern=r"^\d{2}$")
    plate_format: str | None = None
    usage_type: str | None = None
    vehicle_type: str | None = None
    owner_name: str | None = Field(default=None, min_length=2, max_length=200)
    owner_phone: str | None = Field(default=None, max_length=32)
    color: str | None = None
    brand: str | None = None
    model: str | None = None
    manufacture_year: int | None = Field(default=None, ge=1300, le=1600)
    description: str | None = None
    is_active: bool | None = None


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


@router.get("")
@router.get("/")
def list_car_plates(
    active_only: bool = True,
    search: str | None = Query(default=None, max_length=200),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    return runtime.car_plates.list(active_only=active_only, search=search, skip=skip, limit=limit)


@router.get("/{plate_id}")
def get_car_plate(plate_id: int, _: UserRecord = Depends(get_current_user), runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    value = runtime.car_plates.get(plate_id)
    if value is None:
        raise HTTPException(status_code=404, detail="Car plate not found")
    return value


@router.post("", status_code=status.HTTP_201_CREATED)
@router.post("/", status_code=status.HTTP_201_CREATED)
def create_car_plate(payload: CarPlateCreate, _: UserRecord = Depends(require_role("admin")), runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    try:
        return runtime.car_plates.create(payload.model_dump())
    except Exception as exc:
        if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
            raise HTTPException(status_code=409, detail="Car plate already exists") from exc
        raise


@router.patch("/{plate_id}")
def update_car_plate(plate_id: int, payload: CarPlateUpdate, _: UserRecord = Depends(require_role("admin")), runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    value = runtime.car_plates.update(plate_id, payload.model_dump(exclude_unset=True))
    if value is None:
        raise HTTPException(status_code=404, detail="Car plate not found")
    return value


@router.delete("/{plate_id}")
def delete_car_plate(plate_id: int, _: UserRecord = Depends(require_role("superuser")), runtime: Runtime = Depends(get_runtime)) -> dict[str, str]:
    if not runtime.car_plates.delete(plate_id):
        raise HTTPException(status_code=404, detail="Car plate not found")
    return {"message": "Car plate deleted successfully"}
