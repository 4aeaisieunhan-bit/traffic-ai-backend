# schemas/location.py
import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class LocationBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Tên địa điểm, VD: 'Ngã tư Nguyễn Trãi - Khuất Duy Tiến'")
    address: Optional[str] = Field(None, max_length=255)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = " ".join(v.split())  # bỏ khoảng trắng thừa để tránh tạo trùng
        if not v:
            raise ValueError("Tên địa điểm không được để trống")
        return v

    @field_validator("address")
    @classmethod
    def _strip_address(cls, v: Optional[str]) -> Optional[str]:
        v = " ".join(v.split()) if v else None
        return v or None


class LocationCreate(LocationBase):
    pass


class LocationUpdate(LocationBase):
    """PUT: gửi đầy đủ thông tin địa điểm."""
    pass


class LocationResponse(LocationBase):
    id: int
    created_at: datetime.datetime
    video_count: int = 0
    violation_count: int = 0
