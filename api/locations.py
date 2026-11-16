# api/locations.py
# Quản lý địa điểm giám sát. Video upload bắt buộc phải gắn với một địa điểm trong bảng này.
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import AnalysisTask, Location, Video, Violation
from schemas.location import LocationCreate, LocationUpdate, LocationResponse

router = APIRouter(prefix="/api/v1/locations", tags=["Locations"])


def _counts(db: Session):
    """{location_id: (số video, số vi phạm)}"""
    videos = dict(
        db.query(Video.location_id, func.count(Video.id))
        .filter(Video.location_id.isnot(None)).group_by(Video.location_id).all()
    )
    violations = dict(
        db.query(Video.location_id, func.count(Violation.id))
        .join(AnalysisTask, AnalysisTask.video_id == Video.id)
        .join(Violation, Violation.task_id == AnalysisTask.id)
        .filter(Video.location_id.isnot(None)).group_by(Video.location_id).all()
    )
    return videos, violations


def _to_response(loc: Location, videos: dict, violations: dict) -> LocationResponse:
    return LocationResponse(
        id=loc.id, name=loc.name, address=loc.address,
        latitude=loc.latitude, longitude=loc.longitude, created_at=loc.created_at,
        video_count=videos.get(loc.id, 0), violation_count=violations.get(loc.id, 0),
    )


def _get_or_404(db: Session, location_id: int) -> Location:
    loc = db.get(Location, location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Không tìm thấy địa điểm")
    return loc


def _ensure_unique_name(db: Session, name: str, exclude_id: int | None = None) -> None:
    # So sánh không phân biệt hoa thường để chặn 'Ngã tư A' và 'ngã tư a'.
    # Làm ở Python (casefold) vì lower() của SQLite không xử lý chữ có dấu tiếng Việt.
    key = name.casefold()
    for loc_id, existing in db.query(Location.id, Location.name).all():
        if loc_id != exclude_id and existing.casefold() == key:
            raise HTTPException(status_code=409, detail=f"Địa điểm '{existing}' đã tồn tại")


@router.get("", response_model=List[LocationResponse])
def list_locations(db: Session = Depends(get_db)):
    """Danh sách địa điểm (kèm số video / số vi phạm) - dùng cho dropdown khi upload video."""
    videos, violations = _counts(db)
    return [_to_response(l, videos, violations) for l in db.query(Location).order_by(Location.name).all()]


@router.get("/{location_id}", response_model=LocationResponse)
def get_location(location_id: int, db: Session = Depends(get_db)):
    loc = _get_or_404(db, location_id)
    videos, violations = _counts(db)
    return _to_response(loc, videos, violations)


@router.post("", response_model=LocationResponse, status_code=status.HTTP_201_CREATED)
def create_location(payload: LocationCreate, db: Session = Depends(get_db)):
    _ensure_unique_name(db, payload.name)
    loc = Location(**payload.model_dump())
    db.add(loc)
    db.commit()
    db.refresh(loc)
    return _to_response(loc, {}, {})


@router.put("/{location_id}", response_model=LocationResponse)
def update_location(location_id: int, payload: LocationUpdate, db: Session = Depends(get_db)):
    loc = _get_or_404(db, location_id)
    _ensure_unique_name(db, payload.name, exclude_id=location_id)
    for field, value in payload.model_dump().items():
        setattr(loc, field, value)
    db.commit()
    db.refresh(loc)
    videos, violations = _counts(db)
    return _to_response(loc, videos, violations)


@router.delete("/{location_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_location(location_id: int, db: Session = Depends(get_db)):
    """Chỉ xóa được địa điểm chưa có video nào (tránh mất liên kết hồ sơ vi phạm)."""
    loc = _get_or_404(db, location_id)
    n_videos = db.query(func.count(Video.id)).filter(Video.location_id == location_id).scalar()
    if n_videos:
        raise HTTPException(
            status_code=409,
            detail=f"Không thể xóa: địa điểm đang có {n_videos} video. Hãy sửa thông tin thay vì xóa.",
        )
    db.delete(loc)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
