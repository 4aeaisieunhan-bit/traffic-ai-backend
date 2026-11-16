# schemas/violation.py
import datetime
from typing import Optional, List
from pydantic import BaseModel


class ViolationResponse(BaseModel):
    """
    Hồ sơ vi phạm trả về cho Frontend.
    Các trường video_source / location / violation_type / time_str / date_str / video_time
    được TÍNH từ các bảng liên quan (videos, locations, violation_types) và occurred_at,
    giữ nguyên định dạng cũ để Frontend Angular không phải thay đổi.
    """
    id: int
    violation_code: str
    task_id: str
    video_id: Optional[int] = None
    video_source: Optional[str] = None
    location: Optional[str] = None
    license_plate: str
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    violation_type: str
    violation_type_code: str
    ai_reason: Optional[str] = None
    track_id: Optional[int] = None
    frame_index: Optional[int] = None
    video_time_sec: Optional[float] = None
    occurred_at: datetime.datetime
    time_str: str            # 'HH:MM:SS'
    date_str: str            # 'dd/mm/yyyy'
    video_time: Optional[str] = None   # '00:05.4s'
    confidence_overall: Optional[float] = None
    confidence_vehicle: Optional[float] = None
    confidence_light: Optional[float] = None
    confidence_plate: Optional[float] = None
    confidence_violation: Optional[float] = None
    status: str
    evidence_image: Optional[str] = None
    plate_crop_image: Optional[str] = None
    subject_crop_image: Optional[str] = None   # ảnh cận cảnh người vi phạm (lỗi mũ)
    created_at: datetime.datetime
    # Dữ liệu huấn luyện lại mô hình helmet (chỉ điền ở API chi tiết / duyệt; utils/helmet_dataset.py)
    training_sample: bool = False              # vi phạm có mẫu huấn luyện (vi phạm mũ phát hiện từ bản mới)
    training_label: Optional[str] = None       # nhãn đã ghi vào bộ dữ liệu: 'helmet' | 'no_helmet' | None


class ViolationStatusUpdate(BaseModel):
    status: str  # 'Chưa xử lý' | 'Đã xác nhận' | 'Từ chối'
    note: Optional[str] = None
    # Lý do từ chối có mã (VD: 'has_helmet' = người lái có đội mũ, mô hình nhận nhầm -> thành dữ liệu huấn luyện)
    reason_code: Optional[str] = None


class ViolationReviewResponse(BaseModel):
    id: int
    old_status: str
    new_status: str
    note: Optional[str] = None
    reviewer_name: Optional[str] = None
    reviewed_at: datetime.datetime


class ViolationTypeResponse(BaseModel):
    id: int
    code: str
    name: str
    description: Optional[str] = None


class ViolationFilterOptions(BaseModel):
    violation_types: List[ViolationTypeResponse]
    video_sources: List[str]
    locations: List[str]


def format_video_time(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds % 60:04.1f}s"


def violation_to_response(v) -> ViolationResponse:
    """Chuyển ORM Violation -> ViolationResponse (cần load sẵn task.video.location & violation_type)."""
    video = v.task.video if v.task else None
    location = video.location if video else None
    return ViolationResponse(
        id=v.id,
        violation_code=v.violation_code,
        task_id=v.task_id,
        video_id=video.id if video else None,
        video_source=video.original_name if video else None,
        location=location.name if location else None,
        license_plate=v.license_plate,
        vehicle_type=v.vehicle_type,
        vehicle_color=v.vehicle_color,
        violation_type=v.violation_type.name,
        violation_type_code=v.violation_type.code,
        ai_reason=v.ai_reason,
        track_id=v.track_id,
        frame_index=v.frame_index,
        video_time_sec=v.video_time_sec,
        occurred_at=v.occurred_at,
        time_str=v.occurred_at.strftime("%H:%M:%S"),
        date_str=v.occurred_at.strftime("%d/%m/%Y"),
        video_time=format_video_time(v.video_time_sec),
        confidence_overall=v.confidence_overall,
        confidence_vehicle=v.confidence_vehicle,
        confidence_light=v.confidence_light,
        confidence_plate=v.confidence_plate,
        confidence_violation=v.confidence_violation,
        status=v.status,
        evidence_image=v.evidence_image,
        plate_crop_image=v.plate_crop_image,
        subject_crop_image=v.subject_crop_image,
        created_at=v.created_at,
    )
