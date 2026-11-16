# database/models.py
# Mô hình dữ liệu (ERD xem database/ERD.md) - cú pháp SQLAlchemy 2.0 (Mapped[...] + mapped_column)
# để trình soạn thảo hiểu đúng kiểu: user.email là str, task.progress là float, ...
#
#   users ──< videos ──< analysis_tasks ──< violations >── violation_types
#     │        │               └──< task_model_stats        │
#     │   locations ─┘                                       └──< violation_reviews >── users
#     └──< user_sessions
#
# Chưa có dữ liệu camera nên "nguồn dữ liệu" là video được upload.
# Khi có camera: tạo bảng cameras (location_id) và thêm cột videos.camera_id (nullable).
from __future__ import annotations

import datetime
from typing import Optional, cast

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Table, Unicode,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


def _now() -> datetime.datetime:
    return datetime.datetime.now()


# Trạng thái hồ sơ vi phạm (giữ tiếng Việt để tương thích Frontend)
VIOLATION_STATUSES = ("Chưa xử lý", "Đã xác nhận", "Từ chối")
TASK_STATUSES = ("pending", "processing", "completed", "failed")


class User(Base):
    """Tài khoản cán bộ sử dụng hệ thống (đăng nhập, duyệt hồ sơ, upload video)."""
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'officer')", name="ck_users_role"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(Unicode(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(Unicode(100), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="officer")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, default=_now)

    videos: Mapped[list[Video]] = relationship(back_populates="uploader")
    reviews: Mapped[list[ViolationReview]] = relationship(back_populates="reviewer")
    sessions: Mapped[list[UserSession]] = relationship(back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    """
    Phiên đăng nhập. Chỉ lưu SHA-256 của token (lộ DB cũng không dùng được token).
    Đăng xuất = xóa dòng này -> token hết hiệu lực ngay (khác JWT không thu hồi được).
    """
    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, default=_now)
    expires_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, index=True)

    user: Mapped[User] = relationship(back_populates="sessions")


class Location(Base):
    """Địa điểm giám sát (ngã tư, tuyến đường)."""
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Unicode(255), unique=True, nullable=False)
    address: Mapped[Optional[str]] = mapped_column(Unicode(255), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, default=_now)

    videos: Mapped[list[Video]] = relationship(back_populates="location")


class Video(Base):
    """File video đầu vào (upload hoặc video mẫu) - đóng vai trò nguồn dữ liệu thay camera."""
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    original_name: Mapped[str] = mapped_column(Unicode(255), nullable=False)   # Tên file người dùng upload
    stored_path: Mapped[str] = mapped_column(Unicode(500), nullable=False)     # Đường dẫn file trên server
    file_size_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    location_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("locations.id"), nullable=True)
    uploaded_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    recorded_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)  # Thời điểm quay
    uploaded_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, default=_now)

    location: Mapped[Optional[Location]] = relationship(back_populates="videos")
    uploader: Mapped[Optional[User]] = relationship(back_populates="videos")
    tasks: Mapped[list[AnalysisTask]] = relationship(back_populates="video")


class AnalysisTask(Base):
    """Một lần chạy pipeline AI trên một video (lưu DB thay vì RAM để không mất khi restart)."""
    __tablename__ = "analysis_tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')",
            name="ck_analysis_tasks_status",
        ),
        Index("ix_analysis_tasks_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(16), primary_key=True)              # task_id (8 ký tự uuid)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    current_frame: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    violations_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    vehicles_scanned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    legitimate_turns: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_video_path: Mapped[Optional[str]] = mapped_column(Unicode(500), nullable=True)
    processing_time_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Unicode(1000), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, default=_now)
    started_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    video: Mapped[Video] = relationship(back_populates="tasks")
    model_stats: Mapped[list[TaskModelStat]] = relationship(back_populates="task", cascade="all, delete-orphan")
    violations: Mapped[list[Violation]] = relationship(back_populates="task")


class TaskModelStat(Base):
    """Thống kê thực tế của từng mô hình AI trong một lần phân tích."""
    __tablename__ = "task_model_stats"

    task_id: Mapped[str] = mapped_column(String(16), ForeignKey("analysis_tasks.id"), primary_key=True)
    model_name: Mapped[str] = mapped_column(Unicode(50), primary_key=True)     # 'Vehicle', 'Traffic Light', ...
    weight_file: Mapped[str] = mapped_column(Unicode(100), nullable=False)
    detections: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0..1, NULL nếu không có

    task: Mapped[AnalysisTask] = relationship(back_populates="model_stats")


class ViolationType(Base):
    """Danh mục loại vi phạm (chuẩn hóa thay cho chuỗi tự do)."""
    __tablename__ = "violation_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)   # 'RED_LIGHT', 'NO_HELMET', ...
    name: Mapped[str] = mapped_column(Unicode(100), unique=True, nullable=False) # 'Vượt đèn đỏ', ...
    description: Mapped[Optional[str]] = mapped_column(Unicode(255), nullable=True)

    violations: Mapped[list[Violation]] = relationship(back_populates="violation_type")


class Violation(Base):
    """Hồ sơ vi phạm do AI phát hiện."""
    __tablename__ = "violations"
    # ck_violations_status được thêm ở cuối file (cần literal N'...' trên SQL Server)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    violation_code: Mapped[str] = mapped_column(Unicode(50), unique=True, nullable=False)   # 'VP-000001'
    task_id: Mapped[str] = mapped_column(String(16), ForeignKey("analysis_tasks.id"), nullable=False, index=True)
    violation_type_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("violation_types.id"), nullable=False, index=True,
    )

    # Phương tiện
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)            # ID ByteTrack trong video
    license_plate: Mapped[str] = mapped_column(Unicode(50), nullable=False, index=True)
    vehicle_type: Mapped[Optional[str]] = mapped_column(Unicode(100), nullable=True)
    vehicle_color: Mapped[Optional[str]] = mapped_column(Unicode(50), nullable=True)
    ai_reason: Mapped[Optional[str]] = mapped_column(Unicode(255), nullable=True)      # Mô tả gốc từ thuật toán

    # Thời gian
    frame_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    video_time_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # Mốc trên video (giây)
    occurred_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, default=_now, index=True)

    # Độ tin cậy (%), NULL nếu mô hình không cung cấp
    confidence_overall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_vehicle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_light: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_plate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_violation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(Unicode(50), nullable=False, default="Chưa xử lý", index=True)
    evidence_image: Mapped[Optional[str]] = mapped_column(Unicode(255), nullable=True)
    plate_crop_image: Mapped[Optional[str]] = mapped_column(Unicode(255), nullable=True)
    # Ảnh cận cảnh đối tượng vi phạm (VD: người lái không đội mũ, đầu được khoanh đỏ); NULL nếu không có
    subject_crop_image: Mapped[Optional[str]] = mapped_column(Unicode(255), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, default=_now)

    task: Mapped[AnalysisTask] = relationship(back_populates="violations")
    violation_type: Mapped[ViolationType] = relationship(back_populates="violations")
    reviews: Mapped[list[ViolationReview]] = relationship(
        back_populates="violation", cascade="all, delete-orphan", order_by="ViolationReview.reviewed_at",
    )


class ViolationReview(Base):
    """Lịch sử duyệt / từ chối hồ sơ vi phạm."""
    __tablename__ = "violation_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    violation_id: Mapped[int] = mapped_column(Integer, ForeignKey("violations.id"), nullable=False, index=True)
    reviewer_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    old_status: Mapped[str] = mapped_column(Unicode(50), nullable=False)
    new_status: Mapped[str] = mapped_column(Unicode(50), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Unicode(500), nullable=True)
    reviewed_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, default=_now)

    violation: Mapped[Violation] = relationship(back_populates="reviews")
    reviewer: Mapped[Optional[User]] = relationship(back_populates="reviews")


# Dùng biểu thức IN thay vì chuỗi SQL để SQLAlchemy tự sinh literal đúng dialect
# (SQL Server: N'Chưa xử lý' — thiếu tiền tố N thì ký tự tiếng Việt bị biến thành '?').
_violations_table = cast(Table, Violation.__table__)
_violations_table.append_constraint(
    CheckConstraint(_violations_table.c.status.in_(VIOLATION_STATUSES), name="ck_violations_status")
)
_reviews_table = cast(Table, ViolationReview.__table__)
for _col in ("old_status", "new_status"):
    _reviews_table.append_constraint(
        CheckConstraint(_reviews_table.c[_col].in_(VIOLATION_STATUSES), name=f"ck_violation_reviews_{_col}")
    )
