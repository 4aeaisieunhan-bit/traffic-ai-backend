# api/violations.py
import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from database.connection import get_db
from database.models import (
    AnalysisTask, Location, User, Video, Violation, ViolationReview, ViolationType, VIOLATION_STATUSES,
)
from api.deps import get_current_user, require_admin
from api.cleanup import delete_violation
from api.reports import build_excel, build_pdf
from database.lookups import violation_query
from schemas.violation import (
    ViolationResponse, ViolationStatusUpdate, ViolationReviewResponse,
    ViolationTypeResponse, ViolationFilterOptions, violation_to_response,
)

router = APIRouter(prefix="/api/v1/violations", tags=["Violations"])

ALL_TYPES = {"Tất cả loại vi phạm"}
ALL_LOCATIONS = {"Tất cả nguồn", "Tất cả vị trí"}
ALL_STATUSES = {"Tất cả trạng thái"}

# Lý do từ chối có mã (frontend gửi reason_code): người lái có đội mũ, AI nhận nhầm thành "không đội mũ"
REASON_LABELS = {"has_helmet": "Có đội mũ — mô hình nhận nhầm"}


def _today_range():
    start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
    return start, start + datetime.timedelta(days=1)


def _filtered_query(db, search, violation_type, location, status, task_id, date_from=None, date_to=None):
    """Truy vấn hồ sơ vi phạm theo bộ lọc (dùng chung cho danh sách và xuất báo cáo)."""
    query = (
        violation_query(db)
        .join(Violation.violation_type)
        .join(Violation.task)
        .join(AnalysisTask.video)
        .outerjoin(Video.location)
    )

    if search:
        s = f"%{search.strip()}%"
        query = query.filter(or_(
            Violation.license_plate.ilike(s),
            Violation.vehicle_type.ilike(s),
            Violation.violation_code.ilike(s),
        ))
    if violation_type and violation_type not in ALL_TYPES:
        query = query.filter(or_(ViolationType.name == violation_type, ViolationType.code == violation_type))
    if location and location not in ALL_LOCATIONS:
        s = f"%{location}%"
        query = query.filter(or_(Location.name.ilike(s), Video.original_name.ilike(s)))
    if status and status not in ALL_STATUSES:
        query = query.filter(Violation.status == status)
    if task_id:
        query = query.filter(Violation.task_id == task_id)
    if date_from:
        query = query.filter(Violation.occurred_at >= datetime.datetime.combine(date_from, datetime.time.min))
    if date_to:
        query = query.filter(Violation.occurred_at <= datetime.datetime.combine(date_to, datetime.time.max))
    return query


@router.get("", response_model=Dict[str, Any])
def get_violations(
    search: Optional[str] = Query(None, description="Tìm biển số, loại xe hoặc mã hồ sơ"),
    violation_type: Optional[str] = Query(None, description="Tên hoặc mã loại vi phạm (VD: 'Vượt đèn đỏ' / 'RED_LIGHT')"),
    location: Optional[str] = Query(None, description="Tên địa điểm hoặc tên file video"),
    status: Optional[str] = Query(None, description="Chưa xử lý / Đã xác nhận / Từ chối"),
    task_id: Optional[str] = Query(None, description="Chỉ lấy vi phạm của một lần phân tích"),
    date_from: Optional[datetime.date] = Query(None, description="Từ ngày (YYYY-MM-DD)"),
    date_to: Optional[datetime.date] = Query(None, description="Đến ngày (YYYY-MM-DD)"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Danh sách hồ sơ vi phạm có lọc & phân trang (trang /admin/violations)."""
    query = _filtered_query(db, search, violation_type, location, status, task_id, date_from, date_to)
    total_count = query.count()
    items = query.order_by(Violation.id.desc()).offset((page - 1) * limit).limit(limit).all()

    return {
        "total": total_count,
        "page": page,
        "limit": limit,
        "items": [violation_to_response(v) for v in items],
    }


@router.get("/export")
def export_violations(
    format: str = Query("xlsx", pattern="^(xlsx|pdf)$", description="xlsx hoặc pdf"),
    search: Optional[str] = Query(None),
    violation_type: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    task_id: Optional[str] = Query(None),
    date_from: Optional[datetime.date] = Query(None),
    date_to: Optional[datetime.date] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Xuất báo cáo vi phạm theo đúng bộ lọc đang chọn: Excel (đầy đủ) hoặc PDF (A4 ngang)."""
    items = (
        _filtered_query(db, search, violation_type, location, status, task_id, date_from, date_to)
        .order_by(Violation.occurred_at.desc(), Violation.id.desc()).all()
    )
    filters = {"search": search, "violation_type": violation_type if violation_type not in ALL_TYPES else None,
               "location": location if location not in ALL_LOCATIONS else None,
               "status": status if status not in ALL_STATUSES else None, "task_id": task_id,
               "date_from": date_from.strftime("%d/%m/%Y") if date_from else None,
               "date_to": date_to.strftime("%d/%m/%Y") if date_to else None}
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if format == "pdf":
        content, media, ext = build_pdf(items, filters, user.full_name), "application/pdf", "pdf"
    else:
        content = build_excel(items, filters, user.full_name)
        media, ext = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
    return Response(
        content=content, media_type=media,
        headers={"Content-Disposition": f'attachment; filename="bao-cao-vi-pham_{stamp}.{ext}"',
                 "Access-Control-Expose-Headers": "Content-Disposition"},
    )


@router.get("/summary/counts")
def get_violation_counts(db: Session = Depends(get_db)):
    """Số liệu cho các thẻ tóm tắt trên trang Danh sách vi phạm."""
    start, end = _today_range()
    q = db.query(Violation)
    return {
        "totalViolations": q.count(),
        "todayViolations": q.filter(Violation.created_at >= start, Violation.created_at < end).count(),
        "pendingApproval": q.filter(Violation.status == "Chưa xử lý").count(),
        "confirmedViolations": q.filter(Violation.status == "Đã xác nhận").count(),
        "rejectedViolations": q.filter(Violation.status == "Từ chối").count(),
    }


@router.get("/meta/filters", response_model=ViolationFilterOptions)
def get_filter_options(db: Session = Depends(get_db)):
    """Các giá trị cho dropdown lọc (lấy từ DB thay vì viết cứng ở Frontend)."""
    types = db.query(ViolationType).order_by(ViolationType.id).all()
    sources = [
        r[0] for r in db.query(Video.original_name)
        .join(AnalysisTask, AnalysisTask.video_id == Video.id)
        .join(Violation, Violation.task_id == AnalysisTask.id)
        .distinct().order_by(Video.original_name).all()
    ]
    locations = [r[0] for r in db.query(Location.name).order_by(Location.name).all()]
    return ViolationFilterOptions(
        violation_types=[ViolationTypeResponse.model_validate(t, from_attributes=True) for t in types],
        video_sources=sources,
        locations=locations,
    )


@router.get("/{violation_id}", response_model=ViolationResponse)
def get_violation_detail(violation_id: int, db: Session = Depends(get_db)):
    """Chi tiết hồ sơ vi phạm (trang /admin/violations/:id)."""
    item = violation_query(db).filter(Violation.id == violation_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ vi phạm")
    return violation_to_response(item)


@router.get("/{violation_id}/reviews", response_model=List[ViolationReviewResponse])
def get_violation_reviews(violation_id: int, db: Session = Depends(get_db)):
    """Lịch sử duyệt / từ chối của hồ sơ."""
    if not db.get(Violation, violation_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ vi phạm")
    reviews = (
        db.query(ViolationReview).options(joinedload(ViolationReview.reviewer))
        .filter(ViolationReview.violation_id == violation_id)
        .order_by(ViolationReview.reviewed_at.desc()).all()
    )
    return [
        ViolationReviewResponse(
            id=r.id, old_status=r.old_status, new_status=r.new_status, note=r.note,
            reviewer_name=r.reviewer.full_name if r.reviewer else None,
            reviewed_at=r.reviewed_at,
        )
        for r in reviews
    ]


@router.patch("/{violation_id}/status", response_model=ViolationResponse)
def update_violation_status(
    violation_id: int,
    payload: ViolationStatusUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Cán bộ duyệt / từ chối hồ sơ; mỗi lần đổi trạng thái được ghi vào violation_reviews."""
    if payload.status not in VIOLATION_STATUSES:
        raise HTTPException(status_code=400, detail=f"Trạng thái không hợp lệ. Phải là một trong: {list(VIOLATION_STATUSES)}")

    item = violation_query(db).filter(Violation.id == violation_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ vi phạm")

    if payload.reason_code is not None and payload.reason_code not in REASON_LABELS:
        raise HTTPException(status_code=400, detail="Mã lý do không hợp lệ")
    if payload.reason_code and payload.status != "Từ chối":
        raise HTTPException(status_code=400, detail="Lý do có mã chỉ dùng khi từ chối hồ sơ")
    note = payload.note or (REASON_LABELS.get(payload.reason_code or "") or None)

    if item.status != payload.status:
        db.add(ViolationReview(
            violation_id=item.id, reviewer_id=user.id,
            old_status=item.status, new_status=payload.status, note=note,
        ))
        item.status = payload.status
        db.commit()
        db.refresh(item)
    return violation_to_response(item)


@router.delete("/{violation_id}", status_code=204)
def delete_violation_record(violation_id: int, db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    """Xoá hồ sơ vi phạm (VD: AI nhận nhầm) cùng ảnh bằng chứng và lịch sử duyệt. Chỉ quản trị viên."""
    v = db.get(Violation, violation_id)
    if v is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ vi phạm")
    task = v.task
    delete_violation(db, v)
    if task is not None and task.violations_detected > 0:
        task.violations_detected -= 1          # giữ số đếm của lần phân tích khớp với dữ liệu
    db.commit()
