# api/system.py
# Trạng thái hệ thống thật (thay cho số liệu giả trên giao diện):
#   - GET /api/v1/system/status         : thiết bị AI (GPU/CPU), mô hình, đang rảnh/đang phân tích, tốc độ xử lý gần nhất
#   - GET /api/v1/system/notifications  : thông báo sinh từ vi phạm mới + lần phân tích hoàn tất / thất bại
#   - GET /api/v1/public/stats          : số liệu tổng hợp cho trang chủ (KHÔNG cần đăng nhập, chỉ số đếm)
import datetime
import os
import threading

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from api.deps import get_current_user
from database.connection import engine, get_db
from database.models import AnalysisTask, Location, Video, Violation

router = APIRouter(prefix="/api/v1/system", tags=["System"], dependencies=[Depends(get_current_user)])
public_router = APIRouter(prefix="/api/v1/public", tags=["Public"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

_device_lock = threading.Lock()
_device_cache: dict | None = None


def _ai_device() -> dict:
    """Thiết bị chạy AI. Nạp PyTorch lần đầu khi được hỏi (không làm chậm lúc khởi động server)."""
    global _device_cache
    with _device_lock:
        if _device_cache is None:
            try:
                from models.yolo_types import DEVICE, DEVICE_NAME, HALF
                _device_cache = {"name": DEVICE_NAME, "type": "cpu" if DEVICE == "cpu" else "gpu", "fp16": bool(HALF)}
            except Exception as exc:   # thiếu torch / lỗi driver: vẫn trả trạng thái, không làm hỏng API
                _device_cache = {"name": "Không xác định", "type": "unknown", "fp16": False, "error": str(exc)[:200]}
        return _device_cache


def _model_files() -> list[dict]:
    from api.analyze import MODEL_INFO   # danh sách mô hình dùng cho pipeline phân tích
    return [
        {"key": key, "name": name, "weight": weight, "available": os.path.isfile(os.path.join(WEIGHTS_DIR, weight))}
        for key, (name, weight, _module) in MODEL_INFO.items()
    ]


@router.get("/status")
def system_status(db: Session = Depends(get_db)):
    active = db.query(func.count(AnalysisTask.id)).filter(AnalysisTask.status.in_(("pending", "processing"))).scalar() or 0
    last = (
        db.query(AnalysisTask)
        .filter(AnalysisTask.status == "completed", AnalysisTask.processing_time_sec > 0)
        .order_by(AnalysisTask.finished_at.desc())
        .first()
    )
    last_fps = None
    if last is not None and last.total_frames and last.processing_time_sec:
        last_fps = round(last.total_frames / last.processing_time_sec, 1)
    pending_review = db.query(func.count(Violation.id)).filter(Violation.status == "Chưa xử lý").scalar() or 0
    models = _model_files()
    device = _ai_device()
    return {
        "online": True,
        "server_time": datetime.datetime.now().isoformat(timespec="seconds"),
        "database": "SQL Server" if engine.dialect.name == "mssql" else engine.dialect.name,
        "device": device,
        "models": {"total": len(models), "available": sum(m["available"] for m in models), "items": models},
        "ai_state": "busy" if active else ("ready" if all(m["available"] for m in models) else "degraded"),
        "active_tasks": active,
        "last_processing_fps": last_fps,
        "last_task_finished_at": last.finished_at.isoformat(timespec="seconds") if last and last.finished_at else None,
        "pending_review": pending_review,
    }


@router.get("/notifications")
def notifications(limit: int = Query(20, ge=1, le=50), db: Session = Depends(get_db)):
    """Thông báo mới nhất: vi phạm AI vừa ghi nhận + kết quả các lần phân tích (hoàn tất / thất bại)."""
    items = []
    violations = (
        db.query(Violation).options(joinedload(Violation.violation_type))
        .order_by(Violation.created_at.desc()).limit(limit).all()
    )
    for v in violations:
        vtype = v.violation_type.name if v.violation_type else "Vi phạm"
        items.append({
            "id": f"v-{v.id}",
            "type": "alert",
            "title": f"Phát hiện vi phạm: {vtype}",
            "text": f"{v.vehicle_type or 'Phương tiện'} {v.license_plate} · {v.violation_code}",
            "time": v.created_at.isoformat(timespec="seconds"),
            "link": f"/admin/violations/{v.id}",
        })
    tasks = (
        db.query(AnalysisTask).options(joinedload(AnalysisTask.video))
        .filter(AnalysisTask.status.in_(("completed", "failed")), AnalysisTask.finished_at.isnot(None))
        .order_by(AnalysisTask.finished_at.desc()).limit(limit).all()
    )
    for t in tasks:
        if t.finished_at is None:      # đã lọc trong truy vấn; kiểm tra lại cho chắc (và cho trình kiểm tra kiểu)
            continue
        name = t.video.original_name if t.video else t.id
        ok = t.status == "completed"
        items.append({
            "id": f"t-{t.id}",
            "type": "info" if ok else "error",
            "title": "Phân tích video hoàn tất" if ok else "Phân tích video thất bại",
            "text": f"{name} · {t.violations_detected} vi phạm" if ok else f"{name} · {(t.error_message or 'Lỗi không xác định')[:90]}",
            "time": t.finished_at.isoformat(timespec="seconds"),
            "link": "/admin/video-analysis",
        })
    items.sort(key=lambda x: x["time"], reverse=True)
    return {"items": items[:limit]}


@public_router.get("/stats")
def public_stats(db: Session = Depends(get_db)):
    """Số liệu tổng hợp hiển thị trên trang chủ (chỉ số đếm, không có dữ liệu cá nhân / biển số)."""
    avg_conf = db.query(func.avg(Violation.confidence_overall)).filter(Violation.confidence_overall.isnot(None)).scalar()
    return {
        "locations": db.query(func.count(Location.id)).scalar() or 0,
        "videos_analyzed": db.query(func.count(AnalysisTask.id)).filter(AnalysisTask.status == "completed").scalar() or 0,
        "vehicles_scanned": int(db.query(func.coalesce(func.sum(AnalysisTask.vehicles_scanned), 0)).scalar() or 0),
        "violations": db.query(func.count(Violation.id)).scalar() or 0,
        "avg_confidence": round(float(avg_conf), 1) if avg_conf is not None else None,
        "videos_uploaded": db.query(func.count(Video.id)).scalar() or 0,
    }
