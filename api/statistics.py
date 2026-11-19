# api/statistics.py
import datetime
from collections import defaultdict
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import TaskModelStat, Violation, ViolationType
from database.lookups import normalize_vehicle_type

router = APIRouter(prefix="/api/v1/statistics", tags=["Statistics"])


def _avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def _model_accuracy(db: Session, violations) -> dict:
    """Độ tin cậy trung bình từng mô hình: ưu tiên số liệu thật trong task_model_stats (0..1 -> %)."""
    rows = dict(
        db.query(TaskModelStat.model_name, func.avg(TaskModelStat.avg_confidence))
        .filter(TaskModelStat.avg_confidence.isnot(None))
        .group_by(TaskModelStat.model_name).all()
    )

    def pct(name, fallback):
        v = rows.get(name)
        return round(float(v) * 100, 1) if v is not None else fallback

    overall = _avg(v.confidence_overall for v in violations)
    return {
        "vehicle": pct("Vehicle", _avg(v.confidence_vehicle for v in violations)),
        "trafficLight": pct("Traffic Light", _avg(v.confidence_light for v in violations)),
        "helmet": pct("Helmet", None),
        "plateOcr": pct("License Plate", _avg(v.confidence_plate for v in violations)),
        "overall": overall,
    }


@router.get("/summary")
def get_statistics_summary(db: Session = Depends(get_db)):
    """Thống kê tổng hợp cho trang Statistics, tính hoàn toàn từ dữ liệu trong DB."""
    violations = db.query(Violation).all()
    total = len(violations)
    today = datetime.date.today()

    # Theo loại vi phạm: liệt kê đủ mọi loại trong danh mục (kể cả loại = 0)
    type_counts = dict(
        db.query(ViolationType.name, func.count(Violation.id))
        .outerjoin(Violation, Violation.violation_type_id == ViolationType.id)
        .group_by(ViolationType.id, ViolationType.name).all()
    )
    by_type = [
        {"name": n, "count": c, "percentage": round(c / total * 100, 1) if total else 0.0}
        for n, c in sorted(type_counts.items(), key=lambda x: -x[1])
    ]

    veh_counts = defaultdict(int)
    for v in violations:
        veh_counts[normalize_vehicle_type(v.vehicle_type) or "Khác"] += 1
    by_vehicle = [
        {"name": n, "count": c, "percentage": round(c / total * 100, 1)}
        for n, c in sorted(veh_counts.items(), key=lambda x: -x[1])
    ]

    date_counts = defaultdict(int)
    for v in violations:
        date_counts[v.occurred_at.date()] += 1
    by_date = [{"date": d.strftime("%d/%m/%Y"), "count": c} for d, c in sorted(date_counts.items())]

    return {
        "totalViolations": total,
        "todayViolations": sum(1 for v in violations if v.occurred_at.date() == today),
        "confirmedViolations": sum(1 for v in violations if v.status == "Đã xác nhận"),
        "pendingApproval": sum(1 for v in violations if v.status == "Chưa xử lý"),
        "rejectedViolations": sum(1 for v in violations if v.status == "Từ chối"),
        "avgConfidence": _avg(v.confidence_overall for v in violations) or 0.0,
        "byType": by_type,
        "byVehicle": by_vehicle,
        "byDate": by_date,
        "modelAccuracy": _model_accuracy(db, violations),
    }
