# database/migrate.py
# Chuyển dữ liệu từ cấu trúc cũ (1 bảng violations phẳng) sang cấu trúc chuẩn hóa mới.
#
# Quy trình (tự chạy khi khởi động server.py):
#   1. rename_legacy_table(): nếu bảng `violations` là bản cũ (không có cột task_id)
#      -> đổi tên thành `violations_legacy` (KHÔNG xóa dữ liệu).
#   2. Base.metadata.create_all() tạo các bảng mới.
#   3. add_missing_columns(): create_all() KHÔNG thêm cột vào bảng đã có -> tự ALTER TABLE thêm cột mới.
#   4. migrate_legacy_rows(): copy từng hồ sơ cũ sang bảng mới, tạo videos / analysis_tasks tương ứng.
# Sau khi kiểm tra dữ liệu mới đúng, có thể tự xóa bảng violations_legacy.
import datetime
import glob
import os
import re
from typing import Optional

import cv2
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from database.models import AnalysisTask, Video, Violation
from database.lookups import (
    classify_reason, get_violation_type, normalize_vehicle_type,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")

LEGACY_TABLE = "violations_legacy"
LEGACY_TASKS_TABLE = "analysis_tasks_legacy"

# Bảng cũ cần đổi tên -> (tên mới, cột chỉ có ở cấu trúc mới dùng để nhận biết)
LEGACY_RENAMES = {
    "violations": (LEGACY_TABLE, "task_id"),
    "analysis_tasks": (LEGACY_TASKS_TABLE, "video_id"),
}


def rename_legacy_table(engine: Engine) -> None:
    """Đổi tên các bảng còn cấu trúc cũ (violations, analysis_tasks) -> *_legacy, giữ nguyên dữ liệu."""
    insp = inspect(engine)
    tables = insp.get_table_names()
    for table, (legacy_name, marker_col) in LEGACY_RENAMES.items():
        if table not in tables:
            continue
        if marker_col in {c["name"] for c in insp.get_columns(table)}:
            continue  # đã là cấu trúc mới

        target = legacy_name
        if target in tables:
            target = f"{legacy_name}_{datetime.datetime.now():%Y%m%d%H%M%S}"

        with engine.begin() as conn:
            if engine.dialect.name == "mssql":
                conn.execute(text("EXEC sp_rename :src, :t"), {"src": f"dbo.{table}", "t": target})
            else:
                conn.execute(text(f'ALTER TABLE "{table}" RENAME TO "{target}"'))
                # SQLite: tên index là toàn cục -> xóa index cũ để bảng mới tạo lại được
                idx = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=:t AND sql IS NOT NULL"),
                    {"t": target},
                ).scalars().all()
                for name in idx:
                    conn.execute(text(f'DROP INDEX "{name}"'))
        print(f"🔄 Đã đổi tên bảng {table} (cấu trúc cũ) -> {target}")


# Cột được thêm sau khi bảng đã tồn tại: (bảng, cột, kiểu SQL Server, kiểu SQLite). Luôn cho phép NULL.
ADDED_COLUMNS = [
    ("violations", "subject_crop_image", "NVARCHAR(255)", "VARCHAR(255)"),
]


def add_missing_columns(engine: Engine) -> None:
    """Thêm các cột mới (ADDED_COLUMNS) vào bảng đã có dữ liệu, không ảnh hưởng dữ liệu hiện có."""
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    for table, column, mssql_type, sqlite_type in ADDED_COLUMNS:
        if table not in tables or column in {c["name"] for c in insp.get_columns(table)}:
            continue
        col_type = mssql_type if engine.dialect.name == "mssql" else sqlite_type
        with engine.begin() as conn:
            if engine.dialect.name == "mssql":
                conn.execute(text(f"ALTER TABLE [dbo].[{table}] ADD [{column}] {col_type} NULL"))
            else:
                conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type}'))
        print(f"🆕 Đã thêm cột {table}.{column}")


TASK_STATUSES = ("pending", "processing", "completed", "failed", "cancelled")


def upgrade_task_status_constraint(engine: Engine) -> None:
    """Database cũ chỉ cho 4 trạng thái tác vụ -> thêm 'cancelled' (huỷ phân tích). Không ảnh hưởng dữ liệu."""
    if engine.dialect.name != "mssql":
        return   # SQLite: ràng buộc được tạo từ model khi tạo bảng mới
    with engine.begin() as conn:
        definition = conn.execute(text(
            "SELECT definition FROM sys.check_constraints WHERE name = 'ck_analysis_tasks_status'"
        )).scalar()
        if definition is not None and "cancelled" in definition:
            return
        if definition is not None:
            conn.execute(text("ALTER TABLE [dbo].[analysis_tasks] DROP CONSTRAINT [ck_analysis_tasks_status]"))
        values = ", ".join(f"'{s}'" for s in TASK_STATUSES)
        conn.execute(text(f"ALTER TABLE [dbo].[analysis_tasks] ADD CONSTRAINT [ck_analysis_tasks_status] CHECK ([status] IN ({values}))"))
    print("🆕 Đã thêm trạng thái 'cancelled' cho analysis_tasks")


def _find_legacy_table(engine: Engine | Connection, prefix: str = LEGACY_TABLE) -> Optional[str]:
    names = sorted(t for t in inspect(engine).get_table_names() if t.startswith(prefix))
    return names[0] if names else None


def _legacy_task_info(db: Session, task_id: str) -> dict:
    """Thông tin bổ sung từ bảng analysis_tasks_legacy (nếu có)."""
    table = _find_legacy_table(db.get_bind(), LEGACY_TASKS_TABLE)
    if not table:
        return {}
    row = db.execute(text(f'SELECT * FROM "{table}" WHERE id = :id'), {"id": task_id}).mappings().first()
    return dict(row) if row else {}


def _as_dt(value, utc: bool = False) -> Optional[datetime.datetime]:
    """
    SELECT thô trên SQLite trả datetime dạng chuỗi ISO; SQL Server trả sẵn datetime.
    utc=True: giá trị cũ được ghi bằng datetime.utcnow -> đổi sang giờ máy chủ (VN = UTC+7).
    """
    if value is None:
        return None
    if not isinstance(value, datetime.datetime):
        try:
            value = datetime.datetime.fromisoformat(str(value))
        except ValueError:
            return None
    if utc:
        value = value.replace(tzinfo=datetime.timezone.utc).astimezone().replace(tzinfo=None)
    return value


def _parse_video_time(value: Optional[str]) -> Optional[float]:
    m = re.match(r"^\s*(\d+):(\d+(?:\.\d+)?)s?\s*$", value or "")
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def _parse_occurred_at(row) -> datetime.datetime:
    try:
        return datetime.datetime.strptime(f"{row.get('date_str')} {row.get('time_str')}", "%d/%m/%Y %H:%M:%S")
    except (TypeError, ValueError):
        return _as_dt(row.get("timestamp"), utc=True) or _as_dt(row.get("created_at"), utc=True) or datetime.datetime.now()


def _video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return {}
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = fps if 0 < fps <= 120 else None
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
            "fps": fps,
            "total_frames": frames if frames > 0 else None,
            "duration_sec": round(frames / fps, 2) if fps and frames > 0 else None,
        }
    finally:
        cap.release()


def _get_or_create_upload_task(db: Session, task_id: str) -> AnalysisTask:
    """Tạo lại video + task từ file upload/output còn lưu trên đĩa (task cũ chỉ lưu trong RAM)."""
    task = db.get(AnalysisTask, task_id)
    if task:
        return task

    uploads = glob.glob(os.path.join(UPLOAD_DIR, f"input_{task_id}_*"))
    if uploads:
        stored = uploads[0]
        original = os.path.basename(stored)[len(f"input_{task_id}_"):]
        meta = _video_meta(stored)
        size_mb = round(os.path.getsize(stored) / (1024 * 1024), 2)
        uploaded_at = datetime.datetime.fromtimestamp(os.path.getmtime(stored))
    else:
        stored, original, meta, size_mb = "", f"(không còn file) {task_id}", {}, None
        uploaded_at = datetime.datetime.now()

    video = Video(
        original_name=original, stored_path=stored, file_size_mb=size_mb,
        location_id=None,  # dữ liệu cũ không lưu địa điểm -> người dùng tự gán sau nếu cần
        uploaded_at=uploaded_at, **meta,
    )
    db.add(video)
    db.flush()

    out_file = os.path.join(OUTPUT_DIR, f"analyzed_{task_id}.mp4")
    out_ok = os.path.exists(out_file) and bool(_video_meta(out_file))
    old = _legacy_task_info(db, task_id)
    old_created = _as_dt(old.get("created_at"), utc=True)
    task = AnalysisTask(
        id=task_id, video_id=video.id,
        status="completed" if out_ok else "failed",
        progress=100.0 if out_ok else 0.0,
        total_frames=meta.get("total_frames") or 0,
        fps=meta.get("fps"),
        vehicles_scanned=old.get("vehicles_scanned") or 0,
        legitimate_turns=old.get("legitimate_turns") or 0,
        processing_time_sec=old.get("processing_time_seconds"),
        output_video_path=f"/static/outputs/analyzed_{task_id}.mp4" if out_ok else None,
        error_message=None if out_ok else "Dữ liệu cũ: video kết quả không hoàn chỉnh (task bị gián đoạn)",
        created_at=old_created or uploaded_at,
        finished_at=(old_created or uploaded_at) if out_ok else None,
    )
    db.add(task)
    db.flush()
    return task


def migrate_legacy_rows(engine: Engine, db: Session) -> int:
    legacy = _find_legacy_table(engine)
    if not legacy or db.query(Violation.id).first():
        return 0  # không có dữ liệu cũ, hoặc đã migrate rồi

    rows = db.execute(text(f'SELECT * FROM "{legacy}" ORDER BY id')).mappings().all()
    count = skipped = 0
    for r in rows:
        r = dict(r)
        # Chỉ chuyển hồ sơ sinh ra từ một lần phân tích thật (ảnh bằng chứng evid_<task>_<track>.jpg).
        # Hồ sơ demo/seed cũ (ảnh vp_00X.jpg) bị bỏ qua - DB chỉ chứa dữ liệu thật.
        m = re.search(r"/evid_([0-9a-f]+)_(\d+)\.jpg$", r.get("evidence_image") or "")
        if not m:
            skipped += 1
            continue
        task = _get_or_create_upload_task(db, m.group(1))
        track_id = int(m.group(2))

        raw_type = r.get("violation_type")
        raw_vehicle = r.get("vehicle_type")
        color = r.get("vehicle_color")
        vsec = _parse_video_time(r.get("video_time"))

        db.add(Violation(
            violation_code=r["violation_code"],
            task_id=task.id,
            violation_type_id=get_violation_type(db, classify_reason(raw_type)).id,
            track_id=track_id,
            license_plate=r["license_plate"],
            vehicle_type=normalize_vehicle_type(raw_vehicle) if raw_vehicle and raw_vehicle.isupper() else raw_vehicle,
            vehicle_color=None if color in (None, "", "Tự động nhận diện") else color,
            ai_reason=raw_type if raw_type and raw_type.startswith("Vi pham") else None,
            frame_index=round(vsec * task.fps) if vsec is not None and task.fps else None,
            video_time_sec=vsec,
            occurred_at=_parse_occurred_at(r),
            confidence_overall=r.get("confidence_overall"),
            confidence_vehicle=r.get("confidence_vehicle"),
            confidence_light=r.get("confidence_light"),
            confidence_plate=r.get("confidence_plate"),
            confidence_violation=r.get("confidence_violation"),
            status=r.get("status") or "Chưa xử lý",
            evidence_image=r.get("evidence_image"),
            plate_crop_image=r.get("plate_crop_image"),
            created_at=_as_dt(r.get("created_at"), utc=True) or datetime.datetime.now(),
        ))
        db.flush()
        task.violations_detected = (task.violations_detected or 0) + 1
        count += 1

    db.commit()
    print(f"✅ Đã chuyển {count} hồ sơ từ bảng {legacy} sang cấu trúc mới (bỏ qua {skipped} hồ sơ demo)")
    return count


def mark_interrupted_tasks(db: Session) -> None:
    """Task đang pending/processing lúc server tắt sẽ không bao giờ chạy tiếp -> đánh dấu failed."""
    stale = db.query(AnalysisTask).filter(AnalysisTask.status.in_(["pending", "processing"])).all()
    for t in stale:
        t.status = "failed"
        t.error_message = "Server đã khởi động lại khi tác vụ đang chạy, vui lòng phân tích lại"
        t.finished_at = datetime.datetime.now()
    if stale:
        db.commit()
        print(f"⚠️ Đánh dấu {len(stale)} tác vụ bị gián đoạn là failed")
