# api/analyze.py
# Upload video, chạy pipeline AI nền và trả kết quả. Mọi trạng thái task được lưu DB (bảng analysis_tasks).
import datetime
import os
import sys
import threading
import time
import uuid
from typing import Optional

import cv2
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, joinedload

from database.connection import SessionLocal, get_db
from database.lookups import (
    classify_reason, get_violation_type,
    next_violation_code, normalize_vehicle_type, violation_query,
)
from database.models import AnalysisTask, Location, TaskModelStat, User, Video, Violation
from api.deps import get_current_user, require_admin
from api.cleanup import delete_task_results, delete_video_if_orphan
from schemas.violation import violation_to_response

router = APIRouter(prefix="/api/v1/analyze", tags=["Analyze Video AI"])

# Bắt buộc đăng nhập qua header Bearer (gắn vào từng endpoint).
# Riêng /video/output do middleware protect_media trong server.py kiểm tra bằng cookie (thẻ <video> không gửi header).
_AUTH = [Depends(get_current_user)]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
EVIDENCE_DIR = os.path.join(BASE_DIR, "static", "evidence")
for _d in (UPLOAD_DIR, OUTPUT_DIR, EVIDENCE_DIR):
    os.makedirs(_d, exist_ok=True)

# Các model YOLO dùng chung (biến global) nên chỉ cho phép 1 video được phân tích tại một thời điểm.
# Video upload sau sẽ ở trạng thái "pending" cho tới khi video trước xử lý xong.
PIPELINE_LOCK = threading.Lock()
# Yêu cầu huỷ các task đang chạy: vòng lặp phân tích kiểm tra mỗi khung hình (rẻ, chỉ tra set trong RAM)
CANCEL_REQUESTS: set[str] = set()


class TaskCancelled(Exception):
    """Người dùng huỷ phân tích giữa chừng."""

ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DEFAULT_VIDEO_CANDIDATES = ["21s_vidnew.mov", "video 18s.mp4"]

# Thông tin mô tả từng mô hình: (tên hiển thị, file weights, module chứa biến `model`)
from utils.pipeline import MODEL_INFO  # noqa: E402  (khai báo duy nhất, dùng chung với main.py)


def _model_classes(module_name: str) -> list:
    """Tên class lấy trực tiếp từ model YOLO nếu đã được nạp (không ép nạp model chỉ để đọc tên)."""
    mod = sys.modules.get(module_name)
    names = getattr(getattr(mod, "model", None), "names", None) or {}
    return list(names.values()) if isinstance(names, dict) else list(names)


def _video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return {}
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if 0 < raw_fps <= 120 else 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
            "fps": round(fps, 2),
            "total_frames": frames,
            "duration_sec": round(frames / fps, 2) if frames > 0 else 0.0,
        }
    finally:
        cap.release()


def _plate_confidence(sidebar, track_id: int, plate_text: str) -> Optional[float]:
    """Điểm OCR trung bình của các lượt đọc ra đúng biển số đã chốt (%)."""
    track = getattr(sidebar, "vote_tracker", None) and sidebar.vote_tracker.tracks.get(track_id)
    votes = (track or {}).get("votes", {}).get(plate_text)
    return round(sum(votes) / len(votes) * 100, 1) if votes else None


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE NỀN
# ─────────────────────────────────────────────────────────────────────────────
def process_video_task(task_id: str):
    with PIPELINE_LOCK:
        _run_pipeline(task_id)


def _run_pipeline(task_id: str):
    db = SessionLocal()
    try:
        task = db.get(AnalysisTask, task_id)
        if task is None or task.status != "pending":
            return   # đã bị huỷ khi còn trong hàng đợi
        video = task.video
        task.status = "processing"
        task.started_at = datetime.datetime.now()
        db.commit()
        _execute(db, task, video)
    except TaskCancelled:
        db.rollback()
        task = db.get(AnalysisTask, task_id)
        if task:
            delete_task_results(db, task)          # kết quả dở dang không đầy đủ -> bỏ
            task.status = "cancelled"
            task.error_message = "Đã huỷ bởi người dùng"
            task.finished_at = datetime.datetime.now()
            db.commit()
    except Exception as e:
        db.rollback()
        task = db.get(AnalysisTask, task_id)
        if task:
            task.status = "failed"
            task.error_message = str(e)[:1000]
            task.finished_at = datetime.datetime.now()
            db.commit()
    finally:
        CANCEL_REQUESTS.discard(task_id)
        db.close()


def _execute(db: Session, task: AnalysisTask, video: Video):
    """Chạy pipeline dùng chung (utils/pipeline.py) trên video; lưu vi phạm + ảnh bằng chứng vào DB."""
    from utils.draw_detections import NO_HELMET_COLOR
    from utils.pipeline import TrafficPipeline

    start_time = time.time()
    cap = cv2.VideoCapture(video.stored_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở file video {video.original_name}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = raw_fps if 0 < raw_fps <= 120 else 30.0
    task.total_frames = total_frames
    task.fps = round(fps, 2)
    db.commit()

    pipeline = TrafficPipeline(fps)
    output_filename = f"analyzed_{task.id}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    # Ưu tiên H.264 (avc1) để trình duyệt phát được; fallback mp4v nếu OpenCV không hỗ trợ
    out = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*"avc1"), fps, pipeline.output_size)
    if not out.isOpened():
        out = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, pipeline.output_size)
    if not out.isOpened():
        raise RuntimeError("Không thể khởi tạo VideoWriter để ghi video kết quả")

    base_time = video.recorded_at or video.uploaded_at
    frame_idx = 0
    new_count = 0
    saved = []   # [(item sidebar, dòng Violation)] để cập nhật biển số khi xử lý xong
    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                break
            frame_idx += 1
            result = pipeline.process(raw, frame_idx)

            # Mỗi vi phạm mới -> lưu ảnh bằng chứng + 1 dòng violations
            for item in result.new_violations:
                new_count += 1
                saved.append((item, _save_violation(db, task, pipeline, item, result.evidence_frame, frame_idx, fps,
                                                    base_time, NO_HELMET_COLOR)))
                task.violations_detected = new_count
                db.commit()

            out.write(result.canvas)

            if task.id in CANCEL_REQUESTS:
                raise TaskCancelled()
            # Cập nhật tiến độ mỗi 5 frame để giao diện không trông như bị treo
            if frame_idx % 5 == 0:
                task.current_frame = frame_idx
                task.progress = round(frame_idx / total_frames * 100, 1) if total_frames > 0 else 0.0
                db.commit()
    finally:
        cap.release()
        out.release()

    _refresh_saved_plates(pipeline, saved)

    for key, st in pipeline.model_stats().items():
        db.add(TaskModelStat(task_id=task.id, model_name=key, weight_file=st["weight"],
                             detections=st["detections"], avg_confidence=st["avg_confidence"]))

    task.status = "completed"
    task.progress = 100.0
    task.current_frame = frame_idx
    task.total_frames = frame_idx
    task.violations_detected = new_count
    task.vehicles_scanned = len(pipeline.tracker.vehicles)
    task.legitimate_turns = pipeline.tracker.legitimate_turn_count
    task.output_video_path = f"/static/outputs/{output_filename}"
    task.processing_time_sec = round(time.time() - start_time, 2)
    task.finished_at = datetime.datetime.now()
    db.commit()


def _save_violation(db: Session, task: AnalysisTask, pipeline, item: dict, evidence_frame, frame_idx: int,
                    fps: float, base_time: datetime.datetime, helmet_color) -> Violation:
    """Ghi ảnh bằng chứng / biển số / cận cảnh và thêm 1 dòng violations (chưa commit)."""
    tid = item["track_id"]
    is_helmet = item.get("kind") == "NO_HELMET"
    suffix = "_helmet" if is_helmet else ""   # 1 xe có thể có cả 2 lỗi -> không ghi đè ảnh
    evidence_fname = f"evid_{task.id}_{tid}{suffix}.jpg"
    evidence = evidence_frame
    if is_helmet and item.get("highlight_bbox") is not None and evidence_frame is not None:
        # Khoanh xe máy vi phạm trên ảnh bằng chứng (cùng màu hồng của khung "không đội mũ")
        evidence = evidence_frame.copy()
        hx1, hy1, hx2, hy2 = item["highlight_bbox"]
        cv2.rectangle(evidence, (hx1 - 4, hy1 - 4), (hx2 + 4, hy2 + 4), helmet_color, 3)
    if evidence is not None:
        cv2.imwrite(os.path.join(EVIDENCE_DIR, evidence_fname), evidence)
    plate_fname = None
    if item.get("plate_crop") is not None and item["plate_crop"].size > 0:
        plate_fname = f"plate_{task.id}_{tid}{suffix}.jpg"
        cv2.imwrite(os.path.join(EVIDENCE_DIR, plate_fname), item["plate_crop"])
    subject_fname = None
    if item.get("subject_crop") is not None and item["subject_crop"].size > 0:
        subject_fname = f"subject_{task.id}_{tid}{suffix}.jpg"
        cv2.imwrite(os.path.join(EVIDENCE_DIR, subject_fname), item["subject_crop"])

    vsec = round(item.get("frame_idx", frame_idx) / fps, 2)
    conf_vehicle = pipeline.avg_conf_pct("Vehicle")
    conf_light = None if is_helmet else pipeline.avg_conf_pct("Traffic Light")
    conf_plate = _plate_confidence(pipeline.sidebar, tid, item["plate_text"])
    nh = pipeline.no_helmet_conf.get(tid) if is_helmet else None
    conf_extra = round(nh * 100, 1) if nh is not None else None
    available = [c for c in (conf_vehicle, conf_light, conf_plate, conf_extra) if c is not None]

    violation = Violation(
        violation_code=next_violation_code(db),
        task_id=task.id,
        violation_type_id=get_violation_type(db, classify_reason(item.get("reason"))).id,
        track_id=tid,
        license_plate=item["plate_text"],
        vehicle_type=normalize_vehicle_type(item.get("cls_name")),
        ai_reason=item.get("reason"),
        frame_index=item.get("frame_idx", frame_idx),
        video_time_sec=vsec,
        occurred_at=base_time + datetime.timedelta(seconds=vsec),
        confidence_overall=round(sum(available) / len(available), 1) if available else None,
        confidence_vehicle=conf_vehicle,
        confidence_light=conf_light,
        confidence_plate=conf_plate,
        # Không đội mũ: độ tin cậy TB các lần thấy no_helmet; vượt đèn: thuật toán quỹ đạo chưa có điểm
        confidence_violation=conf_extra,
        status="Chưa xử lý",
        evidence_image=f"/static/evidence/{evidence_fname}",
        plate_crop_image=f"/static/evidence/{plate_fname}" if plate_fname else None,
        subject_crop_image=f"/static/evidence/{subject_fname}" if subject_fname else None,
    )
    db.add(violation)
    item["plate_updated"] = False   # đã lưu đúng biển số hiện tại
    return violation


def _refresh_saved_plates(pipeline, saved) -> None:
    """
    Vi phạm được lưu ngay lúc kết luận, khi xe có thể chưa lộ biển số (VD lỗi không đội mũ kết luận lúc xe còn
    ở mép khung hình). Sidebar tiếp tục bình chọn biển số -> xử lý xong video thì ghi lại ảnh + chữ biển số mới
    cho các vi phạm có thay đổi (chưa commit).
    """
    for item, violation in saved:
        if not item.get("plate_updated"):
            continue
        crop = item.get("plate_crop")
        if crop is not None and crop.size > 0:
            suffix = "_helmet" if item.get("kind") == "NO_HELMET" else ""
            plate_fname = f"plate_{violation.task_id}_{item['track_id']}{suffix}.jpg"
            cv2.imwrite(os.path.join(EVIDENCE_DIR, plate_fname), crop)
            violation.plate_crop_image = f"/static/evidence/{plate_fname}"
        violation.license_plate = item["plate_text"]
        violation.confidence_plate = _plate_confidence(pipeline.sidebar, item["track_id"], item["plate_text"])
        parts = [c for c in (violation.confidence_vehicle, violation.confidence_light, violation.confidence_plate,
                             violation.confidence_violation) if c is not None]
        violation.confidence_overall = round(sum(parts) / len(parts), 1) if parts else None
        item["plate_updated"] = False


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────
def _get_or_create_sample_video(db: Session, location: Location) -> Video:
    """Video mẫu có sẵn trên server; dùng lại bản ghi nếu đã có cùng file + cùng địa điểm."""
    for name in DEFAULT_VIDEO_CANDIDATES:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            video = (
                db.query(Video)
                .filter(Video.stored_path == path, Video.location_id == location.id)
                .first()
            )
            if video:
                return video
            video = Video(
                original_name=name, stored_path=path,
                file_size_mb=round(os.path.getsize(path) / (1024 * 1024), 2),
                location_id=location.id,
                **_video_meta(path),
            )
            db.add(video)
            db.flush()
            return video
    raise HTTPException(status_code=400, detail="Không tìm thấy file video mẫu đầu vào")


@router.post("/start")
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    location_id: Optional[int] = Form(None, description="ID địa điểm quay video (bắt buộc)"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Upload video và đưa vào hàng đợi phân tích.
    - BẮT BUỘC location_id (chọn từ /api/v1/locations) cho cả video upload lẫn video mẫu.
    - Không gửi file: dùng video mẫu có sẵn trên server.
    """
    task_id = str(uuid.uuid4())[:8]

    # Kiểm tra địa điểm TRƯỚC khi ghi file lên đĩa
    if location_id is None:
        raise HTTPException(status_code=400, detail="Vui lòng chọn địa điểm quay video (location_id)")
    location = db.get(Location, location_id)
    if location is None:
        raise HTTPException(status_code=400, detail=f"Địa điểm id={location_id} không tồn tại")

    if file and file.filename:
        # Chỉ lấy tên file (chặn path traversal) và kiểm tra định dạng
        input_name = os.path.basename(file.filename.replace("\\", "/"))
        ext = os.path.splitext(input_name)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Định dạng '{ext or 'không xác định'}' không được hỗ trợ. Chỉ chấp nhận: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}",
            )
        input_path = os.path.join(UPLOAD_DIR, f"input_{task_id}_{input_name.replace(' ', '_')}")

        # Ghi theo khối 1MB để không giữ cả video trong RAM
        with open(input_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
        await file.close()

        meta = _video_meta(input_path)
        if not meta or meta["total_frames"] <= 0:
            os.remove(input_path)
            raise HTTPException(status_code=400, detail="File tải lên không phải video hợp lệ hoặc bị hỏng")

        video = Video(
            original_name=input_name,
            stored_path=input_path,
            file_size_mb=round(os.path.getsize(input_path) / (1024 * 1024), 2),
            location_id=location.id,
            uploaded_by=user.id,
            **meta,
        )
        db.add(video)
        db.flush()
    else:
        video = _get_or_create_sample_video(db, location)
        if video.uploaded_by is None:
            video.uploaded_by = user.id
        input_name = video.original_name

    db.add(AnalysisTask(id=task_id, video_id=video.id, status="pending", total_frames=video.total_frames or 0, fps=video.fps))
    db.commit()

    background_tasks.add_task(process_video_task, task_id)

    return {
        "status": "success",
        "message": f"Đã khởi chạy phân tích video AI cho file '{input_name}'",
        "task_id": task_id,
        "video_id": video.id,
        "input_video_name": input_name,
        "location": video.location.name if video.location else None,
    }


def _task_summary(t: AnalysisTask) -> dict:
    return {
        "task_id": t.id,
        "video_id": t.video_id,
        "video_name": t.video.original_name if t.video else None,
        "location": t.video.location.name if t.video and t.video.location else None,
        "status": t.status,
        "progress": t.progress,
        "current_frame": t.current_frame,
        "total_frames": t.total_frames,
        "violations_detected": t.violations_detected,
        "vehicles_scanned": t.vehicles_scanned,
        "processing_time_seconds": t.processing_time_sec,
        "output_video_url": t.output_video_path,
        "created_at": t.created_at,
        "started_at": t.started_at,
        "finished_at": t.finished_at,
        "error": t.error_message,
    }


@router.get("/tasks", dependencies=_AUTH)
def list_tasks(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[str] = Query(None, alias="status"),
    db: Session = Depends(get_db),
):
    """Lịch sử / hàng đợi các lần phân tích (mới nhất trước), có phân trang: {total, items}."""
    q = db.query(AnalysisTask)
    if status_filter:
        q = q.filter(AnalysisTask.status == status_filter)
    total = q.count()
    tasks = (
        q.options(joinedload(AnalysisTask.video).joinedload(Video.location))
        .order_by(AnalysisTask.created_at.desc()).offset(offset).limit(limit).all()
    )
    return {"total": total, "items": [_task_summary(t) for t in tasks]}


@router.post("/tasks/{task_id}/cancel", dependencies=_AUTH)
def cancel_task(task_id: str, db: Session = Depends(get_db)):
    """Huỷ lần phân tích đang chờ / đang chạy. Kết quả dở dang (vi phạm, ảnh, video) bị loại bỏ."""
    task = db.get(AnalysisTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy lần phân tích")
    if task.status == "pending":
        task.status = "cancelled"
        task.error_message = "Đã huỷ bởi người dùng"
        task.finished_at = datetime.datetime.now()
        db.commit()
        return {"status": "cancelled"}
    if task.status == "processing":
        CANCEL_REQUESTS.add(task_id)                 # vòng lặp dừng ở khung hình kế tiếp
        return {"status": "cancelling"}
    raise HTTPException(status_code=409, detail="Lần phân tích này đã kết thúc, không thể huỷ")


@router.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: str, db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    """Xoá 1 lần phân tích cùng toàn bộ kết quả (vi phạm, ảnh bằng chứng, video kết quả).
    Video gốc bị xoá nếu không còn lần phân tích nào dùng (video mẫu hệ thống không bị xoá). Chỉ quản trị viên."""
    task = db.get(AnalysisTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy lần phân tích")
    if task.status in ("pending", "processing"):
        raise HTTPException(status_code=409, detail="Lần phân tích đang chạy — hãy huỷ trước khi xoá")
    video = task.video
    delete_task_results(db, task)
    db.delete(task)
    db.flush()
    delete_video_if_orphan(db, video)
    db.commit()


@router.get("/status/{task_id}", dependencies=_AUTH)
def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """Tiến độ và kết quả chi tiết của một lần phân tích."""
    task = db.query(AnalysisTask).options(joinedload(AnalysisTask.model_stats)).filter(AnalysisTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ phân tích")

    violations = violation_query(db).filter(Violation.task_id == task_id).order_by(Violation.id.asc()).all()
    duration = round(task.total_frames / task.fps, 2) if task.fps and task.total_frames else 0.0

    return {
        "task_id": task.id,
        "status": task.status,
        "progress": task.progress,
        "current_frame": task.current_frame,
        "total_frames": task.total_frames,
        "violations_detected": task.violations_detected,
        "vehicles_scanned": task.vehicles_scanned,
        "legitimate_turns": task.legitimate_turns,
        "fps": task.fps,
        "duration_seconds": duration,
        "processing_time_seconds": task.processing_time_sec,
        "output_video_url": task.output_video_path,
        "model_stats": {
            s.model_name: {"weight": s.weight_file, "detections": s.detections, "avg_confidence": s.avg_confidence}
            for s in task.model_stats
        },
        "violations": [violation_to_response(v).model_dump(mode="json") for v in violations],
        "error": task.error_message,
    }


def _latest_completed_task(db: Session) -> Optional[AnalysisTask]:
    return (
        db.query(AnalysisTask)
        .options(joinedload(AnalysisTask.video), joinedload(AnalysisTask.model_stats))
        .filter(AnalysisTask.status == "completed", AnalysisTask.output_video_path.isnot(None))
        .order_by(AnalysisTask.finished_at.desc())
        .first()
    )


def _output_file(task: Optional[AnalysisTask]) -> Optional[str]:
    """File video kết quả của task gần nhất trong DB (không fallback sang file ngoài DB như output.mp4)."""
    if task and task.output_video_path:
        path = os.path.join(BASE_DIR, task.output_video_path.lstrip("/").replace("/", os.sep))
        if os.path.exists(path):
            return path
    return None


@router.get("/output/details", dependencies=_AUTH)
def get_output_details(db: Session = Depends(get_db)):
    """Thông tin video kết quả gần nhất + tổng hợp vi phạm + thống kê mô hình AI thật của lần chạy đó."""
    task = _latest_completed_task(db)
    output_path = _output_file(task)
    meta = _video_meta(output_path) if output_path else {}

    violations = violation_query(db).order_by(Violation.id.asc()).all()
    type_counts = {}
    for v in violations:
        type_counts[v.violation_type.name] = type_counts.get(v.violation_type.name, 0) + 1

    stats_by_name = {s.model_name: s for s in task.model_stats} if task else {}
    model_stats = {}
    for key, (display, weight, module) in MODEL_INFO.items():
        s = stats_by_name.get(key)
        model_stats[key] = {
            "model_name": display,
            "weight_file": weight,
            "detections": s.detections if s else 0,
            "avg_confidence": s.avg_confidence if s else None,
            "classes": _model_classes(module),
        }

    return {
        "status": "success",
        "latest_task": _task_summary(task) if task else None,
        "video_info": {
            "file_name": os.path.basename(output_path) if output_path else None,
            "source_video": task.video.original_name if task else None,
            "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2) if output_path else 0.0,
            "resolution": f"{meta['width']}x{meta['height']}" if meta else None,
            "width": meta.get("width"),
            "height": meta.get("height"),
            "fps": meta.get("fps"),
            "total_frames": meta.get("total_frames", 0),
            "duration_seconds": meta.get("duration_sec", 0.0),
            "video_stream_url": "/api/v1/analyze/video/output",
            "static_video_url": task.output_video_path if task else None,
            "is_ready": bool(meta),
        },
        "detection_summary": {
            "vehicles_scanned": task.vehicles_scanned if task else 0,
            "legitimate_turns": task.legitimate_turns if task else 0,
            "violations_detected": len(violations),
            "unprocessed_count": sum(1 for v in violations if v.status == "Chưa xử lý"),
            "confirmed_count": sum(1 for v in violations if v.status == "Đã xác nhận"),
            "rejected_count": sum(1 for v in violations if v.status == "Từ chối"),
            "type_breakdown": type_counts,
        },
        "model_stats": model_stats,
        "violations": [violation_to_response(v).model_dump(mode="json") for v in violations],
    }


@router.get("/summary/latest", dependencies=_AUTH)
def get_latest_analysis_summary(db: Session = Depends(get_db)):
    """Tóm tắt video kết quả gần nhất."""
    details = get_output_details(db)
    v_info, d_sum = details["video_info"], details["detection_summary"]
    return {
        "video_name": v_info["file_name"],
        "source_video": v_info["source_video"],
        "video_size_mb": v_info["file_size_mb"],
        "resolution": v_info["resolution"],
        "total_frames": v_info["total_frames"],
        "fps": v_info["fps"],
        "duration_seconds": v_info["duration_seconds"],
        "vehicles_scanned": d_sum["vehicles_scanned"],
        "legitimate_turns": d_sum["legitimate_turns"],
        "violations_count": d_sum["violations_detected"],
        "video_stream_url": v_info["video_stream_url"],
        "static_video_url": v_info["static_video_url"],
        "violations": details["violations"],
    }


@router.get("/video/output")
def stream_output_video(request: Request, db: Session = Depends(get_db)):
    """Phát video kết quả gần nhất, hỗ trợ HTTP 206 (Range) để tua."""
    video_path = _output_file(_latest_completed_task(db))
    if not video_path:
        raise HTTPException(status_code=404, detail="Không tìm thấy file video kết quả")

    file_size = os.path.getsize(video_path)
    range_header = request.headers.get("range")

    if not range_header:
        def iter_file():
            with open(video_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk

        return StreamingResponse(
            iter_file(), media_type="video/mp4",
            headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"},
        )

    parts = range_header.replace("bytes=", "").split("-")
    start = int(parts[0]) if parts[0] else 0
    end = min(int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1, file_size - 1)
    length = end - start + 1
    with open(video_path, "rb") as f:
        f.seek(start)
        data = f.read(length)
    return Response(
        content=data, status_code=status.HTTP_206_PARTIAL_CONTENT, media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(data)),
        },
    )
