# api/cleanup.py
# Xoá dữ liệu phân tích an toàn (dùng chung cho: huỷ phân tích, xoá lần phân tích, xoá vi phạm).
# Chỉ xoá file nằm TRONG thư mục static/ của backend -> đường dẫn lỗi trong DB không thể xoá nhầm file khác.
import os

from sqlalchemy.orm import Session

from database.models import AnalysisTask, Video, Violation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.realpath(os.path.join(BASE_DIR, "static"))
UPLOAD_DIR = os.path.realpath(os.path.join(STATIC_DIR, "uploads"))
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")


def _inside(path: str, folder: str) -> bool:
    real = os.path.realpath(path)
    return os.path.commonpath([real, folder]) == folder


def remove_static_file(url_or_path: str | None) -> bool:
    """'/static/evidence/x.jpg' hoặc đường dẫn tuyệt đối trong static/ -> xoá file. Trả True nếu đã xoá."""
    if not url_or_path:
        return False
    path = url_or_path
    if url_or_path.startswith("/static/"):
        path = os.path.join(BASE_DIR, url_or_path.lstrip("/").replace("/", os.sep))
    if not _inside(path, STATIC_DIR) or not os.path.isfile(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def delete_violation(db: Session, v: Violation) -> None:
    """Xoá 1 vi phạm + lịch sử duyệt (cascade) + ảnh bằng chứng / biển số / cận cảnh. Không commit."""
    for f in (v.evidence_image, v.plate_crop_image, v.subject_crop_image):
        remove_static_file(f)
    db.delete(v)


def delete_task_results(db: Session, task: AnalysisTask) -> int:
    """Xoá mọi kết quả của 1 lần phân tích (vi phạm, ảnh, video kết quả, thống kê mô hình). Giữ lại bản ghi task.
    Trả về số vi phạm đã xoá. Không commit."""
    violations = db.query(Violation).filter(Violation.task_id == task.id).all()
    for v in violations:
        delete_violation(db, v)
    for s in list(task.model_stats):
        db.delete(s)
    remove_static_file(task.output_video_path)
    remove_static_file(os.path.join(OUTPUT_DIR, f"analyzed_{task.id}.mp4"))   # video dở dang (task chưa xong)
    task.output_video_path = None
    task.violations_detected = 0
    return len(violations)


def delete_video_if_orphan(db: Session, video: Video | None) -> bool:
    """Xoá video gốc nếu không còn lần phân tích nào dùng. Chỉ xoá file người dùng upload (static/uploads),
    không bao giờ xoá video mẫu của hệ thống. Không commit."""
    if video is None:
        return False
    still_used = db.query(AnalysisTask.id).filter(AnalysisTask.video_id == video.id).count()
    if still_used:
        return False
    if video.stored_path and _inside(video.stored_path, UPLOAD_DIR):
        remove_static_file(video.stored_path)
    db.delete(video)
    return True
