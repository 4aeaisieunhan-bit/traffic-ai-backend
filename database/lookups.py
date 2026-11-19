# database/lookups.py
# Danh mục dữ liệu dùng chung + các hàm tiện ích truy vấn
import hashlib
import hmac
import os
from typing import Optional
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from database.models import (
    AnalysisTask, User, Video, Violation, ViolationType,
)
from utils.labels import VIOLATION_TYPES, classify_reason, fold_text, vehicle_label  # noqa: F401  (dùng lại ở module khác)

# ── Danh mục loại vi phạm (khai báo ở utils/labels.py) ───────────────────────
VIOLATION_TYPE_SEED = [(code, name, desc) for code, (name, desc) in VIOLATION_TYPES.items()]


def normalize_vehicle_type(raw: Optional[str]) -> Optional[str]:
    """Chuẩn hóa tên class YOLO / dữ liệu cũ: 'XE MAY', 'motorcycle' -> 'Xe máy'."""
    return vehicle_label(raw)


def get_violation_type(db: Session, code: str) -> ViolationType:
    vt = db.query(ViolationType).filter(ViolationType.code == code).first()
    if vt is None:
        raise RuntimeError(f"Thiếu loại vi phạm '{code}' trong bảng violation_types (chưa seed?)")
    return vt


# ── Mật khẩu (PBKDF2 có sẵn trong thư viện chuẩn, không cần cài thêm) ───────
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"pbkdf2_sha256$200000${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(digest.hex(), digest_hex)   # so sánh thời gian hằng
    except (ValueError, AttributeError):
        return False


# ── Tài khoản ───────────────────────────────────────────────────────────────
# Tài khoản quản trị đầu tiên: chỉ tạo khi hệ thống CHƯA có quản trị viên nào.
#   ADMIN_EMAIL / ADMIN_PASSWORD trong .env; không đặt ADMIN_PASSWORD -> sinh mật khẩu ngẫu nhiên, in 1 lần ra console.
ADMIN_EMAIL = (os.getenv("ADMIN_EMAIL") or "admin@gmail.com").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or ""
# Mật khẩu mặc định của các phiên bản cũ: còn dùng thì cảnh báo khi khởi động
LEGACY_DEFAULT_PASSWORD = "123456"

# Tài khoản tự đăng ký phải được quản trị viên kích hoạt mới đăng nhập được.
# AUTO_APPROVE_REGISTRATION=1: kích hoạt ngay (chỉ nên dùng khi chạy demo trên máy cá nhân).
AUTO_APPROVE_REGISTRATION = os.getenv("AUTO_APPROVE_REGISTRATION", "0").strip().lower() in ("1", "true", "yes")


def has_active_admin(db: Session) -> bool:
    return db.query(User.id).filter(User.role == "admin", User.is_active.is_(True)).first() is not None


# ── Truy vấn ────────────────────────────────────────────────────────────────
def violation_query(db: Session):
    """Query Violation kèm sẵn các quan hệ cần để build ViolationResponse (tránh N+1)."""
    return db.query(Violation).options(
        joinedload(Violation.violation_type),
        joinedload(Violation.task).joinedload(AnalysisTask.video).joinedload(Video.location),
    )


def next_violation_code(db: Session) -> str:
    """Sinh mã vi phạm mới không trùng với mã đã có."""
    next_num = (db.query(func.max(Violation.id)).scalar() or 0) + 1
    while True:
        code = f"VP-{next_num:06d}"
        if not db.query(Violation.id).filter(Violation.violation_code == code).first():
            return code
        next_num += 1
