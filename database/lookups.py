# database/lookups.py
# Danh mục dữ liệu dùng chung + các hàm tiện ích truy vấn
import hashlib
import hmac
import os
import unicodedata
from typing import Optional
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from database.models import (
    AnalysisTask, User, Video, Violation, ViolationType,
)

# ── Danh mục loại vi phạm ────────────────────────────────────────────────────
VIOLATION_TYPE_SEED = [
    ("RED_LIGHT", "Vượt đèn đỏ", "Đi thẳng / rẽ trái / rẽ phải khi đèn đỏ"),
    ("STOP_LINE", "Không dừng đúng vạch khi đèn đỏ", "Vượt qua vạch dừng và dừng lại trong giao lộ khi đèn đỏ"),
    ("NO_HELMET", "Không đội mũ bảo hiểm", "Người điều khiển / ngồi sau xe máy không đội mũ bảo hiểm"),
]

DEFAULT_ADMIN = {
    "email": "admin@gmail.com",
    "password": "123456",   # Trùng tài khoản demo ở Frontend; nên đổi khi triển khai thật
    "full_name": "Admin",
    "role": "admin",
}


def fold_text(text: Optional[str]) -> str:
    """Bỏ dấu tiếng Việt + chữ thường: 'Vượt đèn đỏ' -> 'vuot den do'."""
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


def classify_reason(reason: Optional[str]) -> str:
    """
    Ánh xạ mô tả vi phạm (chuỗi tự do từ thuật toán quỹ đạo hoặc dữ liệu cũ) -> mã loại vi phạm.
    VD: 'Vi pham: Di thang khi den do' -> RED_LIGHT
        'Vi pham: Qua stopline va dung yen qua 5.0s khi den do' -> STOP_LINE
    """
    t = fold_text(reason)
    if "mu bao hiem" in t or "helmet" in t:
        return "NO_HELMET"
    if "dung yen" in t or "stopline" in t or "vach" in t:
        return "STOP_LINE"
    return "RED_LIGHT"


def normalize_vehicle_type(raw: Optional[str]) -> Optional[str]:
    """Chuẩn hóa tên class YOLO / dữ liệu cũ: 'XE MAY', 'motorcycle' -> 'Xe máy'."""
    if not raw:
        return None
    t = fold_text(raw)
    if t in ("xe", "unknown"):
        return None
    if "may" in t or "motor" in t:
        return "Xe máy"
    if "bus" in t or "khach" in t or "16 cho" in t:
        return "Xe khách"
    if "truck" in t or "tai" in t:
        return "Xe tải"
    if "o to" in t or "oto" in t or "car" in t or "sedan" in t:
        return "Ô tô"
    return raw.strip()


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


def get_default_admin(db: Session) -> Optional[User]:
    return db.query(User).filter(User.email == DEFAULT_ADMIN["email"]).first()


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
