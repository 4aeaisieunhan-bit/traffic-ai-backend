# database/seed_data.py
# Chỉ nạp DỮ LIỆU HỆ THỐNG bắt buộc (idempotent, chạy mỗi lần khởi động):
#   - violation_types: danh mục loại vi phạm mà thuật toán AI phân loại vào
#   - users: tài khoản admin đầu tiên
# KHÔNG có dữ liệu demo: địa điểm, video, hồ sơ vi phạm đều do người dùng tạo qua API.
from sqlalchemy.orm import Session

from database.models import User, ViolationType
from database.lookups import DEFAULT_ADMIN, VIOLATION_TYPE_SEED, get_default_admin, hash_password


def seed_lookups(db: Session) -> None:
    for code, name, desc in VIOLATION_TYPE_SEED:
        if not db.query(ViolationType).filter(ViolationType.code == code).first():
            db.add(ViolationType(code=code, name=name, description=desc))

    if get_default_admin(db) is None:
        db.add(User(
            email=DEFAULT_ADMIN["email"],
            password_hash=hash_password(DEFAULT_ADMIN["password"]),
            full_name=DEFAULT_ADMIN["full_name"],
            role=DEFAULT_ADMIN["role"],
        ))
    db.commit()
