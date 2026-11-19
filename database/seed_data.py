# database/seed_data.py
# Chỉ nạp DỮ LIỆU HỆ THỐNG bắt buộc (idempotent, chạy mỗi lần khởi động):
#   - violation_types: danh mục loại vi phạm mà thuật toán AI phân loại vào
#   - users: tài khoản quản trị đầu tiên (chỉ khi hệ thống chưa có quản trị viên nào đang hoạt động)
# KHÔNG có dữ liệu demo: địa điểm, video, hồ sơ vi phạm đều do người dùng tạo qua API.
import secrets
import string

from sqlalchemy.orm import Session

from database.models import User, ViolationType
from database.lookups import (
    ADMIN_EMAIL, ADMIN_PASSWORD, LEGACY_DEFAULT_PASSWORD, VIOLATION_TYPE_SEED, has_active_admin, hash_password,
    verify_password,
)


def _random_password(length: int = 14) -> str:
    alphabet = "".join(c for c in string.ascii_letters + string.digits if c not in "0O1lI")
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _ensure_admin(db: Session) -> None:
    if has_active_admin(db):
        return
    password = ADMIN_PASSWORD or _random_password()
    user = db.query(User).filter(User.email == ADMIN_EMAIL).first()
    if user is None:
        db.add(User(email=ADMIN_EMAIL, password_hash=hash_password(password), full_name="Admin", role="admin"))
    else:   # email đã có (VD: bị hạ quyền / bị khóa) -> khôi phục làm quản trị viên với mật khẩu mới
        user.role, user.is_active, user.password_hash = "admin", True, hash_password(password)
    db.commit()
    print(f"🔑 Đã tạo tài khoản quản trị: {ADMIN_EMAIL}", flush=True)
    if not ADMIN_PASSWORD:
        print(f"   Mật khẩu tạm (chỉ hiện 1 lần): {password}  — đăng nhập rồi đổi ở trang Tài khoản,", flush=True)
        print("   hoặc đặt ADMIN_PASSWORD trong .env trước khi khởi động lần đầu.", flush=True)


def _warn_default_passwords(db: Session) -> None:
    for u in db.query(User).filter(User.role == "admin", User.is_active.is_(True)).all():
        if verify_password(LEGACY_DEFAULT_PASSWORD, u.password_hash):
            print(f"⚠️  Tài khoản quản trị {u.email} vẫn dùng mật khẩu mặc định '{LEGACY_DEFAULT_PASSWORD}'. "
                  "Hãy đổi ngay ở trang Tài khoản trước khi mở hệ thống ra mạng ngoài.", flush=True)


def seed_lookups(db: Session) -> None:
    for code, name, desc in VIOLATION_TYPE_SEED:
        if not db.query(ViolationType).filter(ViolationType.code == code).first():
            db.add(ViolationType(code=code, name=name, description=desc))
    db.commit()
    _ensure_admin(db)
    _warn_default_passwords(db)
