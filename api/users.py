# api/users.py
# Quản lý tài khoản - CHỈ quản trị viên. Thay cho "Quên mật khẩu" qua email (hệ thống chưa có máy chủ mail):
# người dùng quên mật khẩu liên hệ admin -> admin đặt lại mật khẩu tạm.
import datetime
import secrets
import string
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.deps import require_admin
from database.connection import get_db
from database.lookups import hash_password
from database.models import User, UserSession
from schemas.auth import ResetPasswordResponse, UserAdminResponse, UserAdminUpdate

router = APIRouter(prefix="/api/v1/users", tags=["Users (admin)"])

# Bỏ các ký tự dễ nhầm (0/O, 1/l/I) để admin đọc mật khẩu tạm cho người dùng
_TEMP_ALPHABET = "".join(c for c in string.ascii_letters + string.digits if c not in "0O1lI")


def _active_sessions(db: Session) -> dict:
    now = datetime.datetime.now()
    return dict(
        db.query(UserSession.user_id, func.count(UserSession.id))
        .filter(UserSession.expires_at > now).group_by(UserSession.user_id).all()
    )


def _to_response(u: User, sessions: dict) -> UserAdminResponse:
    return UserAdminResponse(
        id=u.id, email=u.email, full_name=u.full_name, role=u.role, created_at=u.created_at,
        is_active=u.is_active, active_sessions=sessions.get(u.id, 0),
    )


def _get_or_404(db: Session, user_id: int) -> User:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản")
    return user


def _active_admin_count(db: Session) -> int:
    return db.query(func.count(User.id)).filter(User.role == "admin", User.is_active.is_(True)).scalar()


@router.get("", response_model=List[UserAdminResponse])
def list_users(db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    sessions = _active_sessions(db)
    return [_to_response(u, sessions) for u in db.query(User).order_by(User.created_at.desc()).all()]


@router.patch("/{user_id}", response_model=UserAdminResponse)
def update_user(
    user_id: int,
    payload: UserAdminUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Khóa / mở khóa tài khoản, đổi vai trò. Khóa tài khoản -> đăng xuất mọi phiên của tài khoản đó."""
    user = _get_or_404(db, user_id)

    if user.id == admin.id and (payload.is_active is False or payload.role == "officer"):
        raise HTTPException(status_code=400, detail="Không thể tự khóa hoặc tự bỏ quyền quản trị của chính mình")
    losing_admin = user.role == "admin" and user.is_active and (payload.is_active is False or payload.role == "officer")
    if losing_admin and _active_admin_count(db) <= 1:
        raise HTTPException(status_code=400, detail="Hệ thống phải còn ít nhất 1 quản trị viên đang hoạt động")

    if payload.role is not None:
        user.role = payload.role
    if payload.is_active is not None:
        user.is_active = payload.is_active
        if not payload.is_active:
            db.query(UserSession).filter(UserSession.user_id == user.id).delete()
    db.commit()
    db.refresh(user)
    return _to_response(user, _active_sessions(db))


@router.post("/{user_id}/reset-password", response_model=ResetPasswordResponse)
def reset_password(user_id: int, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    """Sinh mật khẩu tạm (hiển thị 1 lần), đăng xuất mọi phiên. Người dùng nên đổi lại sau khi đăng nhập."""
    user = _get_or_404(db, user_id)
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Hãy dùng chức năng Đổi mật khẩu cho tài khoản của chính mình")
    temp = "".join(secrets.choice(_TEMP_ALPHABET) for _ in range(10))
    user.password_hash = hash_password(temp)
    db.query(UserSession).filter(UserSession.user_id == user.id).delete()
    db.commit()
    return ResetPasswordResponse(temporary_password=temp)
