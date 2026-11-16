# api/auth.py
# Đăng ký / đăng nhập / đăng xuất / thông tin tài khoản / đổi mật khẩu
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from api.deps import (
    MEDIA_COOKIE, create_session, get_current_session, get_current_user, login_limiter,
)
from database.connection import get_db
from database.lookups import hash_password, verify_password
from database.models import User, UserSession
from schemas.auth import (
    ChangePasswordRequest, LoginRequest, LoginResponse, RegisterRequest, UserResponse,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])

# Băm sẵn 1 mật khẩu giả: khi email không tồn tại vẫn tốn thời gian băm như bình thường,
# tránh đoán được email nào đã đăng ký qua thời gian phản hồi.
_DUMMY_HASH = hash_password("dummy-password-for-timing")
_LOGIN_FAILED = "Email hoặc mật khẩu không đúng"


def _user_response(u: User) -> UserResponse:
    return UserResponse(id=u.id, email=u.email, full_name=u.full_name, role=u.role, created_at=u.created_at)


# Frontend và backend KHÁC tên miền (VD: frontend trên Vercel, backend qua Cloudflare Tunnel):
# đặt CROSS_SITE_COOKIE=1 trong .env -> cookie ảnh/video dùng SameSite=None + Secure (bắt buộc HTTPS) + Partitioned
# (trình duyệt mới vẫn gửi cookie "bên thứ ba" dạng phân vùng). Chạy chung máy / cùng tên miền: giữ mặc định (Lax).
CROSS_SITE_COOKIE = os.getenv("CROSS_SITE_COOKIE", "0").strip().lower() in ("1", "true", "yes")
_COOKIE_SAMESITE: Literal["lax", "none"] = "none" if CROSS_SITE_COOKIE else "lax"
_COOKIE_SECURE: bool = CROSS_SITE_COOKIE          # SameSite=None bắt buộc kèm Secure (HTTPS)
_COOKIE_PARTITIONED: bool = CROSS_SITE_COOKIE


def _set_media_cookie(response: Response, token: str, remember: bool, max_age: int) -> None:
    response.set_cookie(
        MEDIA_COOKIE, token,
        max_age=max_age if remember else None,   # không ghi nhớ -> cookie mất khi đóng trình duyệt
        httponly=True,                           # JavaScript không đọc được
        path="/",
        samesite=_COOKIE_SAMESITE,
        secure=_COOKIE_SECURE,
        partitioned=_COOKIE_PARTITIONED,
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    """Đăng ký tài khoản cán bộ (role 'officer'). Tài khoản admin không tự đăng ký được."""
    if db.query(User.id).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email này đã được đăng ký")
    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        role="officer",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _user_response(user)


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    keys = login_limiter.keys(payload.email, request)
    login_limiter.check(keys)

    user = db.query(User).filter(User.email == payload.email).first()
    password_ok = verify_password(payload.password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not password_ok:
        remaining = login_limiter.record_failure(keys)
        detail = _LOGIN_FAILED
        if 0 < remaining <= 2:
            detail += f". Còn {remaining} lần thử trước khi bị tạm khóa 15 phút."
        elif remaining == 0:
            detail = "Đăng nhập sai quá nhiều lần. Vui lòng thử lại sau 15 phút."
        raise HTTPException(status_code=401, detail=detail)
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Tài khoản đã bị khóa, vui lòng liên hệ quản trị viên")

    login_limiter.reset(keys[0])   # đăng nhập đúng -> xóa đếm sai theo email
    token, session = create_session(db, user, payload.remember)
    max_age = int((session.expires_at - session.created_at).total_seconds())
    _set_media_cookie(response, token, payload.remember, max_age)
    return LoginResponse(access_token=token, expires_at=session.expires_at, user=_user_response(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(session: UserSession = Depends(get_current_session), db: Session = Depends(get_db)):
    """Xóa phiên hiện tại -> token hết hiệu lực ngay lập tức."""
    db.delete(session)
    db.commit()
    resp = Response(status_code=status.HTTP_204_NO_CONTENT)
    resp.delete_cookie(   # cùng thuộc tính lúc đặt, để trình duyệt khớp đúng cookie
        MEDIA_COOKIE, path="/", httponly=True,
        samesite=_COOKIE_SAMESITE, secure=_COOKIE_SECURE, partitioned=_COOKIE_PARTITIONED,
    )
    return resp


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return _user_response(user)


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    payload: ChangePasswordRequest,
    session: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
):
    """Đổi mật khẩu; các phiên đăng nhập ở thiết bị khác bị đăng xuất, phiên hiện tại giữ nguyên."""
    user = session.user
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Mật khẩu hiện tại không đúng")
    if payload.new_password == payload.current_password:
        raise HTTPException(status_code=400, detail="Mật khẩu mới phải khác mật khẩu hiện tại")

    user.password_hash = hash_password(payload.new_password)
    db.query(UserSession).filter(UserSession.user_id == user.id, UserSession.id != session.id).delete()
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
