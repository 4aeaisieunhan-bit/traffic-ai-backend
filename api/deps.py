# api/deps.py
# Xác thực: token "Bearer <token>" -> tra bảng user_sessions -> User hiện tại
import datetime
import hashlib
import secrets
import threading
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import User, UserSession

SESSION_TTL_DEFAULT = datetime.timedelta(hours=12)
SESSION_TTL_REMEMBER = datetime.timedelta(days=30)

# Cookie HttpOnly chỉ dùng để trình duyệt tải ảnh bằng chứng / video (<img>, <video> không gửi được header).
# API vẫn bắt buộc header Authorization -> cookie không mở ra lỗ hổng CSRF cho các thao tác ghi.
MEDIA_COOKIE = "traffic_ai_media"

_bearer = HTTPBearer(auto_error=False)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(db: Session, user: User, remember: bool) -> tuple[str, UserSession]:
    """Tạo phiên mới, trả về (token gốc - chỉ gửi cho client 1 lần, bản ghi phiên)."""
    now = datetime.datetime.now()
    # Dọn các phiên đã hết hạn của user này
    db.query(UserSession).filter(UserSession.user_id == user.id, UserSession.expires_at < now).delete()
    token = secrets.token_urlsafe(32)
    session = UserSession(
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=now + (SESSION_TTL_REMEMBER if remember else SESSION_TTL_DEFAULT),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return token, session


def find_valid_session(db: Session, token: Optional[str]) -> Optional[UserSession]:
    """Phiên còn hạn của tài khoản đang hoạt động, hoặc None."""
    if not token:
        return None
    session = db.query(UserSession).filter(UserSession.token_hash == hash_token(token)).first()
    if session is None or session.expires_at < datetime.datetime.now() or not session.user.is_active:
        return None
    return session


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_session(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> UserSession:
    if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
        raise _unauthorized("Vui lòng đăng nhập")
    session = find_valid_session(db, creds.credentials)
    if session is None:
        raise _unauthorized("Phiên đăng nhập không hợp lệ hoặc đã hết hạn, vui lòng đăng nhập lại")
    return session


def get_current_user(session: UserSession = Depends(get_current_session)) -> User:
    return session.user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Chỉ quản trị viên mới có quyền này")
    return user


# ─────────────────────────────────────────────────────────────────────────────
# Giới hạn đăng nhập sai: quá MAX_FAILURES lần trong WINDOW -> khóa LOCKOUT
# (theo email và theo IP). Lưu trong bộ nhớ: khởi động lại server sẽ reset.
# ─────────────────────────────────────────────────────────────────────────────
class LoginRateLimiter:
    MAX_FAILURES = 5        # theo email: chống dò mật khẩu 1 tài khoản
    MAX_FAILURES_IP = 20    # theo IP: cao hơn vì nhiều người có thể dùng chung mạng (NAT)
    WINDOW = datetime.timedelta(minutes=15)
    LOCKOUT = datetime.timedelta(minutes=15)

    def _limit(self, key: str) -> int:
        return self.MAX_FAILURES_IP if key.startswith("ip:") else self.MAX_FAILURES

    def __init__(self):
        self._failures: dict[str, list[datetime.datetime]] = {}
        self._locked_until: dict[str, datetime.datetime] = {}
        self._lock = threading.Lock()

    @staticmethod
    def keys(email: str, request: Request) -> list[str]:
        ip = request.client.host if request.client else "unknown"
        return [f"email:{email}", f"ip:{ip}"]

    def check(self, keys: list[str]) -> None:
        """Ném 429 nếu email hoặc IP đang bị tạm khóa."""
        now = datetime.datetime.now()
        with self._lock:
            until = max((self._locked_until.get(k, now) for k in keys), default=now)
        if until > now:
            minutes = max(1, int((until - now).total_seconds() // 60) + 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Đăng nhập sai quá nhiều lần. Vui lòng thử lại sau {minutes} phút.",
                headers={"Retry-After": str(int((until - now).total_seconds()) + 1)},
            )

    def record_failure(self, keys: list[str]) -> int:
        """Ghi 1 lần sai, trả về số lần còn lại trước khi bị khóa (theo email)."""
        now = datetime.datetime.now()
        remaining = self.MAX_FAILURES
        with self._lock:
            for k in keys:
                recent = [t for t in self._failures.get(k, []) if now - t < self.WINDOW] + [now]
                self._failures[k] = recent
                if len(recent) >= self._limit(k):
                    self._locked_until[k] = now + self.LOCKOUT
                    self._failures[k] = []
                if k.startswith("email:"):
                    remaining = max(0, self._limit(k) - len(recent))
        return remaining

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)


login_limiter = LoginRateLimiter()
