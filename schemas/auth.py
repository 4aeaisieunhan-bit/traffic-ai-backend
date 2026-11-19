# schemas/auth.py
import datetime
import re
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

# Kiểm tra định dạng email cơ bản (không cần cài thêm thư viện email-validator)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _normalize_email(v: str) -> str:
    v = (v or "").strip().lower()
    if not EMAIL_RE.match(v):
        raise ValueError("Email không hợp lệ")
    return v


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=6, max_length=128)

    @field_validator("full_name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("Họ tên không được để trống")
        return v

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _normalize_email(v)


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=1, max_length=128)
    remember: bool = False   # True: phiên 30 ngày, False: 12 giờ

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return (v or "").strip().lower()


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    created_at: datetime.datetime
    is_active: bool = True   # False: tài khoản mới đăng ký đang chờ quản trị viên kích hoạt, hoặc đã bị khóa


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=6, max_length=128)


# ---- Quản lý tài khoản (chỉ admin) ----
class UserAdminResponse(UserResponse):
    active_sessions: int = 0


class UserAdminUpdate(BaseModel):
    """PATCH: chỉ gửi trường cần đổi."""
    is_active: Optional[bool] = None
    role: Optional[Literal["admin", "officer"]] = None


class ResetPasswordResponse(BaseModel):
    temporary_password: str   # chỉ hiển thị 1 lần cho admin để báo người dùng


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime.datetime
    user: UserResponse
