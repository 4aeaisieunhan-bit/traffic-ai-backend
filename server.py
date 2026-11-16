# server.py
# Điểm khởi chạy chính của Backend API (FastAPI) kết nối với Angular Frontend
import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from database.connection import engine, Base, SessionLocal
import database.models  # noqa: F401  (đăng ký toàn bộ bảng vào Base.metadata)
from database.seed_data import seed_lookups
from database.migrate import add_missing_columns, rename_legacy_table, migrate_legacy_rows, mark_interrupted_tasks, upgrade_task_status_constraint
from api.violations import router as violations_router
from api.analyze import router as analyze_router
from api.statistics import router as statistics_router
from api.locations import router as locations_router
from api.auth import router as auth_router
from api.users import router as users_router
from api.system import router as system_router, public_router as public_router
from api.deps import MEDIA_COOKIE, find_valid_session, get_current_user
from fastapi import Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

# 1. Nếu bảng violations còn cấu trúc cũ -> đổi tên thành violations_legacy (giữ nguyên dữ liệu)
rename_legacy_table(engine)

# 2. Tạo các bảng mới nếu chưa có (users, locations, videos, analysis_tasks, ...)
Base.metadata.create_all(bind=engine)
# 2b. Bảng đã có từ trước: thêm các cột mới (VD: violations.subject_crop_image)
add_missing_columns(engine)
upgrade_task_status_constraint(engine)

# 3. Dữ liệu hệ thống (loại vi phạm, admin) -> chuyển dữ liệu cũ (nếu có) -> dọn task bị gián đoạn
#    Không nạp dữ liệu demo: DB chỉ chứa dữ liệu thật do người dùng tạo.
db = SessionLocal()
try:
    seed_lookups(db)
    migrate_legacy_rows(engine, db)
    mark_interrupted_tasks(db)
finally:
    db.close()

# 3. Khởi tạo FastAPI App
app = FastAPI(
    title="Traffic AI Backend API",
    description="Hệ thống Backend RESTful API & AI Vision giám sát vi phạm giao thông thông minh",
    version="2.1.0",
)

# 4a. Bảo vệ file media (ảnh bằng chứng, video upload/kết quả): bắt buộc đăng nhập.
#     <img>/<video> không gửi được header Authorization -> chấp nhận cookie HttpOnly đặt lúc đăng nhập.
#     (Khai báo TRƯỚC CORS để CORS là lớp ngoài cùng, response 401 vẫn có header CORS.)
PROTECTED_MEDIA_PREFIXES = ("/static/", "/api/v1/analyze/video/output")


def _media_session_ok(token: str | None) -> bool:
    db = SessionLocal()
    try:
        return find_valid_session(db, token) is not None
    finally:
        db.close()


@app.middleware("http")
async def protect_media(request: Request, call_next):
    if request.method != "OPTIONS" and request.url.path.startswith(PROTECTED_MEDIA_PREFIXES):
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else request.cookies.get(MEDIA_COOKIE)
        if not await run_in_threadpool(_media_session_ok, token):
            return JSONResponse(status_code=401, content={"detail": "Vui lòng đăng nhập để xem tệp này"})
    return await call_next(request)


# 4b. CORS: Không dùng "*" vì có gửi cookie (allow_credentials).
#     - Cho phép mọi cổng trên máy local: localhost / 127.0.0.1 / [::1] (ng serve 4200, 4201, Live Server 5500...)
#     - Địa chỉ khác (IP mạng LAN, tên miền khi triển khai) thêm qua biến môi trường CORS_ORIGINS, cách nhau dấu phẩy.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=LOCAL_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4c. Ghi log khi trình duyệt bị chặn CORS -> biết ngay frontend đang mở ở địa chỉ nào
#     (khai báo SAU CORS nên là lớp ngoài cùng, thấy được cả request OPTIONS bị CORS từ chối).
@app.middleware("http")
async def log_rejected_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and origin not in CORS_ORIGINS and not re.match(LOCAL_ORIGIN_REGEX, origin):
        print(
            f"⚠️  CORS: chặn yêu cầu {request.method} {request.url.path} từ địa chỉ '{origin}'. "
            f"Hãy mở frontend tại http://localhost:4200 hoặc thêm địa chỉ này vào CORS_ORIGINS trong .env",
            flush=True,
        )
    return await call_next(request)

# 5. Mount thư mục tĩnh để phục vụ ảnh bằng chứng và video kết quả cho Frontend
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 6. Đăng ký các nhóm API Router
# Đăng ký / đăng nhập: công khai
app.include_router(auth_router)
# Các API quản trị: bắt buộc đăng nhập (header Authorization: Bearer <token>)
_auth_required = [Depends(get_current_user)]
app.include_router(violations_router, dependencies=_auth_required)
app.include_router(statistics_router, dependencies=_auth_required)
app.include_router(locations_router, dependencies=_auth_required)
# analyze: từng endpoint tự khai báo đăng nhập; /video/output được middleware protect_media kiểm tra (cookie)
app.include_router(analyze_router)
# Quản lý tài khoản: chỉ admin (kiểm tra trong từng endpoint)
app.include_router(users_router)
# Trạng thái hệ thống / thông báo: đăng nhập (khai báo trong router); số liệu trang chủ: công khai (chỉ số đếm)
app.include_router(system_router)
app.include_router(public_router)


if __name__ == "__main__":
    import uvicorn
    # Khởi chạy server trên cổng 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
