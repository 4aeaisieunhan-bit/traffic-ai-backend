# Traffic AI — Backend

API FastAPI và pipeline AI phát hiện vi phạm giao thông từ video camera ngã tư:
**vượt đèn đỏ** (theo quỹ đạo xe qua vạch dừng), **không dừng đúng vạch**, **không đội mũ bảo hiểm**,
kèm **đọc biển số**. Giao diện web ở repo `traffic-ai-frontend` (Angular).

## Cài đặt

Yêu cầu: Python 3.14, Microsoft SQL Server + *ODBC Driver 17/18 for SQL Server* (không kết nối được thì tự dùng SQLite).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# PyTorch cài riêng (chọn 1):
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130   # GPU NVIDIA
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu     # chỉ CPU
pip install -r requirements.txt
copy .env.example .env      # rồi sửa giá trị cho máy của bạn
```

Tạo CSDL: mở `database/traffic_ai_schema.sql` trong SSMS và chạy (F5). Server tự tạo bảng còn thiếu khi khởi động.

## Chạy

```powershell
python server.py                      # API tại http://localhost:8000, tài liệu API: /docs
python main.py "video.mp4" out.mp4    # phân tích 1 video từ dòng lệnh, không cần CSDL
```

**Lần khởi động đầu tiên** tạo tài khoản quản trị `ADMIN_EMAIL` (mặc định `admin@gmail.com`). Mật khẩu lấy từ
`ADMIN_PASSWORD`; không đặt thì mật khẩu ngẫu nhiên được **in một lần ra console**. Server cảnh báo nếu tài khoản
quản trị còn dùng mật khẩu `123456` của phiên bản cũ.

**Tài khoản tự đăng ký** phải được quản trị viên kích hoạt (trang *Quản lý tài khoản*, nút *Mở khóa*) mới đăng nhập
được. Khi demo trên máy cá nhân có thể đặt `AUTO_APPROVE_REGISTRATION=1`.

## Cấu hình (.env)

| Biến | Ý nghĩa |
|---|---|
| `DB_TYPE`, `MSSQL_*` | Kết nối SQL Server (xem `.env.example`) |
| `AI_DEVICE` | `auto` / `cpu` / `cuda` / `0` |
| `ADMIN_EMAIL`, `ADMIN_PASSWORD` | Tài khoản quản trị đầu tiên |
| `AUTO_APPROVE_REGISTRATION` | `1`: tài khoản đăng ký được kích hoạt ngay |
| `SAMPLE_VIDEO` | Video cho nút *Dùng video mẫu* (mặc định tìm `sample.mp4`, `21s_vidnew.mov` trong thư mục backend) |
| `VEHICLE_WEIGHTS`, `TRAFFIC_LIGHT_WEIGHTS`, `HELMET_WEIGHTS`, `LICENSE_PLATE_WEIGHTS`, `STOP_LINE_WEIGHTS`, `OCR_WEIGHTS` | Đổi file trọng số (mặc định `weights_demo/*.pt`) |
| `CORS_ORIGINS`, `CROSS_SITE_COOKIE` | Khi frontend ở tên miền khác (VD Vercel + Cloudflare Tunnel) |

## Cấu trúc

```
server.py            Khởi tạo CSDL, middleware bảo vệ /static, đăng ký router
main.py              Chạy pipeline trên 1 video từ dòng lệnh
api/                 Các endpoint: auth, users (admin), analyze, violations (+ xuất Excel/PDF), locations, statistics, system
database/            Model SQLAlchemy, schema T-SQL, ERD, migrate dữ liệu cũ, seed
models/              Bọc 6 mô hình YOLO; models/weights.py là nơi duy nhất khai báo file trọng số
utils/pipeline.py    Xử lý 1 khung hình (dùng chung cho main.py và API)
utils/traffic_violation.py   Quỹ đạo xe + đối chiếu đèn tại thời điểm qua vạch
utils/helmet_violation.py    Bỏ phiếu nhiều khung hình cho lỗi không đội mũ
utils/violation_sidebar.py   Bình chọn biển số + thanh bên trên video kết quả
utils/labels.py      Nhãn loại xe / loại vi phạm dùng chung
```

### Quy tắc vượt đèn đỏ

Chỉ xét xe đi từ phía làn chờ (ROI dưới) qua vạch dừng. Hướng đi (thẳng / rẽ trái / rẽ phải) được xác định từ quỹ
đạo trong tối đa 5 giây, rồi đối chiếu với **đèn tại lúc xe qua vạch**. Đèn mới chuyển đỏ sau khi xe đã vào giao lộ
không tạo vi phạm.

- Đi thẳng chỉ bị cấm khi **đèn tròn đỏ**.
- Rẽ trái / phải: được phép khi có mũi tên xanh của hướng đó, bị cấm khi có mũi tên đỏ của hướng đó, ngoài ra theo đèn tròn.
- Khung hình mô hình không thấy đèn nào thì giữ trạng thái đèn gần nhất tối đa 1 giây.

Lần phân tích bị huỷ, thất bại hoặc gián đoạn (tắt server) sẽ bị xoá kết quả dở dang; chỉ giữ bản ghi trong lịch sử.
