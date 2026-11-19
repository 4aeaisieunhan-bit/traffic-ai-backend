# models/weights.py
# Nơi khai báo DUY NHẤT đường dẫn file trọng số của các mô hình YOLO.
#   - Đường dẫn được đổi thành tuyệt đối theo thư mục backend -> chạy server từ thư mục nào cũng được.
#   - Đổi mô hình: đặt biến môi trường tương ứng trong .env (đường dẫn tuyệt đối hoặc tương đối với thư mục backend).
#   - Module này không nạp PyTorch / Ultralytics nên đọc được cả khi chỉ cần thông tin (VD: /system/status).
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(env_name: str, default: str) -> str:
    path = os.getenv(env_name, "").strip() or default
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(BASE_DIR, path))


VEHICLE_WEIGHTS = _resolve("VEHICLE_WEIGHTS", "weights_demo/phuongtienmoi.pt")
TRAFFIC_LIGHT_WEIGHTS = _resolve("TRAFFIC_LIGHT_WEIGHTS", "weights_demo/dengiaothong.pt")
HELMET_WEIGHTS = _resolve("HELMET_WEIGHTS", "weights_demo/mubaohiem.pt")
LICENSE_PLATE_WEIGHTS = _resolve("LICENSE_PLATE_WEIGHTS", "weights_demo/biensoxe.pt")
STOP_LINE_WEIGHTS = _resolve("STOP_LINE_WEIGHTS", "weights_demo/vachdung.pt")
OCR_WEIGHTS = _resolve("OCR_WEIGHTS", "weights_demo/kytu.pt")


def weight_label(path: str) -> str:
    """Tên hiển thị / lưu DB của file trọng số: 'weights_demo/phuongtienmoi.pt' (tương đối với backend),
    hoặc chỉ tên file nếu nằm ngoài thư mục backend."""
    try:
        rel = os.path.relpath(path, BASE_DIR)
    except ValueError:          # Windows: khác ổ đĩa
        rel = os.path.basename(path)
    if rel.startswith(".."):
        rel = os.path.basename(path)
    return rel.replace(os.sep, "/")
