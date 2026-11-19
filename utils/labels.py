# utils/labels.py
# Nhãn tiếng Việt dùng chung cho DB, API, báo cáo, main.py và thanh bên video kết quả.
# Không import database / torch -> main.py và utils/* dùng được mà không kết nối SQL Server.
import unicodedata
from typing import Optional

# Danh mục loại vi phạm: mã -> (tên, mô tả). database/lookups.py seed bảng violation_types từ đây.
VIOLATION_TYPES = {
    "RED_LIGHT": ("Vượt đèn đỏ", "Đi thẳng / rẽ trái / rẽ phải khi đèn đỏ"),
    "STOP_LINE": ("Không dừng đúng vạch khi đèn đỏ", "Vượt qua vạch dừng và dừng lại trong giao lộ khi đèn đỏ"),
    "NO_HELMET": ("Không đội mũ bảo hiểm", "Người điều khiển / ngồi sau xe máy không đội mũ bảo hiểm"),
}


def fold_text(text: Optional[str]) -> str:
    """Bỏ dấu tiếng Việt + chữ thường: 'Vượt đèn đỏ' -> 'vuot den do'."""
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


def classify_reason(reason: Optional[str]) -> str:
    """
    Mô tả vi phạm (chuỗi tự do từ thuật toán) -> mã loại vi phạm.
    VD: 'Vi pham: Di thang khi den do' -> RED_LIGHT
        'Vi pham: Qua stopline va dung yen qua 5.0s khi den do' -> STOP_LINE
    """
    t = fold_text(reason)
    if "mu bao hiem" in t or "helmet" in t:
        return "NO_HELMET"
    if "dung yen" in t or "stopline" in t or "vach" in t:
        return "STOP_LINE"
    return "RED_LIGHT"


# (từ khoá trong tên lớp YOLO / dữ liệu cũ đã bỏ dấu, nhãn hiển thị) — xét theo thứ tự.
# Lớp của mô hình phương tiện: bicycle, bus, car, motorcycle, truck, van.
_VEHICLE_KEYWORDS = [
    (("motor", "xe may"), "Xe máy"),
    (("bicycle", "xe dap"), "Xe đạp"),
    (("bus", "buyt", "khach", "16 cho"), "Xe buýt"),
    (("container",), "Xe container"),
    (("fire", "cuu hoa"), "Xe cứu hỏa"),
    (("truck", "xe tai"), "Xe tải"),
    (("car", "van", "sedan", "o to", "oto", "xe hoi"), "Ô tô"),
]


def vehicle_label(raw: Optional[str]) -> Optional[str]:
    """Tên lớp YOLO / dữ liệu cũ -> tên loại xe tiếng Việt: 'motorcycle' -> 'Xe máy'. None nếu không rõ."""
    t = fold_text(raw).replace("_", " ").strip()
    if not t or t in ("xe", "unknown"):
        return None
    for keywords, label in _VEHICLE_KEYWORDS:
        if any(k in t for k in keywords):
            return label
    return (raw or "").strip() or None
