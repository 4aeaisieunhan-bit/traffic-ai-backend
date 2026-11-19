# utils/text_render.py
# Vẽ chữ tiếng Việt có dấu lên khung hình OpenCV bằng Pillow + font IBM Plex (assets/fonts).
# cv2.putText chỉ vẽ được ASCII -> dùng module này cho thanh bên và bảng thông tin trên video kết quả.
# Thiếu Pillow / font: tự quay về cv2.putText (bỏ dấu), không lỗi.
import os
import unicodedata

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:   # thiếu Pillow: vẫn chạy, chữ vẽ bằng OpenCV (không dấu)
    Image = ImageDraw = ImageFont = None

# ─────────────────────────── FONT ───────────────────────────
_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
_FONT_CACHE = {}


def _font(kind, size):
    """kind: 'sans' | 'sans-medium' | 'mono' | 'mono-medium'. None nếu không nạp được (dùng OpenCV)."""
    key = (kind, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    f = None
    if ImageFont is not None:
        try:
            if kind.startswith("sans"):
                f = ImageFont.truetype(os.path.join(_FONT_DIR, "IBMPlexSans-Variable.ttf"), size)
                f.set_variation_by_axes([500 if kind.endswith("medium") else 400, 100])   # [độ đậm, độ rộng]
            else:
                name = "IBMPlexMono-Medium.ttf" if kind.endswith("medium") else "IBMPlexMono-Regular.ttf"
                f = ImageFont.truetype(os.path.join(_FONT_DIR, name), size)
        except Exception:
            f = None
    _FONT_CACHE[key] = f
    return f


def _ascii(text):
    """Bỏ dấu tiếng Việt cho trường hợp phải vẽ bằng OpenCV."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


class TextLayer:
    """
    Gom các dòng chữ của 1 frame rồi vẽ một lượt bằng Pillow (có dấu tiếng Việt).
    Toạ độ (x, y) là đường chân chữ (baseline) bên trái, giống cv2.putText.
    """

    def __init__(self):
        self.items = []

    def add(self, x, y, text, color, size=13, kind="sans", align="left"):
        if text:
            self.items.append((x, y, str(text), color, size, kind, align))

    @staticmethod
    def width(text, size=13, kind="sans"):
        f = _font(kind, size)
        if f is not None:
            return int(round(f.getlength(str(text))))
        (w, _), _ = cv2.getTextSize(_ascii(str(text)), cv2.FONT_HERSHEY_SIMPLEX, size / 34.0, 1)
        return w

    @classmethod
    def fit(cls, text, max_w, size=13, kind="sans"):
        """Cắt bớt kèm dấu … nếu chuỗi dài hơn max_w pixel."""
        text = str(text)
        if cls.width(text, size, kind) <= max_w:
            return text
        while text and cls.width(text + "…", size, kind) > max_w:
            text = text[:-1]
        return text.rstrip() + "…"

    def flush(self, canvas, x0, x1):
        """Vẽ toàn bộ chữ lên vùng canvas[:, x0:x1] (canvas BGR; màu truyền vào cũng là BGR)."""
        if not self.items:
            return
        if Image is None or _font("sans", 13) is None:
            for x, y, text, color, size, kind, align in self.items:
                w = self.width(text, size, kind)
                tx = x - w if align == "right" else x
                cv2.putText(canvas, _ascii(text), (tx, y), cv2.FONT_HERSHEY_SIMPLEX, size / 34.0, color, 1, cv2.LINE_AA)
            self.items.clear()
            return
        region = np.ascontiguousarray(canvas[:, x0:x1])
        img = Image.fromarray(region)          # kênh BGR được giữ nguyên, màu truyền vào cũng theo BGR
        draw = ImageDraw.Draw(img)
        for x, y, text, color, size, kind, align in self.items:
            draw.text((x - x0, y), text, font=_font(kind, size), fill=tuple(int(c) for c in color),
                      anchor="rs" if align == "right" else "ls")
        canvas[:, x0:x1] = np.asarray(img)
        self.items.clear()




_TextLayer = TextLayer   # tên cũ
