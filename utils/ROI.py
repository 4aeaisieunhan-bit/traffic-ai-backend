# Module xử lý vạch dừng (stopline): ước lượng, ổn định và xác định phía của xe so với vạch.
#
# Vạch dừng được biểu diễn là một ĐOẠN THẲNG (A, B) thay vì bounding box thẳng trục:
#   - Đúng cả khi camera đặt chéo và vạch dừng nằm xiên trên ảnh.
#   - ROI TOP / ROI BOTTOM = hai phía của đường thẳng đi qua A, B (kéo dài hết khung hình).
#   - Xe ở phía nào được xác định bằng khoảng cách có dấu (tích có hướng) -> O(1), không cần polygon test.
#
# Dùng ở: pipeline.py (StoplineStabilizer), traffic_violation.py (signed_distance, horizontal_segment_from_bbox).
from collections import deque
import math

import cv2
import numpy as np


# ───────────────────────────── Hình học vạch dừng ─────────────────────────────

def horizontal_segment_from_bbox(bbox):
    """Đoạn thẳng nằm ngang qua tâm bbox — dùng khi không ước lượng được hướng vạch."""
    x1, y1, x2, y2 = map(int, bbox)
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_mid = (y1 + y2) / 2.0
    return (float(x_min), y_mid), (float(x_max), y_mid)


def estimate_stopline_segment(frame, bbox, min_contrast=30):
    """
    Ước lượng đoạn thẳng vạch dừng bên trong bbox bằng cách fit đường thẳng qua các pixel sơn trắng.

    Vạch dừng sáng hơn mặt đường -> ngưỡng Otsu trong vùng bbox -> cv2.fitLine (Huber, chống nhiễu).
    Nếu vùng quá nhỏ, độ tương phản thấp hoặc đường fit gần thẳng đứng -> trả về đoạn nằm ngang qua tâm bbox.

    Returns:
        ((xa, ya), (xb, yb)) với xa < xb, tọa độ trong khung hình.
    """
    fallback = horizontal_segment_from_bbox(bbox)
    if frame is None:
        return fallback

    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = max(0, min(x1, x2)), min(W - 1, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(H - 1, max(y1, y2))
    cw, ch = x2 - x1 + 1, y2 - y1 + 1
    if cw < 20 or ch < 2:
        return fallback

    crop = frame[y1:y2 + 1, x1:x2 + 1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bright = gray[mask > 0]
    dark = gray[mask == 0]
    if bright.size < max(20, 0.03 * cw * ch) or dark.size == 0:
        return fallback
    if float(bright.mean()) - float(dark.mean()) < min_contrast:
        return fallback

    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if abs(vx) < 0.2:  # dốc hơn ~78° -> không phải vạch dừng ngang/xiên, bỏ
        return fallback

    slope = vy / vx
    yl = float(np.clip(y0 + slope * (0 - x0), 0, ch - 1))
    yr = float(np.clip(y0 + slope * ((cw - 1) - x0), 0, ch - 1))
    return (float(x1), y1 + yl), (float(x2), y1 + yr)


def signed_distance(point, segment):
    """
    Khoảng cách có dấu (pixel) từ điểm tới đường thẳng đi qua đoạn vạch dừng.
      < 0 : phía trên vạch (hướng vào giao lộ — ROI TOP)
      > 0 : phía dưới vạch (làn tiếp cận — ROI BOTTOM)
    """
    (ax, ay), (bx, by) = segment
    if bx < ax:
        ax, ay, bx, by = bx, by, ax, ay
    dx, dy = bx - ax, by - ay
    n = math.hypot(dx, dy) or 1.0
    return (dx * (point[1] - ay) - dy * (point[0] - ax)) / n


# ───────────────────────────── Ổn định vạch dừng theo thời gian ─────────────────────────────

class StoplineStabilizer:
    """
    Làm mượt và khóa vạch dừng cho camera cố định.

    - Giai đoạn hiệu chỉnh: lấy MEDIAN của `window` lần phát hiện gần nhất (bbox + 2 đầu đoạn thẳng)
      -> vạch không còn rung vài pixel giữa các frame.
    - Sau `lock_after` lần phát hiện và độ dao động nhỏ (`lock_max_std` px) -> KHÓA vạch, bỏ qua nhiễu.
    - Khi đã khóa, nếu phát hiện lệch > `relock_px` liên tục `relock_frames` lần (camera bị xoay / đổi cảnh)
      -> mở khóa và hiệu chỉnh lại.
    - Frame không phát hiện được vạch: giữ nguyên vạch hiện tại.
    """

    def __init__(self, window=30, lock_after=45, lock_max_std=6.0, relock_px=40, relock_frames=30):
        self.window = int(window)
        self.lock_after = int(lock_after)
        self.lock_max_std = float(lock_max_std)
        self.relock_px = float(relock_px)
        self.relock_frames = int(relock_frames)
        self.reset()

    def reset(self):
        self._samples = deque(maxlen=self.window)   # (x1, y1, x2, y2, y_left, y_right)
        self._n_seen = 0
        self._deviate_count = 0
        self.locked = False
        self.bbox = None
        self.segment = None

    def _set_from(self, arr):
        x1, y1, x2, y2, yl, yr = (float(v) for v in arr)
        self.bbox = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        self.segment = ((x1, yl), (x2, yr))

    def update(self, frame, bbox):
        """
        Args:
            frame: khung hình (chưa vẽ gì) để ước lượng hướng vạch.
            bbox: (x1, y1, x2, y2) vạch dừng phát hiện được ở frame này, hoặc None.

        Returns:
            (bbox, segment) đã ổn định, hoặc (None, None) nếu chưa từng thấy vạch.
        """
        if bbox is None:
            return self.bbox, self.segment

        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if self.locked and self.bbox is not None:
            lx1, ly1, lx2, ly2 = self.bbox
            shift = math.hypot((x1 + x2 - lx1 - lx2) / 2.0, (y1 + y2 - ly1 - ly2) / 2.0)
            if shift <= self.relock_px:
                self._deviate_count = 0
                return self.bbox, self.segment
            self._deviate_count += 1
            if self._deviate_count < self.relock_frames:
                return self.bbox, self.segment
            # Camera đã đổi góc -> hiệu chỉnh lại từ đầu
            self.reset()

        (_, yl), (_, yr) = estimate_stopline_segment(frame, (x1, y1, x2, y2))
        self._samples.append((x1, y1, x2, y2, yl, yr))
        self._n_seen += 1

        arr = np.asarray(self._samples, dtype=np.float32)
        self._set_from(np.median(arr, axis=0))

        if self._n_seen >= self.lock_after and len(self._samples) >= self.window:
            if float(arr.std(axis=0).max()) <= self.lock_max_std:
                self.locked = True
                self._deviate_count = 0

        return self.bbox, self.segment

