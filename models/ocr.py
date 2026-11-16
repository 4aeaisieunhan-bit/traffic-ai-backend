from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys

# Thêm thư mục gốc vào sys.path để import utils
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.frame_enhance import enhance_plate_crop
from models.yolo_types import INFER_KW, as_results, to_numpy

# Khởi tạo mô hình OCR nhận diện ký tự biển số xe (từ weights/OCR-26s-best.pt)
weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "OCR-26s-best.pt")
ocr_model = YOLO(weights_path)

# Các hằng số cấu hình thuật toán theo tài liệu nd_bien.py
OCR_MIN_CROP_WIDTH = 140      # Chiều rộng tối thiểu để phóng to ảnh biển số trước khi OCR
TWO_LINE_RATIO = 0.35         # Tỷ lệ chênh lệch Y / chiều cao ký tự để phân biệt 1 dòng hay 2 dòng
LINE_SEPARATOR = "-"          # Ký tự phân tách giữa dòng trên và dòng dưới


def chars_to_text(boxes, classes, names, two_line_ratio=TWO_LINE_RATIO):
    """
    Sắp xếp các bounding box ký tự theo tọa độ không gian thành chuỗi biển số chuẩn.
    Hỗ trợ phân tách biển số xe máy / ô tô 2 dòng và 1 dòng theo nd_bien.py.
    """
    if len(boxes) == 0:
        return ""

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    mean_h = float(np.mean(boxes[:, 3] - boxes[:, 1]))

    # Nếu độ lệch y giữa các ký tự vượt quá ngưỡng TWO_LINE_RATIO * mean_h -> Tách làm 2 dòng
    if len(boxes) >= 2 and (cy.max() - cy.min()) > two_line_ratio * mean_h:
        split = (cy.max() + cy.min()) / 2.0
        lines = [np.where(cy < split)[0], np.where(cy >= split)[0]]
    else:
        lines = [np.arange(len(boxes))]

    parts = []
    for idx in lines:
        idx = idx[np.argsort(cx[idx])]  # Sắp xếp từ trái qua phải
        line_str = "".join(str(names[int(classes[i])]) for i in idx if int(classes[i]) in names)
        if line_str:
            parts.append(line_str)

    return LINE_SEPARATOR.join(parts)


def read_plate_details(plate_img, conf_thresh=0.10, iou_thresh=0.45):
    """
    Nhận diện văn bản biển số với kỹ thuật phóng to ảnh (upscaling) và phân dòng thông minh theo nd_bien.py.

    Args:
        plate_img (np.ndarray): Ảnh crop biển số xe.
        conf_thresh (float): Ngưỡng tin cậy tối thiểu cho ký tự.
        iou_thresh (float): Ngưỡng IoU NMS.

    Returns:
        tuple: (plate_text: str, mean_conf: float, crop_in: np.ndarray) hoặc (None, 0.0, None)
    """
    if plate_img is None or plate_img.size == 0:
        return None, 0.0, None

    h, w = plate_img.shape[:2]
    if h < 5 or w < 5:
        return None, 0.0, None

    # Tăng cường chất lượng ảnh biển số: phóng to Lanczos-4, bù gamma, CLAHE kênh L và viền chữ sắc nét
    crop_in = enhance_plate_crop(plate_img, target_width=max(OCR_MIN_CROP_WIDTH, w))

    try:
        results = as_results(ocr_model.predict(
            source=crop_in,
            conf=conf_thresh,
            iou=iou_thresh,
            agnostic_nms=True,
            verbose=False,
            **INFER_KW,
        ))
    except Exception:
        return None, 0.0, None

    det = results[0].boxes if results else None
    if det is None or len(det) < 2:
        return None, 0.0, None

    boxes = to_numpy(det.xyxy)
    scores = to_numpy(det.conf)
    classes = to_numpy(det.cls)

    text = chars_to_text(boxes, classes, ocr_model.names)
    if not text:
        return None, 0.0, None

    mean_score = float(np.mean(scores))
    return text, mean_score, crop_in


def recognize_plate_text(plate_img, conf_thresh=0.10):
    """
    Hàm wrapper tương thích ngược: trả về chuỗi văn bản biển số nhận diện được.
    """
    text, _, _ = read_plate_details(plate_img, conf_thresh=conf_thresh)
    return text or ""
