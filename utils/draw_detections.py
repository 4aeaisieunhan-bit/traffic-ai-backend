# utils/draw_detections.py
# Vẽ bounding box các đối tượng phụ (đèn tín hiệu, mũ bảo hiểm, biển báo, biển số) và vạch dừng lên frame.
# Dùng chung cho main.py (chạy dòng lệnh) và api/analyze.py (phân tích từ giao diện web)
# để video đầu ra của hai luồng luôn giống nhau.
import cv2

# Màu khung mũ bảo hiểm (BGR) — tím / hồng: không trùng với đỏ (vi phạm, đèn đỏ, vạch dừng),
# xanh lá (xe hợp lệ, đèn xanh), vàng/cam (đang theo dõi, đèn vàng, biển số), xanh dương (biển báo, ROI)
HELMET_COLOR = (247, 85, 168)      # tím  #a855f7 — đội mũ
NO_HELMET_COLOR = (200, 0, 255)    # hồng #ff00c8 — không đội mũ / vi phạm không đội mũ


def _box_xyxy(box) -> tuple[int, int, int, int]:
    # Khai báo rõ 4 phần tử để Pylance hiểu khi bung *_box_xyxy(box) vào _draw_labeled
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return int(x1), int(y1), int(x2), int(y2)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _belongs_to_vehicle(cx, cy, vehicle_boxes, margin):
    for v_box in vehicle_boxes:
        vx1, vy1, vx2, vy2 = _box_xyxy(v_box)
        if vx1 - margin <= cx <= vx2 + margin and vy1 - margin <= cy <= vy2 + margin:
            return True
    return False


def _draw_labeled(frame, x1, y1, x2, y2, text, color, font_scale=0.3, pad=3, thickness=1, label_bg=False):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    org = (x1, max(y1 - pad, 10))
    if label_bg:
        # Nền tối sau chữ: nhãn đọc được cả trên nền trời sáng / cây cối
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (org[0] - 2, org[1] - th - 2), (org[0] + tw + 2, org[1] + base), (0, 0, 0), -1)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def draw_detections(
    frame,
    traffic_light_result,
    helmet_result,
    traffic_sign_result,
    license_plate_result,
    recognized_vehicle_boxes,
    active_stopline,
    helmet_violation_boxes=(),
):
    """Vẽ lên `frame` (tại chỗ) và trả về chính frame đó.
    helmet_violation_boxes: khung đầu (của chính mô hình helmet) thuộc người đã bị kết luận không đội mũ
    -> vẽ đậm + nhãn "VI PHAM: KHONG MU" (không vẽ thêm khung nào khác)."""
    height, width = frame.shape[:2]
    sl_y_top = min(active_stopline[1], active_stopline[3]) if active_stopline is not None else None

    # 1. Đèn giao thông: màu theo trạng thái (đỏ / xanh / vàng)
    res = traffic_light_result[0]
    if res.boxes is not None:
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < 0.25:
                continue
            label = res.names[int(box.cls[0])]
            color = (0, 0, 255) if "red" in label else (0, 255, 0) if "green" in label else (0, 215, 255)
            # Đèn ở xa rất nhỏ -> nới khung 3px, nhãn có nền để dễ thấy trong video đầu ra
            x1, y1, x2, y2 = _box_xyxy(box)
            _draw_labeled(frame, x1 - 3, y1 - 3, x2 + 3, y2 + 3, f"{label} {conf:.2f}", color,
                          font_scale=0.32, pad=4, thickness=1, label_bg=True)

    # 2. Mũ bảo hiểm: trong vùng phía trên vạch dừng chỉ vẽ nếu thuộc xe đang được theo dõi
    res = helmet_result[0]
    if res.boxes is not None:
        drawn = []   # mô hình đôi khi gắn CẢ helmet lẫn no_helmet cho cùng 1 đầu -> chỉ vẽ nhãn tin cậy hơn
        for box in sorted(res.boxes, key=lambda b: -float(b.conf[0])):
            conf = float(box.conf[0])
            if conf < 0.35:
                continue
            x1, y1, x2, y2 = _box_xyxy(box)
            if any(_iou((x1, y1, x2, y2), d) >= 0.5 for d in drawn):
                continue
            drawn.append((x1, y1, x2, y2))
            if sl_y_top is not None and y2 < sl_y_top and not _belongs_to_vehicle(
                (x1 + x2) // 2, (y1 + y2) // 2, recognized_vehicle_boxes, 15
            ):
                continue
            label = res.names[int(box.cls[0])]
            color = HELMET_COLOR if label == "helmet" else NO_HELMET_COLOR
            _draw_labeled(frame, x1, y1, x2, y2, f"{label} {conf:.2f}", color, font_scale=0.28, pad=3)
    # Người đã bị kết luận vi phạm: dùng lại đúng khung đầu của mô hình, vẽ đậm + nhãn (luôn hiện, mọi vùng)
    for hx1, hy1, hx2, hy2 in helmet_violation_boxes:
        _draw_labeled(frame, hx1 - 2, hy1 - 2, hx2 + 2, hy2 + 2, "VI PHAM: KHONG MU", NO_HELMET_COLOR,
                      font_scale=0.34, pad=4, thickness=2, label_bg=True)

    # 3. Biển báo giao thông
    res = traffic_sign_result[0]
    if res.boxes is not None:
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < 0.25:
                continue
            label = res.names[int(box.cls[0])]
            _draw_labeled(frame, *_box_xyxy(box), f"{label} {conf:.2f}", (255, 128, 0))

    # 4. Biển số: phía trên vạch dừng chỉ vẽ nếu thuộc xe đang được theo dõi
    res = license_plate_result[0]
    if res.boxes is not None:
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < 0.25:
                continue
            x1, y1, x2, y2 = _box_xyxy(box)
            if sl_y_top is not None and y2 < sl_y_top and not _belongs_to_vehicle(
                (x1 + x2) // 2, (y1 + y2) // 2, recognized_vehicle_boxes, 20
            ):
                continue
            label = res.names[int(box.cls[0])]
            _draw_labeled(frame, x1, y1, x2, y2, f"{label} {conf:.2f}", (0, 255, 255))

    # 5. Vạch dừng vẽ sau cùng để luôn nằm trên mọi lớp
    if active_stopline is not None:
        sx1, sy1, sx2, sy2 = active_stopline
        cv2.rectangle(frame, (0, sy1 - 1), (width - 1, sy2 + 1), (255, 255, 255), 1)
        cv2.rectangle(frame, (0, sy1), (width - 1, sy2), (0, 0, 230), 1)
        cv2.putText(frame, "STOP", (sx1, max(sy1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 80, 255), 1, cv2.LINE_AA)

    return frame
