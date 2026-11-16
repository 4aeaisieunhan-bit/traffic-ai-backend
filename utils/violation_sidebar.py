# utils/violation_sidebar.py
# Sidebar giám sát giao thông thời gian thực — phong cách Traffic AI Command Center
import cv2
import numpy as np
from models.ocr import read_plate_details, recognize_plate_text
from models.license_plate import model as lp_yolo_model
from models.vehicle import model as vehicle_yolo_model
from models.yolo_types import INFER_KW, as_results

# ─────────────────────────── MÀU SẮC THEME ───────────────────────────
# Bảng màu: Nền tối + viền neon
C_BG_DARK    = (18, 16, 24)       # Nền chính
C_BG_MID     = (28, 24, 36)       # Nền card
C_BG_HEADER  = (22, 18, 32)       # Nền header
C_ACCENT     = (0, 215, 255)      # Cyan neon (màu chủ đạo)
C_RED_NEON   = (0, 48, 220)       # Đỏ vi phạm
C_RED_GLOW   = (20, 20, 100)      # Đỏ nhạt glow
C_GREEN_NEON = (0, 230, 80)       # Xanh lá nhận diện
C_GREEN_GLOW = (10, 60, 10)       # Xanh lá nhạt
C_ORANGE     = (0, 140, 255)      # Cam cảnh báo
C_WHITE      = (230, 230, 240)    # Trắng văn bản
C_GRAY_L     = (160, 160, 180)    # Xám sáng (nhãn phụ)
C_GRAY_D     = (60, 55, 75)       # Xám tối (đường kẻ)
C_GOLD       = (40, 200, 255)     # Vàng neon
C_PURPLE     = (180, 60, 120)     # Tím viền
C_TEAL       = (180, 200, 10)     # Teal (ROI label)


def _draw_scan_line(canvas, x1, x2, y, color, alpha=0.4):
    """Vẽ đường quét neon bán trong suốt."""
    overlay = canvas.copy()
    cv2.line(overlay, (x1, y), (x2, y), color, 1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def _paste_fit(canvas, img, x, y, w, h, bg=(30, 28, 36)):
    """Dán ảnh vào ô (x, y, w, h) giữ nguyên tỷ lệ, phần thừa tô nền tối (không méo hình người)."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return
    s = min(w / float(iw), h / float(ih))
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized


def _draw_rounded_rect(img, pt1, pt2, color, thickness=1, r=6, fill=None):
    """Vẽ hình chữ nhật với góc bo tròn."""
    x1, y1 = pt1
    x2, y2 = pt2
    if fill is not None:
        # Lấp đầy nền
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), fill, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), fill, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, fill, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, fill, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, fill, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, fill, -1)
    if thickness > 0:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness)


def _hex_pulse_dots(canvas, cx, y, n=3, r=3, color=C_ACCENT, frame_idx=0):
    """Vẽ dãy chấm nhấp nháy (loading indicator)."""
    gap = 10
    total_w = (n - 1) * gap
    sx = cx - total_w // 2
    for i in range(n):
        phase = (frame_idx // 8 + i) % n
        alpha = 1.0 if phase == 0 else 0.3
        c = tuple(int(ch * alpha) for ch in color)
        cv2.circle(canvas, (sx + i * gap, y), r, c, -1)


class PlateVoteTracker:
    """
    Theo dõi và biểu quyết biển số xe qua nhiều frame theo thuật toán từ nd_bien.py:
    - Lưu trữ lịch sử đọc OCR của từng xe (votes).
    - Biểu quyết chọn chuỗi ký tự xuất hiện nhiều nhất và có độ tin cậy cao nhất.
    - Duy trì ảnh crop biển số đẹp nhất (best_crop) có diện tích lớn và rõ nét nhất.
    """

    def __init__(self):
        self.tracks = {}

    def add_reading(self, track_id, cls_name, crop, conf, text, frame_idx=0):
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "votes": {},
                "best_crop": None,
                "best_conf": 0.0,
                "cls_name": cls_name,
                "last_seen": frame_idx,
            }
        t = self.tracks[track_id]
        t["cls_name"] = cls_name
        t["last_seen"] = frame_idx

        clean_t = text.strip().upper() if text else ""
        if len(clean_t.replace("-", "")) >= 2:
            t["votes"].setdefault(clean_t, []).append(conf)

        if crop is not None and crop.size > 0:
            area = crop.shape[0] * crop.shape[1]
            prev_area = t["best_crop"].shape[0] * t["best_crop"].shape[1] if t["best_crop"] is not None else 0
            if area > prev_area * 1.05 or (area >= prev_area * 0.85 and conf > t["best_conf"]):
                t["best_conf"] = conf
                t["best_crop"] = crop.copy()

    def get_best(self, track_id):
        if track_id not in self.tracks:
            return None, None
        t = self.tracks[track_id]
        votes = t["votes"]
        if not votes:
            return None, t["best_crop"]

        candidates = [k for k in votes.keys() if len(k.replace("-", "")) >= 4]
        if not candidates:
            candidates = [k for k in votes.keys() if len(k.replace("-", "")) >= 3]
        if not candidates:
            candidates = list(votes.keys())

        best_text = max(candidates, key=lambda k: sum(votes[k]) * (len(k.replace("-", "")) ** 0.5))
        return best_text, t["best_crop"]


class ViolationSidebar:
    """
    Thanh giám sát Traffic AI Command Center:
    - Header: logo + chỉ số thống kê realtime
    - Card khu vực trên: xe vi phạm (đỏ neon) + xe đang nhận diện (xanh)
    - Log panel dưới: danh sách đầy đủ biển số vi phạm
    - Footer: timestamp + chế độ hệ thống
    """

    def __init__(self, sidebar_width=416, height=720, fps=30.0, max_cards=4):
        self.sidebar_width = int(sidebar_width)
        self.height = int(height)
        self.fps = max(float(fps), 1.0)
        self.max_cards = int(max_cards)

        self.vote_tracker = PlateVoteTracker()
        self.vehicle_plates = {}
        self.live_recognized_plates = []
        self.violations_history = []
        self.recorded_violation_ids = set()   # {(track_id, kind)}

    # ─────────────────────── CẬP NHẬT BIỂN SỐ ───────────────────────

    def update_plates(self, vehicle_boxes, plate_boxes, frame, frame_idx=0, orig_frame=None):
        """
        Nhận diện biển số 2 lượt kết hợp phóng to và biểu quyết theo nd_bien.py.
        Hỗ trợ trích xuất crop trực tiếp từ orig_frame (4K) để đạt độ nét tối đa.
        """
        if vehicle_boxes is None or len(vehicle_boxes) == 0:
            return

        h, w = frame.shape[:2]
        h_orig, w_orig = (orig_frame.shape[:2]) if orig_frame is not None else (h, w)
        scale_x = w_orig / float(w)
        scale_y = h_orig / float(h)

        for v_box in vehicle_boxes:
            if v_box.id is None:
                continue
            track_id = int(v_box.id[0])
            vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0].tolist())
            car_w = vx2 - vx1
            car_h = vy2 - vy1

            cls_name = "xe"
            if v_box.cls is not None:
                cls_id = int(v_box.cls[0])
                cls_name = vehicle_yolo_model.names.get(cls_id, "xe")

            found_crop = None
            found_conf = 0.0

            # --- LƯỢT 1: Tìm biển số từ plate_boxes toàn frame ---
            if plate_boxes is not None and len(plate_boxes) > 0:
                for p_box in plate_boxes:
                    px1, py1, px2, py2 = map(int, p_box.xyxy[0].tolist())
                    p_conf = float(p_box.conf[0])
                    p_cx = (px1 + px2) // 2
                    p_cy = (py1 + py2) // 2

                    if (vx1 - 15) <= p_cx <= (vx2 + 15) and (vy1 - 15) <= p_cy <= (vy2 + 15):
                        pad = 4
                        if orig_frame is not None:
                            cpy1 = max(0, int((py1 - pad) * scale_y))
                            cpy2 = min(h_orig, int((py2 + pad) * scale_y))
                            cpx1 = max(0, int((px1 - pad) * scale_x))
                            cpx2 = min(w_orig, int((px2 + pad) * scale_x))
                            if cpy2 > cpy1 and cpx2 > cpx1:
                                found_crop = orig_frame[cpy1:cpy2, cpx1:cpx2].copy()
                                found_conf = p_conf
                                break
                        else:
                            cpy1 = max(0, py1 - pad)
                            cpy2 = min(h, py2 + pad)
                            cpx1 = max(0, px1 - pad)
                            cpx2 = min(w, px2 + pad)
                            if cpy2 > cpy1 and cpx2 > cpx1:
                                found_crop = frame[cpy1:cpy2, cpx1:cpx2].copy()
                                found_conf = p_conf
                                break

            # --- LƯỢT 2: Quét biển số trên vùng crop xe ---
            if orig_frame is not None:
                sub_y0 = max(0, int((vy1 + 0.30 * car_h) * scale_y))
                sub_y1 = min(h_orig, int(vy2 * scale_y))
                pad_x = int(car_w * 0.08 * scale_x)
                sub_x0 = max(0, int(vx1 * scale_x) - pad_x)
                sub_x1 = min(w_orig, int(vx2 * scale_x) + pad_x)
                sub_car = orig_frame[sub_y0:sub_y1, sub_x0:sub_x1]

                if sub_car.size > 0:
                    try:
                        lp_res = as_results(lp_yolo_model.predict(sub_car, conf=0.14, verbose=False, **INFER_KW))
                        lp_boxes = lp_res[0].boxes if lp_res else None
                        if lp_boxes is not None and len(lp_boxes) > 0:
                            best_b = lp_boxes[int(lp_boxes.conf.argmax())]
                            bx1, by1, bx2, by2 = map(int, best_b.xyxy[0].tolist())
                            bx1 = max(0, bx1 - 4)
                            by1 = max(0, by1 - 4)
                            bx2 = min(sub_car.shape[1], bx2 + 4)
                            by2 = min(sub_car.shape[0], by2 + 4)
                            crop_sub = sub_car[by1:by2, bx1:bx2].copy()
                            conf_sub = float(best_b.conf[0])

                            if found_crop is None or conf_sub > found_conf:
                                found_crop = crop_sub
                                found_conf = conf_sub
                    except Exception:
                        pass
            else:
                sub_y0 = max(0, int(vy1 + 0.30 * car_h))
                sub_y1 = min(h, vy2)
                pad_x = int(car_w * 0.08)
                sub_x0 = max(0, vx1 - pad_x)
                sub_x1 = min(w, vx2 + pad_x)
                sub_car = frame[sub_y0:sub_y1, sub_x0:sub_x1]

                if sub_car.size > 0:
                    try:
                        upscale = 2.5
                        sub_big = cv2.resize(sub_car, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)
                        lp_res = as_results(lp_yolo_model.predict(sub_big, conf=0.14, verbose=False, **INFER_KW))
                        lp_boxes = lp_res[0].boxes if lp_res else None
                        if lp_boxes is not None and len(lp_boxes) > 0:
                            best_b = lp_boxes[int(lp_boxes.conf.argmax())]
                            bx1, by1, bx2, by2 = map(int, (best_b.xyxy[0] / upscale).tolist())
                            bx1 = max(0, bx1 - 2)
                            by1 = max(0, by1 - 2)
                            bx2 = min(sub_car.shape[1], bx2 + 2)
                            by2 = min(sub_car.shape[0], by2 + 2)
                            crop_sub = sub_car[by1:by2, bx1:bx2].copy()
                            conf_sub = float(best_b.conf[0])

                            if found_crop is None or conf_sub > found_conf:
                                found_crop = crop_sub
                                found_conf = conf_sub
                    except Exception:
                        pass

            if found_crop is not None and found_crop.size > 0:
                plate_text, ocr_conf, _ = read_plate_details(found_crop, conf_thresh=0.10)
                if plate_text:
                    self.vote_tracker.add_reading(
                        track_id=track_id,
                        cls_name=cls_name,
                        crop=found_crop,
                        conf=max(found_conf, ocr_conf),
                        text=plate_text,
                        frame_idx=frame_idx,
                    )

                best_text, best_crop = self.vote_tracker.get_best(track_id)
                if best_text and best_crop is not None:
                    p_area = best_crop.shape[0] * best_crop.shape[1]
                    self.vehicle_plates[track_id] = (
                        best_crop,
                        found_conf,
                        (vx1, vy1, vx2, vy2),
                        p_area,
                        best_text,
                    )
                    self._update_live_recognized(track_id, cls_name, best_crop, best_text, frame_idx)

    def _update_live_recognized(self, track_id, cls_name, crop, text, frame_idx):
        existing = next((item for item in self.live_recognized_plates if item["track_id"] == track_id), None)
        if existing:
            existing["plate_crop"] = crop
            existing["plate_text"] = text
            existing["frame_idx"] = frame_idx
            self.live_recognized_plates.remove(existing)
            self.live_recognized_plates.insert(0, existing)
        else:
            self.live_recognized_plates.insert(0, {
                "track_id": track_id,
                "cls_name": cls_name,
                "plate_crop": crop,
                "plate_text": text,
                "frame_idx": frame_idx,
            })
        if len(self.live_recognized_plates) > 20:
            self.live_recognized_plates.pop()

    def check_violations(self, violation_tracker, frame, frame_idx, orig_frame=None):
        """Ghi các xe vượt đèn đỏ mà thuật toán quỹ đạo vừa kết luận."""
        for track_id, v_info in violation_tracker.vehicles.items():
            if v_info.get("status") == "VIOLATION":
                self.record_violation(
                    track_id, v_info.get("class_name", "xe"), v_info.get("reason", "Vuot den do"),
                    v_info.get("bbox"), frame, frame_idx, orig_frame=orig_frame, kind="RED_LIGHT",
                )

    def record_violation(self, track_id, cls_name, reason, v_bbox, frame, frame_idx, orig_frame=None, kind="RED_LIGHT",
                         subject_crop=None):
        """
        Ghi 1 vi phạm (lấy biển số tốt nhất đã bình chọn, nếu chưa có thì dò lại trên xe) vào sidebar.
        Chống trùng theo (track_id, kind): một xe có thể vừa vượt đèn đỏ vừa không đội mũ bảo hiểm.
        subject_crop: ảnh cận cảnh người vi phạm (lỗi mũ) -> hiện trên thẻ sidebar + lưu làm bằng chứng.
        Trả về True nếu là vi phạm mới.
        """
        key = (track_id, kind)
        if key in self.recorded_violation_ids:
            return False
        self.recorded_violation_ids.add(key)

        h, w = frame.shape[:2]
        h_orig, w_orig = (orig_frame.shape[:2]) if orig_frame is not None else (h, w)
        scale_x = w_orig / float(w)
        scale_y = h_orig / float(h)

        plate_crop = None
        plate_text = ""

        best_text, best_crop = self.vote_tracker.get_best(track_id)
        if best_text and best_crop is not None:
            plate_text = best_text
            plate_crop = best_crop
        elif track_id in self.vehicle_plates:
            plate_crop = self.vehicle_plates[track_id][0]
            plate_text = self.vehicle_plates[track_id][4]

        if plate_crop is None and v_bbox is not None:
            bx1, by1, bx2, by2 = v_bbox
            if orig_frame is not None:
                sub_y1 = max(0, int((by1 + 0.35 * (by2 - by1)) * scale_y))
                sub_y2 = min(h_orig, int(by2 * scale_y))
                sub_x1 = max(0, int(bx1 * scale_x))
                sub_x2 = min(w_orig, int(bx2 * scale_x))
                source_img = orig_frame
            else:
                sub_y1 = max(0, int(by1 + 0.35 * (by2 - by1)))
                sub_y2 = min(h, by2)
                sub_x1 = max(0, bx1)
                sub_x2 = min(w, bx2)
                source_img = frame

            if sub_y2 > sub_y1 and sub_x2 > sub_x1:
                sub_car = source_img[sub_y1:sub_y2, sub_x1:sub_x2]
                try:
                    lp_res = as_results(lp_yolo_model.predict(sub_car, conf=0.10, verbose=False, **INFER_KW))
                    lp_boxes = lp_res[0].boxes if lp_res else None
                    if lp_boxes is not None and len(lp_boxes) > 0:
                        best_b = lp_boxes[int(lp_boxes.conf.argmax())]
                        pbx1, pby1, pbx2, pby2 = map(int, best_b.xyxy[0].tolist())
                        pbx1 = max(0, pbx1 - 4)
                        pby1 = max(0, pby1 - 4)
                        pbx2 = min(sub_car.shape[1], pbx2 + 4)
                        pby2 = min(sub_car.shape[0], pby2 + 4)
                        plate_crop = sub_car[pby1:pby2, pbx1:pbx2].copy()
                except Exception:
                    pass
                if plate_crop is None:
                    plate_crop = sub_car.copy()

            if plate_crop is not None and plate_crop.size > 0:
                ptxt, _, _ = read_plate_details(plate_crop, conf_thresh=0.10)
                plate_text = ptxt or ""

        if not plate_text or len(plate_text.strip().replace("-", "")) < 2:
            clean_text = f"ID-{track_id:03d}"
        else:
            clean_text = plate_text.strip().upper()

        time_sec = frame_idx / self.fps
        minutes = int(time_sec // 60)
        seconds = time_sec % 60
        time_str = f"{minutes:02d}:{seconds:04.1f}s"

        self.violations_history.insert(0, {
            "track_id": track_id,
            "cls_name": cls_name,
            "plate_crop": plate_crop,
            "plate_text": clean_text,
            "time_str": time_str,
            "frame_idx": frame_idx,
            "reason": reason,
            "is_violation": True,
            "kind": kind,
            "highlight_bbox": v_bbox,
            "subject_crop": subject_crop,
        })
        return True

    # ─────────────────────────── RENDER ───────────────────────────────

    def render(self, frame, frame_idx):
        h, w = frame.shape[:2]
        sb_w = self.sidebar_width
        total_w = w + sb_w

        canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame

        sx = w          # X bắt đầu sidebar
        ex = total_w    # X kết thúc sidebar

        # ═══════════════ NỀN SIDEBAR ═══════════════
        canvas[:, sx:ex] = C_BG_DARK

        # Lưới nền dạng grid nhỏ (texture công nghệ)
        for gy in range(0, h, 20):
            cv2.line(canvas, (sx, gy), (ex, gy), (28, 24, 38), 1)
        for gx in range(sx, ex, 20):
            cv2.line(canvas, (gx, 0), (gx, h), (28, 24, 38), 1)

        # ═══════════════ ĐƯỜNG PHÂN CÁCH SIDEBAR ═══════════════
        # Viền neon 3 lớp (glow effect)
        cv2.line(canvas, (sx, 0), (sx, h), (45, 40, 60), 4)
        cv2.line(canvas, (sx + 1, 0), (sx + 1, h), C_ACCENT, 1)

        num_violations = len(self.violations_history)
        num_scanned = len(self.vehicle_plates)

        # ═══════════════ HEADER ═══════════════
        self._render_header(canvas, sx, ex, num_violations, num_scanned, frame_idx)

        # ═══════════════ CARDS + LOG PANEL (adaptive layout) ═══════════════
        cards_bottom_y = self._render_cards(canvas, sx, ex, num_violations, h, frame_idx)

        # ═══════════════ LOG PANEL KHU VỰC DƯỚI ═══════════════
        self._render_log_panel(canvas, sx, ex, num_violations, cards_bottom_y, h, frame_idx)

        # ═══════════════ FOOTER ═══════════════
        self._render_footer(canvas, sx, ex, h, frame_idx)

        return canvas

    # ─────────────────── HEADER ───────────────────

    def _render_header(self, canvas, sx, ex, num_vio, num_scan, frame_idx):
        hdr_h = 68
        # Nền header gradient giả
        cv2.rectangle(canvas, (sx, 0), (ex, hdr_h), C_BG_HEADER, -1)
        # Viền dưới header 2 màu
        cv2.line(canvas, (sx, hdr_h - 2), (ex, hdr_h - 2), C_GRAY_D, 1)
        cv2.line(canvas, (sx, hdr_h - 1), (ex, hdr_h - 1), C_ACCENT, 1)

        # Icon radar/vòng tròn bên trái
        ic_cx, ic_cy = sx + 24, 34
        cv2.circle(canvas, (ic_cx, ic_cy), 14, (30, 25, 40), -1)
        cv2.circle(canvas, (ic_cx, ic_cy), 14, C_ACCENT, 1)
        cv2.circle(canvas, (ic_cx, ic_cy), 9,  C_ACCENT, 1)
        cv2.circle(canvas, (ic_cx, ic_cy), 4,  C_ACCENT, -1)
        # Đường kẻ chữ thập (radar crosshair)
        cv2.line(canvas, (ic_cx - 14, ic_cy), (ic_cx + 14, ic_cy), C_ACCENT, 1)
        cv2.line(canvas, (ic_cx, ic_cy - 14), (ic_cx, ic_cy + 14), C_ACCENT, 1)

        # Tiêu đề hệ thống
        cv2.putText(canvas, "TRAFFIC AI MONITOR",
                    (sx + 46, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "HE THONG GIAM SAT GIAO THONG 4.0",
                    (sx + 46, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_GRAY_L, 1, cv2.LINE_AA)

        # Thanh thống kê nhanh — 2 chỉ số nằm trên 1 dòng
        # --- Vi phạm ---
        vio_x = sx + 10
        cv2.rectangle(canvas, (vio_x, 46), (vio_x + 95, 62), C_RED_GLOW, -1)
        cv2.rectangle(canvas, (vio_x, 46), (vio_x + 95, 62), C_RED_NEON, 1)
        cv2.putText(canvas, f"VI PHAM: {num_vio}",
                    (vio_x + 6, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 120, 255), 1, cv2.LINE_AA)
        # --- Đã quét ---
        sc_x = vio_x + 103
        cv2.rectangle(canvas, (sc_x, 46), (sc_x + 100, 62), C_GREEN_GLOW, -1)
        cv2.rectangle(canvas, (sc_x, 46), (sc_x + 100, 62), C_GREEN_NEON, 1)
        cv2.putText(canvas, f"DA QUET: {num_scan}",
                    (sc_x + 6, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_GREEN_NEON, 1, cv2.LINE_AA)
        # --- LIVE badge ---
        live_x = sc_x + 108
        pulse = frame_idx % 24 < 12
        badge_col = (0, 0, 200) if pulse else (20, 20, 80)
        cv2.rectangle(canvas, (live_x, 46), (live_x + 55, 62), badge_col, -1)
        cv2.rectangle(canvas, (live_x, 46), (live_x + 55, 62), (0, 60, 255), 1)
        cv2.putText(canvas, "● LIVE",
                    (live_x + 6, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 100, 255), 1, cv2.LINE_AA)

    # ─────────────────── CARDS ───────────────────

    def _render_cards(self, canvas, sx, ex, num_violations, h, frame_idx):
        """
        Vẽ khu vực cards với layout thích ứng (adaptive):
        - 0 vi phạm: 1 card standby
        - 1–2 vi phạm: 2 card lớn (card_h=130) + log panel chiếm phần còn lại
        - 3–5 vi phạm: 3 card nhỏ hơn (card_h=100)
        - 6+ vi phạm: chỉ hiện 2 card highlight + toàn bộ chuyển xuống log panel

        Trả về y cuối cùng của khu vực cards.
        """
        card_y_start = 76
        margin = 10
        footer_h = 30
        log_min_h = 180   # chiều cao tối thiểu cần giữ cho log panel

        # Xác định số cards và chiều cao card dựa trên số vi phạm
        # Luôn dành đủ chỗ cho log panel
        available_h = h - card_y_start - log_min_h - footer_h

        if num_violations == 0:
            # Chưa có vi phạm: 1 card standby nhỏ
            card_slots = 1
            card_h = min(130, available_h - 6)
        elif num_violations <= 2:
            # Ít vi phạm: 2 card, hiển thị ảnh đầy đủ
            card_slots = min(2, self.max_cards)
            card_h = min(130, (available_h - 6) // 2)
        elif num_violations <= 5:
            # Trung bình: 3 card, thu nhỏ chiều cao để fit
            card_slots = min(3, self.max_cards)
            card_h = min(110, (available_h - 12) // 3)
        else:
            # Nhiều vi phạm (6+): chỉ hiện 2 card nổi bật mới nhất,
            # phần còn lại dành hết cho log panel (hiển thị tất cả)
            card_slots = 2
            card_h = min(120, (available_h - 6) // 2)

        card_h = max(card_h, 90)   # tối thiểu 90px để hiển thị được ảnh
        card_gap = 6

        # Lập danh sách thẻ — vi phạm ưu tiên, còn slot thì thêm live plates
        cards = []
        for item in self.violations_history[:card_slots]:
            c = item.copy(); c["is_violation"] = True; cards.append(c)

        violation_tids = {v["track_id"] for v in self.violations_history}
        for item in self.live_recognized_plates:
            if len(cards) >= card_slots:
                break
            if item["track_id"] not in violation_tids:
                c = item.copy(); c["is_violation"] = False; cards.append(c)

        if not cards:
            self._render_standby(canvas, sx, ex, card_y_start, card_h, frame_idx)
            return card_y_start + card_h + card_gap

        curr_y = card_y_start
        for item in cards[:card_slots]:
            self._render_one_card(canvas, sx, ex, item, curr_y, card_h, margin, frame_idx)
            curr_y += card_h + card_gap

        # Nếu số vi phạm vượt quá card_slots → hiển thị badge "+N xe vi phạm nữa"
        hidden = num_violations - card_slots
        if hidden > 0:
            bx1 = sx + margin
            bx2 = ex - margin
            info_y1 = curr_y
            info_y2 = curr_y + 22
            cv2.rectangle(canvas, (bx1, info_y1), (bx2, info_y2), (28, 14, 10), -1)
            cv2.rectangle(canvas, (bx1, info_y1), (bx2, info_y2), C_RED_NEON, 1)
            cv2.putText(canvas,
                        f"+ {hidden} xe vi pham khac  v  Xem trong Nhat ky bên duoi",
                        (bx1 + 8, info_y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 255), 1, cv2.LINE_AA)
            curr_y = info_y2 + 4

        return curr_y

    def _render_standby(self, canvas, sx, ex, y_start, card_h, frame_idx):
        """Màn hình chờ khi chưa có xe nào nhận diện."""
        bx1, bx2 = sx + 10, ex - 10
        by1, by2 = y_start, y_start + card_h
        _draw_rounded_rect(canvas, (bx1, by1), (bx2, by2), C_GRAY_D, r=6, fill=C_BG_MID)

        # Icon camera với vòng quét
        ic_cx = sx + self.sidebar_width // 2
        ic_cy = by1 + min(52, card_h // 2 - 10)
        cv2.circle(canvas, (ic_cx, ic_cy), 22, C_GRAY_D, 1)
        cv2.rectangle(canvas, (ic_cx - 12, ic_cy - 8), (ic_cx + 12, ic_cy + 8), C_GRAY_D, 1)
        cv2.circle(canvas, (ic_cx, ic_cy), 5, C_GRAY_D, 1)

        txt_y = ic_cy + 40
        if txt_y < by2 - 10:
            cv2.putText(canvas, "DANG QUAN SAT...",
                        (bx1 + 70, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY_L, 1, cv2.LINE_AA)
        if txt_y + 23 < by2 - 10:
            cv2.putText(canvas, "Chua phat hien xe trong vung ROI",
                        (bx1 + 30, txt_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_GRAY_D, 1, cv2.LINE_AA)
        if txt_y + 43 < by2 - 2:
            _hex_pulse_dots(canvas, ic_cx, txt_y + 43, frame_idx=frame_idx)

    def _render_one_card(self, canvas, sx, ex, item, cy1, card_h, margin, frame_idx):
        """Vẽ 1 card xe — vi phạm hoặc nhận diện."""
        is_vio = item.get("is_violation", False)
        cx1 = sx + margin
        cx2 = ex - margin
        cy2 = cy1 + card_h

        # ── Nền card ──
        if is_vio:
            fill_col = (22, 10, 10)
            border_col = C_RED_NEON
            strip_col = C_RED_NEON
            glow_col = (40, 10, 80)
        else:
            fill_col = (10, 20, 18)
            border_col = C_GREEN_NEON
            strip_col = C_GREEN_NEON
            glow_col = (10, 40, 20)

        _draw_rounded_rect(canvas, (cx1, cy1), (cx2, cy2), border_col, r=5, fill=fill_col)

        # Vạch neon bên trái
        cv2.rectangle(canvas, (cx1, cy1 + 5), (cx1 + 4, cy2 - 5), strip_col, -1)

        # ── Badge góc trên phải ──
        badge_text = "!! VI PHAM !!" if is_vio else "DA NHAN DIEN"
        badge_col = C_RED_NEON if is_vio else (0, 140, 50)
        badge_text_col = (255, 255, 255)
        bw = 108 if is_vio else 100
        bx1_b = cx2 - bw - 4
        bx2_b = cx2 - 4
        # Nhấp nháy badge vi phạm
        if is_vio:
            pulse = frame_idx % 20 < 10
            badge_col = C_RED_NEON if pulse else (0, 20, 100)
        _draw_rounded_rect(canvas, (bx1_b, cy1 + 5), (bx2_b, cy1 + 22), badge_text_col, r=3, fill=badge_col)
        cv2.putText(canvas, badge_text,
                    (bx1_b + 6, cy1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, badge_text_col, 1, cv2.LINE_AA)

        # ── (Lỗi mũ) Ô DỌC TRÁI CÙNG: ảnh cận cảnh người vi phạm ──
        subject = item.get("subject_crop")
        has_subject = is_vio and subject is not None and subject.size > 0
        img_x1 = cx1 + 10
        if has_subject:
            sub_w, sub_h = 62, max(40, card_h - 50)
            sub_y1 = cy1 + 14
            _paste_fit(canvas, subject, img_x1, sub_y1, sub_w, sub_h)
            _draw_rounded_rect(canvas, (img_x1 - 1, sub_y1 - 1), (img_x1 + sub_w + 1, sub_y1 + sub_h + 1),
                               C_RED_NEON, r=3, thickness=1)
            img_x1 += sub_w + 8

        # ── PHẦN TRÁI: ảnh crop biển số ──
        img_y1 = cy1 + 14
        img_w, img_h = (96, 44) if has_subject else (118, 54)
        img_x2 = img_x1 + img_w
        img_y2 = img_y1 + img_h

        plate_crop = item.get("plate_crop")
        if plate_crop is not None and plate_crop.size > 0:
            try:
                # Nếu ảnh crop lớn hơn khung hiển thị (từ 4K), dùng INTER_AREA để ảnh thu nhỏ siêu nét
                # Nếu ảnh nhỏ hơn, dùng INTER_LANCZOS4 để phóng to mượt mà
                if plate_crop.shape[1] >= img_w and plate_crop.shape[0] >= img_h:
                    interp = cv2.INTER_AREA
                else:
                    interp = cv2.INTER_LANCZOS4
                resized = cv2.resize(plate_crop, (img_w, img_h), interpolation=interp)
                canvas[img_y1:img_y2, img_x1:img_x2] = resized
            except Exception:
                cv2.rectangle(canvas, (img_x1, img_y1), (img_x2, img_y2), C_GRAY_D, -1)
        else:
            cv2.rectangle(canvas, (img_x1, img_y1), (img_x2, img_y2), (30, 28, 36), -1)
            cv2.putText(canvas, "NO IMG",
                        (img_x1 + 25, img_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY_D, 1)

        # Viền neon quanh ảnh biển
        _draw_rounded_rect(canvas, (img_x1 - 1, img_y1 - 1), (img_x2 + 1, img_y2 + 1),
                           strip_col, r=3, thickness=1)
        cv2.putText(canvas, "BIEN SO",
                    (img_x1 + (img_w - 44) // 2, img_y2 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_GRAY_L, 1, cv2.LINE_AA)

        # ── PHẦN PHẢI: thông tin văn bản ──
        info_x = img_x2 + 12
        # Nhãn nhỏ "OCR"
        cv2.putText(canvas, "OCR BIEN SO:",
                    (info_x, cy1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_GRAY_L, 1, cv2.LINE_AA)
        # Chữ biển số to + màu nổi bật
        plate_txt = item.get("plate_text", "---")
        txt_col = (0, 60, 255) if is_vio else C_ACCENT
        cv2.putText(canvas, plate_txt,
                    (info_x, cy1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, txt_col, 2, cv2.LINE_AA)

        # Loại xe + ID
        cls_name = item.get("cls_name", "xe").upper()
        tid = item.get("track_id", 0)
        cv2.putText(canvas, f"{cls_name}  ID#{tid:03d}",
                    (info_x, cy1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_GRAY_L, 1, cv2.LINE_AA)

        # ── DÒNG THÔNG TIN DƯỚI CARD ──
        bottom_y = cy1 + card_h - 32

        # Đường kẻ ngang phân cách
        cv2.line(canvas, (cx1 + 8, bottom_y - 2), (cx2 - 8, bottom_y - 2), C_GRAY_D, 1)

        if is_vio:
            reason = item.get("reason", "Vuot den do")
            cv2.putText(canvas, f"LOI: {reason[:30]}",
                        (cx1 + 10, bottom_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, item.get("time_str", ""),
                        (cx2 - 90, bottom_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_GOLD, 1, cv2.LINE_AA)
            # Góc dưới phải: tag
            cv2.putText(canvas, "[CAM]",
                        (cx1 + 10, cy2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_RED_NEON, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Luu thong hop le — ROI Bottom",
                        (cx1 + 10, bottom_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_GREEN_NEON, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"F{item.get('frame_idx', 0):05d}",
                        (cx2 - 70, bottom_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_TEAL, 1, cv2.LINE_AA)
            cv2.putText(canvas, "[SCAN]",
                        (cx1 + 10, cy2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_GREEN_NEON, 1, cv2.LINE_AA)

    # ─────────────────── LOG PANEL ───────────────────

    def _render_log_panel(self, canvas, sx, ex, num_vio, top_y, h, frame_idx):
        """
        Bảng danh sách tất cả biển số vi phạm dạng bảng kỹ thuật số.
        - Số hàng tự động tính theo không gian còn lại (adaptive).
        - Hiện tất cả vi phạm, không giới hạn cứng.
        - Nếu danh sách dài hơn số hàng hiển thị: cuộn tự động theo thời gian (auto-scroll).
        """
        footer_h = 28
        log_x1 = sx + 8
        log_x2 = ex - 8
        log_y1 = top_y + 6
        log_y2 = h - footer_h

        # Đảm bảo log panel luôn có tối thiểu 60px
        if log_y2 - log_y1 < 60:
            return

        # Nền panel
        _draw_rounded_rect(canvas, (log_x1, log_y1), (log_x2, log_y2),
                           C_GRAY_D, r=5, fill=(20, 18, 28))

        # ── Header log panel ──
        hdr_h = 26
        hdr_y2 = log_y1 + hdr_h
        cv2.rectangle(canvas, (log_x1, log_y1), (log_x2, hdr_y2), (28, 22, 40), -1)
        cv2.line(canvas, (log_x1, hdr_y2), (log_x2, hdr_y2), C_ORANGE, 1)

        cv2.circle(canvas, (log_x1 + 12, log_y1 + 13), 4, C_ORANGE, -1)
        cv2.putText(canvas, "NHAT KY VI PHAM",
                    (log_x1 + 24, log_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ORANGE, 1, cv2.LINE_AA)

        # Badge tổng số — đổi màu khi nhiều vi phạm
        badge_col = (0, 60, 200) if num_vio >= 5 else C_GREEN_NEON
        badge_txt = f"TONG: {num_vio}"
        cv2.putText(canvas, badge_txt,
                    (log_x2 - 76, log_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, badge_col, 1, cv2.LINE_AA)

        # ── Tiêu đề cột ──
        col_y = hdr_y2 + 18
        if col_y >= log_y2 - 10:
            return
        cv2.putText(canvas, "STT",      (log_x1 + 6,   col_y), cv2.FONT_HERSHEY_SIMPLEX, 0.29, C_GRAY_L, 1, cv2.LINE_AA)
        cv2.putText(canvas, "BIEN SO",  (log_x1 + 36,  col_y), cv2.FONT_HERSHEY_SIMPLEX, 0.29, C_GRAY_L, 1, cv2.LINE_AA)
        cv2.putText(canvas, "LOAI XE",  (log_x1 + 155, col_y), cv2.FONT_HERSHEY_SIMPLEX, 0.29, C_GRAY_L, 1, cv2.LINE_AA)
        cv2.putText(canvas, "GIO",      (log_x1 + 258, col_y), cv2.FONT_HERSHEY_SIMPLEX, 0.29, C_GRAY_L, 1, cv2.LINE_AA)
        cv2.line(canvas, (log_x1 + 4, col_y + 5), (log_x2 - 4, col_y + 5), C_GRAY_D, 1)

        if not self.violations_history:
            mid_y = (log_y1 + log_y2) // 2 + 15
            cv2.putText(canvas, "[ CHUA CO VI PHAM DUOC GHI NHAN ]",
                        (log_x1 + 14, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_GRAY_D, 1, cv2.LINE_AA)
            return

        # ── Tính số hàng tối đa vừa với không gian ──
        row_y_start = col_y + 18
        row_h = 20
        max_rows = max(1, (log_y2 - row_y_start - 6) // row_h)

        total_vio = len(self.violations_history)

        # Auto-scroll: nếu số vi phạm vượt quá max_rows, cuộn chậm
        # Cứ 90 frame (3 giây) dịch chuyển 1 hàng, sau đó quay lại đầu
        if total_vio > max_rows:
            scroll_period = max_rows * 90        # thời gian để cuộn hết 1 vòng
            scroll_offset = (frame_idx // 90) % total_vio
            # Giữ scroll_offset trong giới hạn sao cho vẫn hiển thị đủ max_rows
            scroll_offset = min(scroll_offset, total_vio - max_rows)
        else:
            scroll_offset = 0

        # Hiển thị indicator cuộn nếu danh sách bị cắt
        if total_vio > max_rows:
            scroll_txt = f"^ Cuon  {scroll_offset + 1}–{min(scroll_offset + max_rows, total_vio)}/{total_vio}"
            cv2.putText(canvas, scroll_txt,
                        (log_x2 - 130, col_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, C_TEAL, 1, cv2.LINE_AA)

        # In các hàng
        row_y = row_y_start
        display_items = self.violations_history[scroll_offset: scroll_offset + max_rows]

        for i, item in enumerate(display_items):
            # Số thứ tự thực tế (mới nhất = 1)
            real_idx = scroll_offset + i
            stt = total_vio - real_idx

            # Màu nền hàng xen kẽ; hàng mới nhất (stt == total_vio) sáng hơn
            if stt == total_vio:
                row_bg = (38, 20, 20)   # Đỏ nhạt: vi phạm mới nhất
            else:
                row_bg = (26, 22, 36) if i % 2 == 0 else (20, 18, 28)

            cv2.rectangle(canvas,
                          (log_x1 + 2, row_y - row_h + 4),
                          (log_x2 - 2, row_y + 4),
                          row_bg, -1)

            # Chấm màu đầu hàng
            dot_col = C_RED_NEON if stt == total_vio else C_GRAY_D
            cv2.circle(canvas, (log_x1 + 4, row_y - 6), 3, dot_col, -1)

            cv2.putText(canvas, f"#{stt:02d}",
                        (log_x1 + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.31, C_ORANGE, 1, cv2.LINE_AA)
            cv2.putText(canvas, item["plate_text"][:14],
                        (log_x1 + 36, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_ACCENT, 1, cv2.LINE_AA)
            cv2.putText(canvas, item["cls_name"][:8].upper(),
                        (log_x1 + 155, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, item["time_str"],
                        (log_x1 + 258, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_GREEN_NEON, 1, cv2.LINE_AA)

            row_y += row_h

        # Thanh cuộn dọc (scrollbar) bên phải nếu cần
        if total_vio > max_rows:
            sb_x = log_x2 - 5
            sb_y1 = row_y_start
            sb_y2 = log_y2 - 4
            sb_h = sb_y2 - sb_y1
            # Vị trí thanh trượt
            thumb_ratio = max_rows / total_vio
            thumb_h = max(10, int(sb_h * thumb_ratio))
            thumb_top = sb_y1 + int(sb_h * scroll_offset / total_vio)
            cv2.rectangle(canvas, (sb_x, sb_y1), (sb_x + 3, sb_y2), C_GRAY_D, -1)
            cv2.rectangle(canvas, (sb_x, thumb_top), (sb_x + 3, min(thumb_top + thumb_h, sb_y2)), C_ORANGE, -1)

    # ─────────────────── FOOTER ───────────────────

    def _render_footer(self, canvas, sx, ex, h, frame_idx):
        """Footer: nhãn hệ thống + thanh tiến trình."""
        fy = h - 26
        cv2.line(canvas, (sx, fy - 2), (ex, fy - 2), C_GRAY_D, 1)
        cv2.rectangle(canvas, (sx, fy - 2), (ex, h), (16, 14, 22), -1)

        # Chữ footer
        cv2.putText(canvas, "TRAFFIC AI v2.0  |  YOLO + ELECTRA OCR",
                    (sx + 10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_GRAY_D, 1, cv2.LINE_AA)

        # Thanh nhỏ bên phải — nhấp nháy RECORD
        pulse = frame_idx % 30 < 15
        rc = (0, 0, 200) if pulse else (30, 20, 60)
        cv2.circle(canvas, (ex - 16, h - 13), 4, rc, -1)
        cv2.putText(canvas, "REC",
                    (ex - 36, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, rc, 1, cv2.LINE_AA)
