# utils/violation_sidebar.py
# Thanh bên của video kết quả: hồ sơ vi phạm mới nhất + nhật ký biển số.
# Thiết kế đồng bộ với giao diện web (theo skill unslop của Damir Sokolovsky, @heydamir, CC BY 4.0):
# nền tối trung tính, 3 cấp chữ, 1 màu đỏ tín hiệu cho vi phạm, chữ tiếng Việt có dấu (IBM Plex, vẽ bằng Pillow).
import cv2
import numpy as np
from models.ocr import read_plate_details
from models.license_plate import model as lp_yolo_model
from models.vehicle import model as vehicle_yolo_model
from models.yolo_types import INFER_KW, as_results
from utils.labels import vehicle_label
from utils.text_render import TextLayer

# Xe có mép dưới bbox cách đáy khung hình <= số pixel này coi như đang bị cắt (biển số chưa vào khung)
BOTTOM_EDGE_PX = 3

# ─────────────────────────── BẢNG MÀU (BGR) ───────────────────────────
# Cùng token với web: --bg-deep, --card, --raised, --line, --t1/--t2/--t3, --signal, --positive
C_BG     = (13, 9, 8)          # #08090d nền thanh bên
C_CARD   = (25, 20, 18)        # #121419 thẻ
C_RAISED = (36, 29, 26)        # #1a1d24 ô ảnh trống
C_LINE   = (34, 31, 29)        # đường kẻ mảnh
C_T1     = (243, 239, 237)     # #edeff3 chữ chính
C_T2     = (179, 168, 164)     # #a4a8b3 chữ phụ
C_T3     = (143, 130, 125)     # #7d828f chữ mờ, meta
C_SIGNAL = (77, 72, 229)       # #e5484d vi phạm / vạch dừng
C_OK     = (138, 195, 76)      # #4cc38a nhận diện hợp lệ

# ─────────────────────────── NỘI DUNG HIỂN THỊ ───────────────────────────
_REASON_VI = [
    ("Khong doi mu bao hiem", "Không đội mũ bảo hiểm"),
    ("Di thang / re trai khi den do", "Đi thẳng hoặc rẽ trái khi đèn đỏ"),
    ("Di thang khi den do", "Đi thẳng khi đèn đỏ"),
    ("Re trai khi den do", "Rẽ trái khi đèn đỏ"),
    ("Re phai khi den do", "Rẽ phải khi đèn đỏ"),
    ("Qua stopline va dung", "Qua vạch dừng khi đèn đỏ"),
    ("Vuot den do", "Vượt đèn đỏ"),
]


def _reason_vi(reason):
    """Lý do vi phạm (chuỗi không dấu từ thuật toán) -> câu tiếng Việt có dấu để hiển thị."""
    r = (reason or "").strip()
    for key, vi in _REASON_VI:
        if key.lower() in r.lower():
            return vi
    return r.replace("Vi pham: ", "") or "Vi phạm"


def _vehicle_vi(cls_name):
    """Tên loại xe trên thanh bên — cùng bảng nhãn với database / báo cáo (utils/labels.py)."""
    return vehicle_label(cls_name) or "Xe"


# ─────────────────────────── HÀM VẼ ẢNH / KHUNG ───────────────────────────

def _paste_fit(canvas, img, x, y, w, h, bg=C_RAISED):
    """Dán ảnh vào ô (x, y, w, h) giữ nguyên tỷ lệ, phần thừa tô màu nền (không méo hình)."""
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


def _paste_cover(canvas, img, x, y, w, h, anchor_y=0.3):
    """
    Dán ảnh LẤP ĐẦY ô (x, y, w, h), giữ tỷ lệ: phóng cho phủ kín rồi cắt phần thừa.
    anchor_y: vị trí giữ lại theo chiều dọc khi phải cắt (0 = giữ mép trên, 0.5 = giữa) —
    ảnh cận cảnh người vi phạm có phần đầu ở phía trên nên giữ lệch lên trên.
    """
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0 or w <= 0 or h <= 0:
        return
    s = max(w / float(iw), h / float(ih))
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    ox = (nw - w) // 2
    oy = int((nh - h) * anchor_y)
    canvas[y:y + h, x:x + w] = resized[oy:oy + h, ox:ox + w]


def _draw_rounded_rect(img, pt1, pt2, color, thickness=1, r=6, fill=None):
    """Vẽ hình chữ nhật với góc bo tròn."""
    x1, y1 = pt1
    x2, y2 = pt2
    if fill is not None:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), fill, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), fill, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, fill, -1, cv2.LINE_AA)
    if thickness > 0:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def _dot(canvas, x, y, color, r=3):
    cv2.circle(canvas, (x, y), r, color, -1, cv2.LINE_AA)


class PlateVoteTracker:
    """
    Theo dõi và biểu quyết biển số xe qua nhiều frame theo thuật toán từ nd_bien.py:
    - Lưu trữ lịch sử đọc OCR của từng xe (votes).
    - Biểu quyết chọn chuỗi ký tự xuất hiện nhiều nhất và có độ tin cậy cao nhất.
    - Lưu ảnh crop đẹp nhất (lớn, tin cậy cao) THEO TỪNG CHUỖI đọc được -> ảnh trả về luôn là ảnh
      của đúng chuỗi thắng phiếu (không ghép chữ của lần đọc này với ảnh của lần đọc khác).
    """

    def __init__(self):
        self.tracks = {}

    def add_reading(self, track_id, cls_name, crop, conf, text, frame_idx=0):
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "votes": {},
                "crops": {},          # {chuỗi: (crop, conf)} ảnh đẹp nhất của từng chuỗi
                "best_crop": None,    # ảnh đẹp nhất bất kể chuỗi (chỉ dùng khi chưa có chuỗi hợp lệ)
                "best_conf": 0.0,
                "cls_name": cls_name,
                "last_seen": frame_idx,
            }
        t = self.tracks[track_id]
        t["cls_name"] = cls_name
        t["last_seen"] = frame_idx

        clean_t = text.strip().upper() if text else ""
        valid = len(clean_t.replace("-", "")) >= 2
        if valid:
            t["votes"].setdefault(clean_t, []).append(conf)

        if crop is not None and crop.size > 0:
            if self._better(crop, conf, t["best_crop"], t["best_conf"]):
                t["best_conf"] = conf
                t["best_crop"] = crop.copy()
            if valid:
                prev_crop, prev_conf = t["crops"].get(clean_t, (None, 0.0))
                if self._better(crop, conf, prev_crop, prev_conf):
                    t["crops"][clean_t] = (crop.copy(), conf)

    @staticmethod
    def _better(crop, conf, prev_crop, prev_conf):
        """Ảnh mới tốt hơn nếu lớn hơn rõ rệt, hoặc gần bằng nhưng tin cậy cao hơn."""
        if prev_crop is None:
            return True
        area = crop.shape[0] * crop.shape[1]
        prev_area = prev_crop.shape[0] * prev_crop.shape[1]
        return area > prev_area * 1.05 or (area >= prev_area * 0.85 and conf > prev_conf)

    def get_best(self, track_id):
        if track_id not in self.tracks:
            return None, None
        t = self.tracks[track_id]
        votes = t["votes"]
        if not votes:
            return None, t["best_crop"]

        # Ưu tiên chuỗi đủ dài như biển số VN đầy đủ (>= 7 ký tự, VD 29D2-20743, 29A-12345): chuỗi cụt
        # (đọc được 1 phần, VD "66-905") không được thắng chỉ vì lặp lại vài lần khi đã có chuỗi đầy đủ
        candidates = []
        for min_len in (7, 4, 3):
            candidates = [k for k in votes.keys() if len(k.replace("-", "")) >= min_len]
            if candidates:
                break
        if not candidates:
            candidates = list(votes.keys())

        best_text = max(candidates, key=lambda k: sum(votes[k]) * (len(k.replace("-", "")) ** 0.5))
        crop = t["crops"].get(best_text, (None, 0.0))[0]
        return best_text, crop if crop is not None else t["best_crop"]


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
            if vy2 >= h - BOTTOM_EDGE_PX:
                # Xe chạm mép dưới khung hình: phần đuôi (biển số) đang bị cắt, thứ "giống biển số" còn
                # thấy được thường là gương / mũ / lưng người lái -> không đọc, chờ xe vào hẳn khung hình
                continue
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

    def refresh_violation_plates(self):
        """
        Vi phạm được ghi ngay khi kết luận — lúc đó xe có thể còn ở xa / chưa thấy biển số (hay gặp với lỗi
        không đội mũ: thấy đầu trước khi thấy biển). Mỗi frame, thay biển số của các vi phạm đã ghi bằng kết quả
        bình chọn mới nhất của xe đó (chữ + ảnh của đúng chuỗi thắng phiếu).
        Đánh dấu item["plate_updated"] = True để nơi lưu DB cập nhật lại.
        """
        for item in self.violations_history:
            best_text, best_crop = self.vote_tracker.get_best(item["track_id"])
            if not best_text or best_crop is None or len(best_text.replace("-", "")) < 2:
                continue
            if best_text != item["plate_text"] or best_crop is not item["plate_crop"]:
                item["plate_text"] = best_text
                item["plate_crop"] = best_crop
                item["plate_updated"] = True

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

        if plate_crop is None and v_bbox is not None and v_bbox[3] < h - BOTTOM_EDGE_PX:
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

        sx = w          # X bắt đầu thanh bên
        ex = total_w    # X kết thúc thanh bên

        # Nền phẳng + 1 đường kẻ mảnh ngăn cách với video
        canvas[:, sx:ex] = C_BG
        cv2.line(canvas, (sx, 0), (sx, h), C_LINE, 1)

        self._text = TextLayer()
        num_violations = len(self.violations_history)
        num_scanned = len(self.vehicle_plates)

        self._render_header(canvas, sx, ex, num_violations, num_scanned)
        cards_bottom_y = self._render_cards(canvas, sx, ex, num_violations, h, frame_idx)
        self._render_log_panel(canvas, sx, ex, num_violations, cards_bottom_y, h, frame_idx)

        self._text.flush(canvas, sx, ex)
        return canvas

    # ─────────────────── ĐẦU THANH BÊN ───────────────────

    def _render_header(self, canvas, sx, ex, num_vio, num_scan):
        T = self._text
        hdr_h = 56

        # Biểu tượng đèn tín hiệu phẳng (giống logo web) + tên sản phẩm
        mx, my = sx + 16, 19
        _draw_rounded_rect(canvas, (mx, my), (mx + 12, my + 18), C_T1, thickness=1, r=3)
        _dot(canvas, mx + 6, my + 5, C_T1, r=2)
        _dot(canvas, mx + 6, my + 9, C_T3, r=2)
        _dot(canvas, mx + 6, my + 13, C_T3, r=2)
        T.add(sx + 38, 33, "Traffic AI", C_T1, 15, "sans-medium")

        # Số liệu bên phải: con số (mono) + nhãn mờ
        x = ex - 16
        for num, label in ((num_scan, "xe đã quét"), (num_vio, "vi phạm")):
            T.add(x, 33, label, C_T3, 12, "sans", align="right")
            x -= T.width(label, 12) + 5
            T.add(x, 33, str(num), C_T1, 13, "mono-medium", align="right")
            x -= T.width(str(num), 13, "mono-medium") + 16

        cv2.line(canvas, (sx, hdr_h), (ex, hdr_h), C_LINE, 1)

    # ─────────────────── THẺ HỒ SƠ ───────────────────

    def _render_cards(self, canvas, sx, ex, num_violations, h, frame_idx):
        """
        Khu vực thẻ, bố cục thích ứng theo số vi phạm (luôn chừa chỗ cho nhật ký):
        - 0 vi phạm: 1 thẻ chờ
        - 1–2 vi phạm: 2 thẻ
        - 3–5 vi phạm: 3 thẻ thấp hơn
        - 6+ vi phạm: 2 thẻ mới nhất, phần còn lại xem trong nhật ký
        Trả về y cuối cùng của khu vực thẻ.
        """
        card_y_start = 68
        margin = 12
        card_gap = 8
        log_min_h = 200
        available_h = h - card_y_start - log_min_h - 12

        if num_violations == 0:
            card_slots, card_h = 1, min(108, available_h)
        elif num_violations <= 2:
            card_slots, card_h = min(2, self.max_cards), min(108, (available_h - card_gap) // 2)
        elif num_violations <= 5:
            card_slots, card_h = min(3, self.max_cards), min(96, (available_h - 2 * card_gap) // 3)
        else:
            card_slots, card_h = 2, min(104, (available_h - card_gap) // 2)
        card_h = max(card_h, 88)

        # Vi phạm ưu tiên, còn chỗ thì thêm xe vừa đọc được biển số
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
            self._render_standby(canvas, sx, ex, card_y_start, 88)
            return card_y_start + 88 + card_gap

        curr_y = card_y_start
        for item in cards[:card_slots]:
            self._render_one_card(canvas, sx, ex, item, curr_y, card_h, margin, frame_idx)
            curr_y += card_h + card_gap

        hidden = num_violations - card_slots
        if hidden > 0:
            self._text.add(sx + margin + 4, curr_y + 12, f"+{hidden} vi phạm khác trong nhật ký", C_T3, 12)
            curr_y += 20

        return curr_y

    def _render_standby(self, canvas, sx, ex, y_start, card_h):
        """Thẻ chờ khi chưa có xe nào được ghi nhận."""
        x1, x2 = sx + 12, ex - 12
        _draw_rounded_rect(canvas, (x1, y_start), (x2, y_start + card_h), C_LINE, r=10, fill=C_CARD)
        cx = (x1 + x2) // 2
        self._text.add(cx + self._text.width("Đang quan sát", 13, "sans-medium") // 2, y_start + 40,
                       "Đang quan sát", C_T2, 13, "sans-medium", align="right")
        self._text.add(cx + self._text.width("Chưa có xe vi phạm", 12) // 2, y_start + 60,
                       "Chưa có xe vi phạm", C_T3, 12, align="right")

    def _render_one_card(self, canvas, sx, ex, item, cy1, card_h, margin, frame_idx):
        """Một thẻ: ảnh người vi phạm (lỗi mũ) + ảnh biển số + lỗi, biển số, loại xe, thời điểm."""
        T = self._text
        is_vio = item.get("is_violation", False)
        cx1, cx2 = sx + margin, ex - margin
        cy2 = cy1 + card_h
        pad = 12
        top = cy1 + pad
        content_h = card_h - 2 * pad

        _draw_rounded_rect(canvas, (cx1, cy1), (cx2, cy2), C_LINE, r=10, fill=C_CARD)

        x = cx1 + pad

        # Ảnh cận cảnh người vi phạm (lỗi không đội mũ): lấp đầy khung, giữ phần đầu
        subject = item.get("subject_crop")
        has_subject = is_vio and subject is not None and subject.size > 0
        if has_subject:
            sub_w = 64
            _paste_cover(canvas, subject, x, top, sub_w, content_h, anchor_y=0.25)
            cv2.rectangle(canvas, (x, top), (x + sub_w, top + content_h), C_LINE, 1)
            x += sub_w + 8

        # Ảnh biển số: giữ đúng tỷ lệ, khung ôm sát ảnh
        area_w, area_h = (76 if has_subject else 88), min(52, content_h)
        plate_crop = item.get("plate_crop")
        if plate_crop is not None and plate_crop.size > 0:
            ph, pw = plate_crop.shape[:2]
            s = min(area_w / float(pw), area_h / float(ph))
            iw, ih = max(1, int(round(pw * s))), max(1, int(round(ph * s)))
            try:
                interp = cv2.INTER_AREA if iw < pw else cv2.INTER_LANCZOS4
                canvas[top:top + ih, x:x + iw] = cv2.resize(plate_crop, (iw, ih), interpolation=interp)
            except Exception:
                cv2.rectangle(canvas, (x, top), (x + iw, top + ih), C_RAISED, -1)
            cv2.rectangle(canvas, (x, top), (x + iw, top + ih), C_LINE, 1)
        else:
            cv2.rectangle(canvas, (x, top), (x + area_w, top + area_h), C_RAISED, -1)
            T.add(x + 10, top + area_h // 2 + 4, "Chưa có ảnh", C_T3, 11)

        # Cột chữ (vị trí cố định để các thẻ thẳng hàng)
        col_x = x + area_w + 16
        col_r = cx2 - pad
        col_w = col_r - col_x

        # Dòng 1: chấm trạng thái + lỗi vi phạm (cả chiều rộng cột)
        if is_vio:
            status_col, status_txt = C_SIGNAL, _reason_vi(item.get("reason"))
            time_txt = (item.get("time_str") or "").rstrip("s")
        else:
            status_col, status_txt = C_OK, "Đã nhận diện"
            t = item.get("frame_idx", 0) / self.fps
            time_txt = f"{int(t // 60):02d}:{t % 60:04.1f}"
        _dot(canvas, col_x + 3, top + 8, status_col)
        T.add(col_x + 12, top + 12, T.fit(status_txt, col_w - 12, 13, "sans-medium"),
              C_T1 if is_vio else C_T2, 13, "sans-medium")

        # Dòng 2: biển số
        plate_txt = item.get("plate_text") or "—"
        T.add(col_x, top + 38, T.fit(plate_txt, col_w, 20, "mono-medium"), C_T1, 20, "mono-medium")

        # Dòng 3: loại xe · mã theo dõi, thời điểm bên phải
        meta = f"{_vehicle_vi(item.get('cls_name'))} · ID {item.get('track_id', 0)}"
        time_w = T.width(time_txt, 12, "mono")
        T.add(col_r, top + 58, time_txt, C_T3, 12, "mono", align="right")
        T.add(col_x, top + 58, T.fit(meta, col_w - time_w - 12, 12), C_T3, 12)

    # ─────────────────── NHẬT KÝ VI PHẠM ───────────────────

    def _render_log_panel(self, canvas, sx, ex, num_vio, top_y, h, frame_idx):
        """
        Bảng tất cả biển số vi phạm, mới nhất ở trên.
        Số hàng tự tính theo chỗ còn lại; danh sách dài hơn thì tự cuộn chậm theo thời gian.
        """
        T = self._text
        x1, x2 = sx + 16, ex - 16
        y1 = top_y + 12
        y2 = h - 12
        if y2 - y1 < 72:
            return

        cv2.line(canvas, (sx, y1 - 4), (ex, y1 - 4), C_LINE, 1)

        # Tiêu đề
        T.add(x1, y1 + 18, "Nhật ký vi phạm", C_T1, 13, "sans-medium")

        # Cột
        col_y = y1 + 44
        cols = (x1, x1 + 30, x1 + 172)
        T.add(cols[0], col_y, "#", C_T3, 12)
        T.add(cols[1], col_y, "Biển số", C_T3, 12)
        T.add(cols[2], col_y, "Loại xe", C_T3, 12)
        T.add(x2, col_y, "Thời điểm", C_T3, 12, align="right")
        cv2.line(canvas, (x1, col_y + 8), (x2, col_y + 8), C_LINE, 1)

        total = len(self.violations_history)
        if not total:
            T.add(x1, y1 + 18, "", C_T3)
            T.add(x2, y1 + 18, "0 hồ sơ", C_T3, 12, align="right")
            T.add(x1, col_y + 34, "Chưa có vi phạm nào", C_T3, 12)
            return

        row_h = 24
        first_base = col_y + 8 + 17
        max_rows = max(1, (y2 - (col_y + 8)) // row_h)

        # Tự cuộn 1 hàng mỗi 90 khung hình khi danh sách dài hơn số hàng hiển thị
        if total > max_rows:
            offset = min((frame_idx // 90) % total, total - max_rows)
            T.add(x2, y1 + 18, f"{offset + 1}–{min(offset + max_rows, total)} / {total}", C_T3, 12, "mono", align="right")
        else:
            offset = 0
            T.add(x2, y1 + 18, f"{total} hồ sơ", C_T3, 12, align="right")

        for i, item in enumerate(self.violations_history[offset: offset + max_rows]):
            stt = total - (offset + i)          # mới nhất = số lớn nhất
            base = first_base + i * row_h
            newest = stt == total
            T.add(cols[0], base, f"{stt:02d}", C_T3, 12, "mono")
            if newest:
                _dot(canvas, cols[1] - 9, base - 4, C_SIGNAL)
            T.add(cols[1], base, T.fit(item.get("plate_text") or "—", cols[2] - cols[1] - 8, 13, "mono-medium"),
                  C_T1, 13, "mono-medium")
            T.add(cols[2], base, _vehicle_vi(item.get("cls_name")), C_T2, 12)
            T.add(x2, base, (item.get("time_str") or "").rstrip("s"), C_T2, 12, "mono", align="right")
            if i < max_rows - 1:
                cv2.line(canvas, (x1, base + 7), (x2, base + 7), C_LINE, 1)

        # Thanh cuộn mảnh khi danh sách bị cắt
        if total > max_rows:
            sb_x = ex - 6
            sb_y1, sb_y2 = col_y + 12, y2
            thumb_h = max(12, int((sb_y2 - sb_y1) * max_rows / total))
            thumb_top = sb_y1 + int((sb_y2 - sb_y1) * offset / total)
            cv2.line(canvas, (sb_x, thumb_top), (sb_x, min(thumb_top + thumb_h, sb_y2)), C_T3, 2)
