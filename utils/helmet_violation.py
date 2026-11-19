# utils/helmet_violation.py
# Phát hiện vi phạm KHÔNG ĐỘI MŨ BẢO HIỂM trên TOÀN KHUNG HÌNH (mọi xe máy đang được tracking).
#
# Quy trình mỗi frame:
#   1. Lấy các xe máy có track_id.
#   2. Gắn mỗi khung đầu (helmet / no_helmet của mô hình helmet) vào ĐÚNG MỘT xe máy:
#      tâm khung đầu phải nằm trong "vùng đầu" ở phần trên khung xe (khung xe máy bao cả người lái),
#      nếu nhiều xe cùng chứa thì chọn xe có mép trên gần nhất.
#   3. Bỏ phiếu theo thời gian cho từng xe: chỉ kết luận vi phạm khi xe bị báo no_helmet ở
#      đủ nhiều frame (tuyệt đối) VÀ chiếm tỷ lệ cao (tương đối) trong số frame quan sát được đầu.
#      Mỗi phiếu được tính theo CHIỀU CAO khung đầu: frame xe ở gần (đầu to, rõ) đáng tin hơn frame
#      xe ở xa / đứng chờ đèn (đầu nhỏ, mô hình dễ nhầm mũ màu tối/xám thành đầu trần).
#      => một vài frame nhận nhầm (tóc đen, mũ lưỡi trai, bị che khuất) không tạo ra vi phạm.
#   4. Mỗi xe chỉ bị ghi tối đa 1 lần vi phạm này.
#
# Chống nhầm MŨ ĐEN / vật màu tối trên đầu thành "không đội mũ" (ở 10–30px mũ đen và tóc đen rất giống nhau):
#   a. Ca PHÂN VÂN: cùng 1 đầu mà mô hình gắn cả helmet (dù tin cậy thấp) lẫn no_helmet -> không bỏ phiếu.
#   b. Phiếu "không mũ" cần độ tin cậy cao (no_helmet_conf).
#   c. Kết luận vi phạm cần đủ số frame thấy đầu trần ĐỦ TO (min_decisive_px) — đầu nhỏ chỉ góp phiếu phụ.
import cv2

from utils.draw_detections import NO_HELMET_COLOR

REASON = "Vi pham: Khong doi mu bao hiem"   # ASCII: cv2.putText không vẽ được tiếng Việt có dấu

STATUS_CHECKING = "CHECKING"
STATUS_VIOLATION = "VIOLATION"


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _is_motorbike(cls_name):
    t = (cls_name or "").lower()
    return "xe may" in t or "motor" in t


class HelmetViolationDetector:
    def __init__(
        self,
        fps=30.0,
        min_observe_seconds=0.4,
        min_no_helmet_frames=8,
        no_helmet_ratio=0.6,
        no_helmet_conf=0.55,
        helmet_conf=0.35,
        min_head_px=7,
        ambiguous_helmet_conf=0.20,
        ambiguous_iou=0.35,
        min_decisive_px=14,
        min_decisive_frames=3,
    ):
        """
        Args:
            fps: tốc độ khung hình video.
            min_observe_seconds: thời gian tối thiểu xe phải được quan sát thấy đầu người.
            min_no_helmet_frames: số frame tối thiểu bị báo no_helmet.
            no_helmet_ratio: tỷ lệ tối thiểu của phiếu no_helmet (đã tính theo chiều cao đầu) trên tổng phiếu.
            no_helmet_conf / helmet_conf: ngưỡng tin cậy của từng lớp.
            min_head_px: chiều cao khung đầu tối thiểu (xe ở quá xa, đầu quá nhỏ => không đủ chi tiết, bỏ qua).
                         7px: vẫn xét được xe máy ở xa (phía trên vạch dừng, ~2/3 khung đầu ở đó cao < 10px);
                         đầu nhỏ hơn nữa thì mô hình gần như đoán ngẫu nhiên. Phiếu đã tính theo cỡ đầu nên
                         các frame đầu nhỏ ảnh hưởng ít tới kết luận.
            ambiguous_helmet_conf / ambiguous_iou: khung no_helmet bị một khung helmet (tin cậy >= ngưỡng này,
                         kể cả rất thấp) chồng lên (IoU >= ambiguous_iou) => mô hình đang phân vân (hay gặp với mũ đen)
                         => đầu đó không được bỏ phiếu.
            min_decisive_px / min_decisive_frames: phải có ít nhất min_decisive_frames frame thấy đầu trần cao
                         >= min_decisive_px mới được kết luận (đủ chi tiết để phân biệt mũ tối màu với tóc).
        """
        self.fps = max(float(fps), 1.0)
        self.min_observe_frames = max(int(self.fps * min_observe_seconds), 6)
        self.min_no_helmet_frames = int(min_no_helmet_frames)
        self.no_helmet_ratio = float(no_helmet_ratio)
        self.no_helmet_conf = float(no_helmet_conf)
        self.helmet_conf = float(helmet_conf)
        self.min_head_px = int(min_head_px)
        self.ambiguous_helmet_conf = float(ambiguous_helmet_conf)
        self.ambiguous_iou = float(ambiguous_iou)
        self.min_decisive_px = int(min_decisive_px)
        self.min_decisive_frames = int(min_decisive_frames)

        # {track_id: {status, cls_name, bbox, observed, no_helmet, last_frame, heads, evidence_heads}}
        self.tracks = {}
        self.violation_count = 0

    # ───────────────────────────── CẬP NHẬT ─────────────────────────────
    def update(self, frame_idx, vehicle_boxes, vehicle_names, helmet_boxes, helmet_names):
        """
        Trả về danh sách vi phạm MỚI phát hiện ở frame này:
        [{track_id, cls_name, bbox, head_boxes, frame_idx, reason, confidence}]
        """
        if vehicle_boxes is None or len(vehicle_boxes) == 0:
            return []

        # 1. Xe máy đang được tracking
        motos = []
        for box in vehicle_boxes:
            if box.id is None:
                continue
            cls_name = vehicle_names.get(int(box.cls[0]), "xe")
            if not _is_motorbike(cls_name):
                continue
            bbox = tuple(map(int, box.xyxy[0].tolist()))
            motos.append((int(box.id[0]), bbox, cls_name))
        if not motos:
            return []

        # 2. Khung đầu đủ tin cậy và đủ lớn
        raw = []
        if helmet_boxes is not None:
            for box in helmet_boxes:
                label = helmet_names.get(int(box.cls[0]), "")
                hx1, hy1, hx2, hy2 = map(int, box.xyxy[0].tolist())
                raw.append(((hx1, hy1, hx2, hy2), label, float(box.conf[0])))
        # Khung helmet kể cả tin cậy thấp: dùng để phát hiện ca phân vân helmet / no_helmet
        weak_helmets = [b for b, lb, c in raw if lb == "helmet" and c >= self.ambiguous_helmet_conf]
        heads = []
        for hbox, label, conf in raw:
            hx1, hy1, hx2, hy2 = hbox
            if hy2 - hy1 < self.min_head_px:
                continue
            if label == "no_helmet":
                if any(_iou(hbox, hb) >= self.ambiguous_iou for hb in weak_helmets):
                    label = "uncertain"            # có thể là mũ tối màu -> không bỏ phiếu
                elif conf < self.no_helmet_conf:
                    continue
            elif conf < self.helmet_conf:
                continue
            heads.append(((hx1 + hx2) / 2, (hy1 + hy2) / 2, hbox, label, conf))

        # Mô hình đôi khi gắn CẢ helmet lẫn no_helmet cho cùng một cái đầu -> chỉ giữ nhãn tin cậy hơn
        heads.sort(key=lambda h: -h[4])
        kept = []
        for h in heads:
            if all(_iou(h[2], k[2]) < 0.5 for k in kept):
                kept.append(h)
        heads = kept

        # Gắn mỗi đầu vào đúng 1 xe máy
        assigned = {tid: [] for tid, _, _ in motos}
        for cx, cy, hbox, label, conf in heads:
            best_tid, best_dist = None, None
            for tid, (x1, y1, x2, y2), _ in motos:
                w, h = x2 - x1, y2 - y1
                # Vùng đầu: hơi nhô lên trên mép xe, xuống tới ~45% chiều cao xe, rộng thêm 15% mỗi bên
                if not (x1 - 0.15 * w <= cx <= x2 + 0.15 * w and y1 - 0.25 * h <= cy <= y1 + 0.45 * h):
                    continue
                dist = abs(cx - (x1 + x2) / 2) + abs(cy - y1)
                if best_dist is None or dist < best_dist:
                    best_tid, best_dist = tid, dist
            if best_tid is not None:
                assigned[best_tid].append((hbox, label, conf))

        # 3. Bỏ phiếu theo thời gian
        new_violations = []
        for tid, bbox, cls_name in motos:
            t = self.tracks.setdefault(tid, {
                "status": STATUS_CHECKING, "cls_name": cls_name, "observed": 0, "no_helmet": 0, "conf_sum": 0.0,
                "w_total": 0.0, "w_no_helmet": 0.0, "decisive": 0,
                "bbox": bbox, "last_frame": frame_idx, "heads": [], "evidence_heads": [],
            })
            t.update(bbox=bbox, cls_name=cls_name, last_frame=frame_idx, heads=assigned[tid])
            # Đầu "uncertain" (phân vân, có thể là mũ đen) không góp phiếu nào
            voting = [hb for hb in assigned[tid] if hb[1] != "uncertain"]
            if not voting or t["status"] == STATUS_VIOLATION:
                continue

            t["observed"] += 1
            bare = [hb for hb in voting if hb[1] == "no_helmet"]
            # Trọng số phiếu = chiều cao khung đầu quyết định của frame này
            decisive = bare if bare else voting
            weight = max(hb[0][3] - hb[0][1] for hb in decisive)
            t["w_total"] += weight
            if bare:
                t["w_no_helmet"] += weight
                t["no_helmet"] += 1
                t["conf_sum"] += max(hb[2] for hb in bare)
                t["evidence_heads"] = [hb[0] for hb in bare]
                if weight >= self.min_decisive_px:
                    t["decisive"] += 1

            if (
                t["observed"] >= self.min_observe_frames
                and t["no_helmet"] >= self.min_no_helmet_frames
                and t["w_no_helmet"] / t["w_total"] >= self.no_helmet_ratio
                and t["decisive"] >= self.min_decisive_frames   # đủ frame thấy rõ đầu trần (đầu đủ to)
                and bare   # kết luận đúng frame đang thấy người không đội mũ => ảnh bằng chứng rõ ràng
            ):
                t["status"] = STATUS_VIOLATION
                t["violation_frame"] = frame_idx
                self.violation_count += 1
                new_violations.append({
                    "track_id": tid,
                    "cls_name": cls_name,
                    "bbox": bbox,
                    "head_boxes": list(t["evidence_heads"]),
                    "frame_idx": frame_idx,
                    "reason": REASON,
                    # Độ tin cậy TB của các lần thấy người không đội mũ (0..1)
                    "confidence": round(t["conf_sum"] / t["no_helmet"], 4),
                })
        return new_violations

    # ───────────────────────────── HIỂN THỊ ─────────────────────────────
    def violation_heads(self, frame_idx):
        """Khung đầu (lấy NGUYÊN từ mô hình helmet ở frame này) của những người đã bị kết luận không đội mũ.
        Không tự vẽ gì: draw_detections dùng chính các khung này để tô đậm + gắn nhãn vi phạm."""
        boxes = []
        for t in self.tracks.values():
            if t["status"] == STATUS_VIOLATION and t["last_frame"] == frame_idx:
                boxes += [hbox for hbox, label, _conf in t["heads"] if label == "no_helmet"]
        return boxes


# ───────────────────────────── ẢNH CẬN CẢNH NGƯỜI VI PHẠM ─────────────────────────────
def make_subject_crop(frame, bbox, head_boxes, orig_frame=None, min_height=240):
    """
    Cắt ảnh cận cảnh người trên xe máy (đầu + thân trên + phần trên xe) và khoanh (màu hồng) đầu không đội mũ.

    Args:
        frame: frame đang xử lý (toạ độ bbox / head_boxes tính trên frame này).
        bbox: (x1, y1, x2, y2) khung xe máy.
        head_boxes: [(x1, y1, x2, y2)] các đầu bị báo no_helmet.
        orig_frame: frame gốc chưa vẽ gì (ưu tiên dùng: ảnh sạch, có thể nét hơn nếu video > 720p).
        min_height: phóng to ảnh cắt tới chiều cao tối thiểu này để xem rõ trên web / sidebar.
    Returns:
        ảnh BGR hoặc None.
    """
    if bbox is None or frame is None:
        return None
    src = orig_frame if orig_frame is not None else frame
    h, w = frame.shape[:2]
    sh, sw = src.shape[:2]
    sx, sy = sw / float(w), sh / float(h)

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    top = min([y1] + [hb[1] for hb in head_boxes])
    head_h = max([hb[3] - hb[1] for hb in head_boxes] or [int(bh * 0.2)])
    # Vùng cắt: rộng thêm 20% mỗi bên, từ trên đầu một chút xuống ~75% chiều cao xe
    cx1 = max(0, int(x1 - 0.2 * bw))
    cx2 = min(w, int(x2 + 0.2 * bw))
    cy1 = max(0, int(top - 0.6 * head_h))
    cy2 = min(h, int(y1 + 0.75 * bh))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None

    crop = src[int(cy1 * sy):int(cy2 * sy), int(cx1 * sx):int(cx2 * sx)]
    if crop.size == 0:
        return None
    scale = max(1.0, min_height / float(crop.shape[0]))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        crop = crop.copy()

    # Khoanh đầu không đội mũ (vẽ SAU khi phóng to để nét viền sắc)
    kx, ky = scale * sx, scale * sy
    for hx1, hy1, hx2, hy2 in head_boxes:
        p1 = (int((hx1 - cx1) * kx) - 3, int((hy1 - cy1) * ky) - 3)
        p2 = (int((hx2 - cx1) * kx) + 3, int((hy2 - cy1) * ky) + 3)
        cv2.rectangle(crop, p1, p2, NO_HELMET_COLOR, max(2, int(round(scale))))
    return crop
