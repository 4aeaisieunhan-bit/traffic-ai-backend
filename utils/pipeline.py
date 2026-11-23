# utils/pipeline.py
# Pipeline phân tích 1 khung hình — DÙNG CHUNG cho:
#   - main.py           (chạy dòng lệnh, ghi output.mp4 + in bảng tổng kết)
#   - api/analyze.py    (phân tích từ giao diện web, lưu vi phạm vào database)
# => Thêm / sửa tính năng nhận diện chỉ cần sửa 1 chỗ, hai luồng luôn cho kết quả giống nhau.
#
# Mỗi khung hình: chuẩn hoá 1280x720 + tăng nét -> 5 mô hình YOLO -> vạch dừng -> theo dõi quỹ đạo (vượt đèn)
# -> biển số -> vi phạm không đội mũ (toàn khung hình) -> vẽ khung + HUD -> ghép sidebar.
from dataclasses import dataclass, field

import cv2

from models.weights import (
    HELMET_WEIGHTS, LICENSE_PLATE_WEIGHTS, OCR_WEIGHTS, STOP_LINE_WEIGHTS, TRAFFIC_LIGHT_WEIGHTS,
    VEHICLE_WEIGHTS,
)

# Mô hình phát hiện chạy trên mỗi khung hình: key -> (tên hiển thị, đường dẫn file weights, module)
# Đường dẫn lấy từ models/weights.py — đúng file mà các module models/*.py thực sự nạp.
MODEL_INFO = {
    "Vehicle": ("Vehicle Detector + ByteTrack", VEHICLE_WEIGHTS, "models.vehicle"),
    "Traffic Light": ("Traffic Light Classifier", TRAFFIC_LIGHT_WEIGHTS, "models.traffic_light"),
    "Helmet": ("Helmet Detector", HELMET_WEIGHTS, "models.helmet"),
    "License Plate": ("License Plate Detector", LICENSE_PLATE_WEIGHTS, "models.license_plate"),
    "Stop Line": ("Stop Line Detector", STOP_LINE_WEIGHTS, "models.stop_line"),
}
# Mô hình đọc ký tự biển số: chạy trên ảnh cắt biển số (không có thống kê theo khung hình)
OCR_INFO = ("Plate Character OCR", OCR_WEIGHTS, "models.ocr")

PROCESS_SIZE = (1280, 720)     # xử lý ở 1280x720: nhanh hơn ~5 lần so với 4K, ngưỡng thuật toán ổn định
SIDEBAR_WIDTH = 416            # 1280 + 416 = 1696 chia hết cho 16 -> nén video không lỗi macroblock


@dataclass
class FrameResult:
    canvas: "cv2.typing.MatLike"                 # khung hình đã vẽ + sidebar (ghi vào video kết quả)
    new_violations: list = field(default_factory=list)   # vi phạm MỚI ghi nhận ở khung hình này
    evidence_frame: "cv2.typing.MatLike | None" = None   # khung hình 720p (chưa vẽ HUD) để lưu ảnh bằng chứng


def reset_vehicle_tracker():
    """Xoá trạng thái ByteTrack còn sót từ video trước (model.track dùng persist=True)."""
    from models.vehicle import model as vehicle_model
    predictor = getattr(vehicle_model, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        tracker.reset()


class TrafficPipeline:
    def __init__(self, fps: float):
        # Nạp mô hình khi tạo pipeline (không làm chậm lúc khởi động server)
        from models.traffic_light import detect_traffic_light
        from models.license_plate import detect_license_plate
        from models.vehicle import detect_vehicle
        from models.stop_line import detect_stopline
        from models.helmet import detect_helmet
        from utils.traffic_violation import TrafficViolationTracker
        from utils.helmet_violation import HelmetViolationDetector
        from utils.violation_sidebar import ViolationSidebar
        from utils.ROI import StoplineStabilizer

        self._detect = {
            "Stop Line": detect_stopline,
            "Traffic Light": detect_traffic_light,
            "Helmet": detect_helmet,
            "License Plate": detect_license_plate,
            "Vehicle": lambda f: detect_vehicle(f, track=True),
        }
        self.fps = fps
        self.width, self.height = PROCESS_SIZE
        self.sidebar = ViolationSidebar(sidebar_width=SIDEBAR_WIDTH, height=self.height, fps=fps, max_cards=4)
        # Vượt đèn đỏ: quan sát quỹ đạo tối đa 5 s, lệch ngang 25 px = rẽ, tiến sâu 150 px = đi thẳng
        self.tracker = TrafficViolationTracker(fps=fps, observe_seconds=5.0, turn_threshold_px=25, deep_threshold_px=150)
        self.helmet_detector = HelmetViolationDetector(fps=fps)   # không đội mũ — toàn khung hình
        self.active_stopline = None                              # bbox vạch dừng đã làm mượt (dùng để vẽ)
        self.stopline_segment = None                             # đoạn thẳng vạch dừng ((xa, ya), (xb, yb))
        self.stopline_stabilizer = StoplineStabilizer()          # median theo thời gian, tự khóa khi ổn định
        self.stats = {k: {"detections": 0, "conf_sum": 0.0} for k in MODEL_INFO}
        self.no_helmet_conf: dict[int, float] = {}               # track_id -> độ tin cậy no_helmet TB (0..1)
        self.seen_track_ids: set[int] = set()                    # mọi ID xe ByteTrack đã gán -> "phương tiện đã quét"
        reset_vehicle_tracker()

    @property
    def output_size(self) -> tuple[int, int]:
        return self.width + self.sidebar.sidebar_width, self.height

    # ───────────────────────────── 1 khung hình ─────────────────────────────
    def process(self, raw_frame, frame_idx: int) -> FrameResult:
        from utils.draw_detections import NO_HELMET_COLOR, draw_detections
        from utils.frame_enhance import enhance_frame
        from utils.helmet_violation import make_subject_crop

        W, H = self.width, self.height
        orig_frame = raw_frame                                   # giữ khung gốc (có thể 4K) để crop biển số nét
        frame = raw_frame
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(orig_frame, (W, H), interpolation=cv2.INTER_AREA)
        frame = enhance_frame(frame, sharpen_amount=0.30, contrast_clip=1.4)

        # 5 mô hình YOLO
        res = {key: list(fn(frame)) for key, fn in self._detect.items()}
        for key, r in res.items():
            boxes = r[0].boxes if r else None
            if boxes is not None and len(boxes) > 0:
                self.stats[key]["detections"] += len(boxes)
                self.stats[key]["conf_sum"] += float(boxes.conf.sum())
        vehicle, light, helmet, plate = res["Vehicle"][0], res["Traffic Light"][0], res["Helmet"][0], res["License Plate"][0]
        if vehicle.boxes is not None and vehicle.boxes.id is not None:
            self.seen_track_ids.update(int(i) for i in vehicle.boxes.id.tolist())

        # Vạch dừng: chọn khung tin cậy nhất nằm trong khoảng 35–65% chiều cao, rồi đưa qua bộ ổn định
        # (median nhiều frame + khóa khi camera cố định; giữ lại khi mất dấu)
        sl = res["Stop Line"][0]
        detected = None
        if sl.boxes is not None and len(sl.boxes) > 0:
            valid = [b for b in sl.boxes if H * 0.35 <= float(b.xyxy[0][1]) <= H * 0.65]
            if valid:
                detected = tuple(map(int, max(valid, key=lambda b: float(b.conf[0])).xyxy[0].tolist()))
        self.active_stopline, self.stopline_segment = self.stopline_stabilizer.update(frame, detected)
        # Video chỉ vẽ 1 đường mảnh theo self.stopline_segment (draw_detections), không vẽ khung bbox

        # Vượt đèn đỏ (quỹ đạo) + biển số
        self.tracker.update(
            frame_idx=frame_idx, vehicle_boxes=vehicle.boxes, vehicle_names=vehicle.names,
            traffic_light_boxes=light.boxes, light_names=light.names,
            stopline_bbox=self.active_stopline, stopline_segment=self.stopline_segment,
        )
        rec_boxes = self.tracker.get_recognized_boxes(vehicle.boxes)
        self.sidebar.update_plates(rec_boxes, plate.boxes, frame, frame_idx, orig_frame=orig_frame)
        self.sidebar.refresh_violation_plates()   # vi phạm đã ghi: thay bằng biển số bình chọn mới nhất
        prev_count = len(self.sidebar.violations_history)
        self.sidebar.check_violations(self.tracker, frame, frame_idx, orig_frame=orig_frame)

        # Không đội mũ bảo hiểm (mọi xe máy trên toàn khung hình)
        for hv in self.helmet_detector.update(frame_idx, vehicle.boxes, vehicle.names, helmet.boxes, helmet.names):
            self.sidebar.record_violation(
                hv["track_id"], hv["cls_name"], hv["reason"], hv["bbox"], frame, frame_idx,
                orig_frame=orig_frame, kind="NO_HELMET",
                subject_crop=make_subject_crop(frame, hv["bbox"], hv["head_boxes"], orig_frame=orig_frame),
            )
            self.no_helmet_conf[hv["track_id"]] = hv["confidence"]

        new_items = self.sidebar.violations_history[: len(self.sidebar.violations_history) - prev_count]
        evidence_frame = frame.copy() if new_items else None     # ảnh bằng chứng: trước khi vẽ HUD / khung

        # Vẽ: quỹ đạo + HUD -> đèn, mũ (khung vi phạm = chính khung mũ của mô hình), vạch dừng
        frame = self.tracker.draw_visuals(
            frame, frame_idx,
            extra_lines=[("Không đội mũ bảo hiểm", self.helmet_detector.violation_count, NO_HELMET_COLOR)],
        )
        frame = draw_detections(
            frame,
            traffic_light_result=res["Traffic Light"], helmet_result=res["Helmet"],
            recognized_vehicle_boxes=rec_boxes, active_stopline=self.active_stopline,
            helmet_violation_boxes=self.helmet_detector.violation_heads(frame_idx),
            stopline_segment=self.stopline_segment,
        )
        canvas = self.sidebar.render(frame, frame_idx)
        return FrameResult(canvas=canvas, new_violations=new_items, evidence_frame=evidence_frame)

    # ───────────────────────────── thống kê ─────────────────────────────
    def avg_conf_pct(self, key: str) -> float | None:
        s = self.stats[key]
        return round(s["conf_sum"] / s["detections"] * 100, 1) if s["detections"] else None

    def model_stats(self) -> dict:
        """{key: {weight, detections, avg_confidence (0..1) | None}}"""
        from models.weights import weight_label
        return {
            key: {
                "weight": weight_label(MODEL_INFO[key][1]),
                "detections": s["detections"],
                "avg_confidence": round(s["conf_sum"] / s["detections"], 4) if s["detections"] else None,
            }
            for key, s in self.stats.items()
        }
