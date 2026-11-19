# Module quản lý, theo dõi quỹ đạo xe và phát hiện vi phạm vượt đèn đỏ
# Hỗ trợ Ý tưởng 3: Thiết lập ngưỡng thời gian quan sát quỹ đạo xe sau khi qua vạch dừng
# để phân biệt xe đi thẳng (vượt đèn đỏ) và xe rẽ phải/trái hợp lệ khi có đèn rẽ nhánh.

import cv2
import numpy as np
from collections import defaultdict, deque
from utils.ROI import horizontal_segment_from_bbox, signed_distance


class VehicleStatus:
    APPROACHING = "APPROACHING"           # Xe đang tiếp cận trước vạch dừng
    OBSERVING = "OBSERVING"               # Đang trong ngưỡng thời gian quan sát quỹ đạo (observe_seconds)
    VIOLATION = "VIOLATION"               # Xác nhận vi phạm vượt đèn đỏ
    LEGITIMATE = "LEGITIMATE"             # Xác nhận rẽ hợp lệ theo đèn tín hiệu nhánh
    LEGAL_PASS = "LEGAL_PASS"             # Qua vạch khi đèn xanh thẳng hợp lệ
    IGNORED = "IGNORED"                   # Xe trong ROI TOP không qua stopline (không nhận diện)


def parse_traffic_lights(traffic_light_boxes, class_names, conf_thresh=0.25):
    """
    Phân tích kết quả phát hiện đèn giao thông thành trạng thái boolean có cấu trúc.
    
    Args:
        traffic_light_boxes: ultralytics Boxes object hoặc list các box
        class_names: dict ánh xạ class_id -> class_name
        conf_thresh: ngưỡng tin cậy tối thiểu

    Returns:
        dict: {
            'red_light': bool,
            'green_light': bool,
            'green_right_arrow': bool,
            'green_left_arrow': bool,
            'red_left_arrow': bool,
            'red_right_arrow': bool,
            'yellow_light': bool,
            'yellow_left_arrow': bool,
            'yellow_right_arrow': bool,
            'active_labels': list[str]
        }
        Khóa trùng tên lớp của mô hình đèn (weights_demo/dengiaothong.pt).
    """
    state = {
        "red_light": False,
        "green_light": False,
        "green_right_arrow": False,
        "green_left_arrow": False,
        "red_left_arrow": False,
        "red_right_arrow": False,
        "yellow_light": False,
        "yellow_left_arrow": False,
        "yellow_right_arrow": False,
        "active_labels": [],
    }

    if traffic_light_boxes is None or len(traffic_light_boxes) == 0:
        return state

    for box in traffic_light_boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        cls_id = int(box.cls[0])
        label = class_names.get(cls_id, "")
        if label in state:
            state[label] = True
            if label not in state["active_labels"]:
                state["active_labels"].append(label)

    return state


def straight_allowed(lights):
    """Đi thẳng chỉ bị cấm khi đèn tròn đỏ. Mũi tên đỏ (trái / phải) chỉ cấm hướng rẽ tương ứng."""
    return not lights.get("red_light", False)


def turn_allowed(lights, direction):
    """
    Rẽ trái / phải có được phép không:
    - Mũi tên xanh của hướng đó -> được phép.
    - Mũi tên đỏ của hướng đó -> cấm (kể cả khi đèn tròn xanh).
    - Không có mũi tên -> theo đèn tròn (cấm khi đỏ).
    Không thấy đèn nào -> không có căn cứ kết luận vi phạm (giống đi thẳng).
    """
    side = "left" if direction == "LEFT" else "right"
    if lights.get(f"green_{side}_arrow", False):
        return True
    if lights.get(f"red_{side}_arrow", False):
        return False
    return not lights.get("red_light", False)


class TrafficViolationTracker:
    def __init__(
        self,
        fps=30.0,
        observe_seconds=5.0,
        turn_threshold_px=25,
        deep_threshold_px=150,
        history_len=90,
        line_band_px=6,
        cross_confirm_frames=3,
        straight_depth_px=60,
        right_guard_px=20,
    ):
        """
        Khởi tạo hệ thống theo dõi vi phạm và phân tích quỹ đạo xe.

        Args:
            fps (float): Tốc độ khung hình của video.
            observe_seconds (float): Ngưỡng thời gian quan sát tối đa (mặc định 5.0s).
            turn_threshold_px (int): Độ lệch ngang tối thiểu (pixel) để xác định xe rẽ phải hoặc rẽ trái (25px).
            deep_threshold_px (int): Độ sâu Y tiến vào tim giao lộ (pixel) để xác định xe đi thẳng vượt đèn đỏ (150px).
            history_len (int): Số frame tối đa lưu trữ lịch sử di chuyển của mỗi xe.
            line_band_px (int): Tâm đáy xe phải ở phía trên đường vạch dừng quá số pixel này mới tính là ROI TOP.
            cross_confirm_frames (int): Số frame liên tiếp ở ROI TOP để xác nhận xe đã qua vạch (chống rung bbox).
            straight_depth_px (int): Độ sâu tối thiểu (pixel) tiến vào giao lộ để kết luận ĐI THẲNG theo quỹ đạo.
                Đủ sâu để không nhầm với xe rẽ trái (xe rẽ trái cũng tiến thẳng một đoạn trước khi ngoặt).
            right_guard_px (int): Xe phải lệch sang phải so với hướng LÀN của chính nó (về điểm tụ) ít nhất
                số pixel này mới được tính là RẼ PHẢI. Chặn nhận nhầm xe làn trái đi thẳng / rẽ trái
                (trên ảnh các xe này trôi sang phải theo phối cảnh).
        """
        self.fps = max(float(fps), 1.0)
        self.observe_seconds = float(observe_seconds)
        self.max_observe_frames = max(int(self.fps * self.observe_seconds), 20)
        self.turn_threshold_px = int(turn_threshold_px)
        self.deep_threshold_px = int(deep_threshold_px)
        self.history_len = int(history_len)
        self.line_band_px = float(line_band_px)
        self.cross_confirm_frames = max(int(cross_confirm_frames), 1)
        self.straight_depth_px = float(straight_depth_px)
        self.right_guard_px = float(right_guard_px)
        # ByteTrack giữ ID tối đa 30 frame khi mất dấu -> quá ngưỡng này xe coi như đã mất hẳn
        self.lost_frames = max(31, int(self.fps * 0.6))

        # Đèn tín hiệu không biến mất giữa chừng: frame mô hình không thấy đèn nào thì giữ trạng thái
        # gần nhất tối đa 1 giây -> xe qua vạch đúng frame bị sót đèn vẫn được đối chiếu đúng.
        self.light_hold_frames = max(int(self.fps * 1.0), 1)
        self._held_lights = None
        self._held_lights_frame = None

        # Điểm tụ phối cảnh (vanishing point) của các LÀN TIẾP CẬN, ước lượng tự động từ quỹ đạo trước vạch
        # của các xe đã qua vạch (các làn song song ngoài thực tế hội tụ về 1 điểm trên ảnh).
        # Cho biết hướng làn tại từng vị trí: làn bên trái trôi sang phải trên ảnh, làn bên phải trôi sang trái.
        # CHỈ dùng để chặn rẽ phải giả — không dùng làm hướng "đi thẳng" vì đường ra phía sau giao lộ
        # có thể lệch so với các làn tiếp cận.
        self.approach_paths = defaultdict(lambda: deque(maxlen=150))   # {track_id: deque([(cx, cy), ...])}
        self.lane_lines = deque(maxlen=60)                              # [(vx, vy, x0, y0), ...]
        self.vanishing_point = None

        # Lịch sử tọa độ tâm đáy xe CHỈ trong vùng ROI TOP (khi đã qua stopline):
        # {track_id: deque([(cx, cy, frame_idx), ...])}
        self.trajectories = defaultdict(lambda: deque(maxlen=self.history_len))

        # Lịch sử vị trí ngắn (30 frame) để tính tốc độ di chuyển của xe:
        # {track_id: deque([(cx, cy), ...])}
        self.recent_positions = defaultdict(lambda: deque(maxlen=30))

        # Thông tin trạng thái của từng xe:
        # {track_id: {
        #     'status': VehicleStatus,
        #     'class_name': str,
        #     'start_frame': int,
        #     'start_pos': (cx, cy),
        #     'lights_at_crossing': dict,
        #     'direction': str,       # 'STRAIGHT', 'RIGHT', 'LEFT', 'STRAIGHT_OR_LEFT' (mất dấu)
        #     'reason': str,
        #     'resolved_frame': int,
        # }}
        self.vehicles = {}

        # Thống kê tổng hợp
        self.violation_count = 0
        self.legitimate_turn_count = 0   # rẽ phải + rẽ trái hợp lệ
        self.legitimate_right_count = 0
        self.legitimate_left_count = 0
        self.straight_legal_count = 0
        self.violation_ids = set()
        self.legitimate_ids = set()

    def is_vehicle_recognized(self, track_id):
        """
        Kiểm tra xem xe có đang được nhận diện / hiển thị hay không:
        - Các phương tiện đi từ làn khác vào thẳng ROI TOP: KHÔNG nhận diện.
        - CHỈ những xe từ ROI BOTTOM rồi đi qua vạch dừng: được nhận diện và theo dõi quỹ đạo.
        """
        if track_id not in self.vehicles:
            return False
        return bool(self.vehicles[track_id].get("is_recognized", False))

    def get_recognized_boxes(self, vehicle_boxes):
        """
        Lọc danh sách bounding boxes của xe chỉ giữ lại các xe đang được công nhận nhận diện.
        """
        if vehicle_boxes is None or len(vehicle_boxes) == 0:
            return []
        recognized = []
        for box in vehicle_boxes:
            if box.id is not None and self.is_vehicle_recognized(int(box.id[0])):
                recognized.append(box)
        return recognized

    def update(
        self,
        frame_idx,
        vehicle_boxes,
        vehicle_names,
        traffic_light_boxes,
        light_names,
        stopline_bbox,
        stopline_segment=None,
    ):
        """
        Cập nhật vị trí các xe, trạng thái đèn và phân tích quỹ đạo trong frame hiện tại.
        Quy tắc:
        1. Các phương tiện đi từ làn khác vào thẳng ROI TOP sẽ KHÔNG nhận diện phương tiện đó.
        2. CHỈ những xe ở ROI BOTTOM rồi đi qua vạch dừng thì bắt đầu theo dõi quỹ đạo xe
           trong ngưỡng thời gian observe_seconds (mặc định 5 giây).

        Vạch dừng là đoạn thẳng `stopline_segment` ((xa, ya), (xb, yb)); nếu không có thì lấy đường nằm ngang
        qua tâm `stopline_bbox`. ROI TOP / BOTTOM = hai phía của đường thẳng này (kéo dài hết khung hình).
        """
        light_state = parse_traffic_lights(traffic_light_boxes, light_names)
        if light_state["active_labels"]:
            self._held_lights, self._held_lights_frame = light_state, frame_idx
        elif self._held_lights is not None and frame_idx - self._held_lights_frame <= self.light_hold_frames:
            light_state = self._held_lights

        segment = stopline_segment
        if segment is None and stopline_bbox is not None:
            segment = horizontal_segment_from_bbox(stopline_bbox)

        current_active_ids = set()

        if vehicle_boxes is not None and len(vehicle_boxes) > 0:
            for box in vehicle_boxes:
                # Lấy track_id của xe
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                current_active_ids.add(track_id)

                cls_id = int(box.cls[0])
                vehicle_cls_name = vehicle_names.get(cls_id, "xe")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Tâm đáy xe (điểm tiếp xúc với mặt đường)
                cx = (x1 + x2) // 2
                cy = y2

                # Xe ở phía nào của vạch dừng: khoảng cách có dấu < 0 là phía giao lộ (ROI TOP)
                if segment is not None:
                    in_top_roi = signed_distance((cx, cy), segment) < -self.line_band_px
                    in_bottom_roi = not in_top_roi
                else:
                    in_top_roi = in_bottom_roi = False

                # Khởi tạo trạng thái cho xe mới xuất hiện
                if track_id not in self.vehicles:
                    if in_bottom_roi:
                        # Xe xuất hiện ở ROI BOTTOM -> Hợp lệ để theo dõi tiếp cận stopline
                        was_in_bottom = True
                        is_recognized = True
                        init_status = VehicleStatus.APPROACHING
                        reason = "Tiếp cận vạch dừng (ROI Bottom)"
                    elif in_top_roi:
                        # Xe xuất hiện trực tiếp trong ROI TOP (đi từ làn khác hoặc rẽ ngang vào):
                        # QUY TẮC: KHÔNG NHẬN DIỆN PHƯƠNG TIỆN ĐÓ!
                        was_in_bottom = False
                        is_recognized = False
                        init_status = VehicleStatus.IGNORED
                        reason = "Xe từ làn khác đi vào ROI TOP (Không nhận diện)"
                    else:
                        was_in_bottom = False
                        is_recognized = False
                        init_status = VehicleStatus.IGNORED
                        reason = "Ngoài vùng giám sát"

                    self.vehicles[track_id] = {
                        "status": init_status,
                        "class_name": vehicle_cls_name,
                        "start_frame": None,
                        "start_pos": None,
                        "lights_at_crossing": None,
                        "direction": "UNKNOWN",
                        "reason": reason,
                        "resolved_frame": None,
                        "bbox": (x1, y1, x2, y2),
                        "was_in_bottom": was_in_bottom,
                        "passed_stopline": False,
                        "is_recognized": is_recognized,
                        "last_frame": frame_idx,
                        "top_streak": 0,           # số frame liên tiếp ở phía ROI TOP (chưa xác nhận qua vạch)
                        "cross_candidate": None,   # (frame, (cx, cy), đèn) tại frame đầu tiên của chuỗi trên
                    }
                else:
                    self.vehicles[track_id]["bbox"] = (x1, y1, x2, y2)
                    self.vehicles[track_id]["class_name"] = vehicle_cls_name
                    self.vehicles[track_id]["last_frame"] = frame_idx

                v_info = self.vehicles[track_id]

                # Cập nhật lịch sử vị trí ngắn hạn (dùng để tính tốc độ di chuyển)
                self.recent_positions[track_id].append((cx, cy))

                # Nếu xe đang trong ROI BOTTOM thì cập nhật was_in_bottom = True
                if in_bottom_roi:
                    v_info["was_in_bottom"] = True
                    if not v_info.get("passed_stopline", False):
                        self.approach_paths[track_id].append((cx, cy))
                    # Cơ chế Dynamic Promotion: nếu trước đó xe bị gán IGNORED (ví dụ mới ló dạng ở góc rìa camera),
                    # nhưng nay đã tiến rõ ràng vào ROI BOTTOM (chưa qua stopline):
                    # Tự động chuyển thành xe hợp lệ để tiếp tục theo dõi
                    if not v_info.get("passed_stopline", False) and v_info.get("status") == VehicleStatus.IGNORED:
                        v_info["is_recognized"] = True
                        v_info["status"] = VehicleStatus.APPROACHING
                        v_info["reason"] = "Tiếp cận vạch dừng (ROI Bottom)"

                # A. KÍCH HOẠT BẮT ĐẦU THEO DÕI QUỸ ĐẠO:
                # "chỉ những xe ở ROI BOTTOM rồi đi qua vạch dừng thì bắt đầu theo dõi quỹ đạo xe trong ngưỡng thời gian là 5 giây"
                if v_info["was_in_bottom"] and not v_info.get("passed_stopline", False):
                    # Xe chuyển từ phía ROI BOTTOM sang phía ROI TOP và giữ nguyên `cross_confirm_frames`
                    # frame liên tiếp -> xác nhận đã qua vạch. Mốc thời gian / vị trí / đèn lấy ở frame ĐẦU TIÊN
                    # sang phía trên, nên độ trễ xác nhận không làm lệch phân tích quỹ đạo.
                    if in_top_roi:
                        if v_info.get("cross_candidate") is None:
                            v_info["cross_candidate"] = (frame_idx, (cx, cy), light_state.copy())
                        v_info["top_streak"] = v_info.get("top_streak", 0) + 1
                    else:
                        v_info["top_streak"] = 0
                        v_info["cross_candidate"] = None

                    if v_info["top_streak"] >= self.cross_confirm_frames:
                        cross_frame, cross_pos, cross_lights = v_info["cross_candidate"]
                        # Xe bắt đầu đi hẳn vào giao lộ → reset bộ đếm đứng yên
                        v_info["passed_stopline"] = True
                        v_info["status"] = VehicleStatus.OBSERVING
                        v_info["start_frame"] = cross_frame  # Mốc bắt đầu tính thời gian quan sát quỹ đạo
                        v_info["start_pos"] = cross_pos
                        v_info["lights_at_crossing"] = cross_lights
                        v_info["is_recognized"] = True
                        v_info["reason"] = f"Đi qua stopline -> Bắt đầu theo dõi ({self.observe_seconds}s)"
                        self.trajectories[track_id].append((cross_pos[0], cross_pos[1], cross_frame))
                        self._add_lane_line(track_id, cross_pos[1])

                # B. QUY TẮC NHẬN DIỆN PHƯƠNG TIỆN TRONG ROI TOP:
                # - Các phương tiện đi từ làn khác vào thẳng ROI TOP sẽ KHÔNG nhận diện.
                # - CHỈ những xe ở ROI BOTTOM rồi đi qua vạch dừng thì mới nhận diện và theo dõi quỹ đạo.
                if in_top_roi:
                    # (cross_candidate != None: xe vừa sang ROI TOP, đang chờ xác nhận qua vạch -> vẫn nhận diện)
                    if v_info["was_in_bottom"] and (
                        v_info.get("passed_stopline", False) or v_info.get("cross_candidate") is not None
                    ):
                        v_info["is_recognized"] = True
                        # Tiếp tục cập nhật quỹ đạo di chuyển trong ROI TOP
                        if v_info["status"] in (VehicleStatus.OBSERVING, VehicleStatus.VIOLATION, VehicleStatus.LEGITIMATE, VehicleStatus.LEGAL_PASS):
                            self.trajectories[track_id].append((cx, cy, frame_idx))
                    else:
                        # Xe từ làn khác đi vào ROI TOP -> Hoàn toàn KHÔNG NHẬN DIỆN!
                        v_info["is_recognized"] = False
                        v_info["status"] = VehicleStatus.IGNORED
                        v_info["reason"] = "Xe từ làn khác đi vào ROI TOP (Không nhận diện)"
                else:
                    # Xe ở ngoài ROI TOP (trong ROI BOTTOM hoặc đang dừng chờ trước vạch dừng):
                    if v_info["was_in_bottom"]:
                        v_info["is_recognized"] = True
                        if not v_info.get("passed_stopline", False):
                            v_info["status"] = VehicleStatus.APPROACHING
                    else:
                        v_info["is_recognized"] = False

                # 3. Xử lý xe đang trong thời gian QUAN SÁT QUỸ ĐẠO (observe_seconds):
                # Phân định hướng di chuyển dựa trên độ sâu tiến vào giao lộ và độ dạt ngang so với hướng đi thẳng
                if v_info["status"] == VehicleStatus.OBSERVING and v_info.get("is_recognized", False):
                    start_x, start_y = v_info["start_pos"]
                    elapsed_frames = frame_idx - v_info["start_frame"]

                    dx = cx - start_x
                    forward_depth = start_y - cy   # Xe di chuyển lên trên (cy < start_y) -> forward_depth dương

                    # Hướng trục đường tự nhiên theo phối cảnh camera:
                    # Khi tiến lên giao lộ (-dy), làn đường có góc nghiêng phối cảnh dịch sang trái ~ -0.80 * forward_depth.
                    # Độ dạt ngang thực tế so với trục làn đường:
                    lateral_offset = dx + 0.80 * forward_depth
                    # Độ dạt ngang so với hướng làn tiếp cận của chính xe (None khi chưa ước lượng được điểm tụ)
                    lane_offset = self._lane_offset(start_x, start_y, dx, forward_depth)

                    # 1. Điều kiện phát hiện RẼ PHẢI:
                    # - Xe rẽ phải bắt buộc phải dạt sang bên phải (dx > 15px) và dạt mạnh so với trục đường (lateral_offset >= 40px)
                    # - Hoặc độ dạt sang phải cực lớn (dx >= 45px khi đã nhích được forward_depth >= 15px)
                    # - VÀ lệch phải so với hướng làn của chính xe (lane_offset >= right_guard_px): xe làn trái đi thẳng
                    #   hoặc rẽ trái cũng trôi sang phải trên ảnh nhưng KHÔNG lệch khỏi hướng làn của nó
                    is_turn_right = (
                        dx > 15 and forward_depth >= 15 and (lateral_offset >= 40 or dx >= 45)
                        and (lane_offset is None or lane_offset >= self.right_guard_px)
                    )

                    # 2. Điều kiện phát hiện RẼ TRÁI:
                    # - Lệch hẳn sang trái so với trục đường: lateral_offset <= -50px khi forward_depth >= 20px HOẶC
                    # - Xuất phát từ làn rẽ trái cạnh dải phân cách (start_x <= 450) và dạt mạnh sang trái (dx <= -50px)
                    is_turn_left = (forward_depth >= 20 and lateral_offset <= -50) or (
                        start_x <= 450 and dx <= -50 and forward_depth >= 15
                    )

                    # 3. ĐI THẲNG:
                    # - Bám sát hướng đi thẳng và đã tiến đủ sâu (straight_depth_px, |lateral_offset| < 35px)
                    # - HOẶC đã tiến sâu qua ngưỡng deep_threshold_px mà không rẽ
                    # - HOẶC đã hết thời gian quan sát (observe_seconds) và đã tiến về phía trước >= 25px
                    not_turning = not is_turn_right and not is_turn_left
                    is_straight_track = forward_depth >= self.straight_depth_px and abs(lateral_offset) < 35
                    is_straight_deep = forward_depth >= self.deep_threshold_px and not_turning
                    is_straight_timeout = elapsed_frames >= self.max_observe_frames and forward_depth >= 25 and not_turning

                    # Kiểm tra thêm: xe qua vạch dừng KHI ĐÈN ĐỎ rồi đứng yên trong giao lộ quá observe_seconds.
                    # Chỉ xét đèn lúc qua vạch: xe vào giao lộ khi đèn xanh rồi dừng chờ (VD chờ rẽ trái)
                    # trong lúc đèn chuyển đỏ là hợp lệ.
                    cross_lights = v_info.get("lights_at_crossing") or light_state
                    is_red_cross = not straight_allowed(cross_lights)
                    is_still_timeout_on_red = elapsed_frames >= self.max_observe_frames and is_red_cross and not_turning

                    direction = None
                    if is_turn_right:
                        direction = "RIGHT"
                    elif is_turn_left:
                        direction = "LEFT"
                    elif is_straight_track or is_straight_deep or is_straight_timeout or is_still_timeout_on_red:
                        direction = "STRAIGHT"

                    if direction is not None:
                        stood_still = is_still_timeout_on_red and forward_depth < 25
                        self._resolve(track_id, v_info, direction, cross_lights, frame_idx, stood_still=stood_still)

        # 4. Xe đang QUAN SÁT nhưng bị mất dấu (YOLO không còn phát hiện được, VD bị xe khác che khi đang rẽ):
        # kết luận theo phần quỹ đạo đã thấy thay vì để kẹt ở OBSERVING mãi mãi.
        for track_id, v_info in self.vehicles.items():
            if (
                v_info["status"] != VehicleStatus.OBSERVING
                or v_info.get("lost")
                or track_id in current_active_ids
                or frame_idx - v_info["last_frame"] < self.lost_frames
            ):
                continue
            v_info["lost"] = True
            traj = self.trajectories.get(track_id)
            if not traj:
                continue
            start_x, start_y = v_info["start_pos"]
            last_x, last_y, _ = traj[-1]
            dx = last_x - start_x
            forward_depth = start_y - last_y
            if forward_depth < 15:
                v_info["reason"] = "Mat dau truoc khi du du lieu quy dao"
                continue
            lane_offset = self._lane_offset(start_x, start_y, dx, forward_depth)
            if lane_offset is not None:
                is_right = dx > 15 and lane_offset >= self.right_guard_px
            else:
                is_right = dx > 15 and dx + 0.80 * forward_depth >= 40
            cross_lights = v_info["lights_at_crossing"] or light_state
            if is_right:
                # Rẽ phải lệch sang phải rất sớm và rõ -> đủ tin cậy dù quỹ đạo ngắn
                self._resolve(track_id, v_info, "RIGHT", cross_lights, frame_idx, note=" (mat dau)")
                continue
            # Đoạn đầu của xe rẽ trái trông giống hệt xe đi thẳng (cùng tiến thẳng vào giao lộ trước khi ngoặt)
            # -> không đoán hướng. Chỉ kết luận khi đi thẳng và rẽ trái cho CÙNG kết quả theo đèn.
            if straight_allowed(cross_lights) == turn_allowed(cross_lights, "LEFT"):
                self._resolve(track_id, v_info, "STRAIGHT_OR_LEFT", cross_lights, frame_idx, note=" (mat dau)")
            else:
                v_info["reason"] = "Mat dau: chua phan biet duoc di thang / re trai"

        return light_state

    # ───────────────────────────── Phối cảnh: điểm tụ của các làn ─────────────────────────────

    def _add_lane_line(self, track_id, stopline_y):
        """Fit đường thẳng qua quỹ đạo tiếp cận của xe vừa qua vạch rồi cập nhật lại điểm tụ."""
        path = self.approach_paths.pop(track_id, None)
        if not path or len(path) < 15:
            return
        pts = np.asarray(path, dtype=np.float32)
        if float(np.ptp(pts[:, 1])) < 60:          # quãng tiếp cận quá ngắn -> hướng không tin cậy
            return
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        if abs(vy) < 1e-6 or abs(vx / vy) > 1.5:    # làn tiếp cận phải gần dọc ảnh (bỏ xe chuyển làn ngang)
            return
        self.lane_lines.append((float(vx), float(vy), float(x0), float(y0)))
        self._update_vanishing_point(stopline_y)

    def _update_vanishing_point(self, stopline_y):
        """Giao điểm bình phương tối thiểu của các làn; loại làn lệch nhiều (xe chuyển làn) rồi tính lại."""
        lines = np.asarray(self.lane_lines, dtype=np.float64)
        if len(lines) < 3:
            return
        normals = np.column_stack([-lines[:, 1], lines[:, 0]])          # pháp tuyến đơn vị của từng làn
        offsets = np.einsum("ij,ij->i", normals, lines[:, 2:4])
        keep = np.ones(len(lines), dtype=bool)
        vp = None
        for _ in range(2):
            if keep.sum() < 3:
                return
            vp = np.linalg.lstsq(normals[keep], offsets[keep], rcond=None)[0]
            residual = np.abs(normals @ vp - offsets)                     # khoảng cách từ điểm tụ tới từng làn
            keep = residual <= max(40.0, float(np.median(residual)) * 3)
        # Điểm tụ phải nằm phía trên vạch dừng (phía xa camera)
        if vp is not None and vp[1] < stopline_y - 50:
            self.vanishing_point = (float(vp[0]), float(vp[1]))

    def _lane_offset(self, start_x, start_y, dx, forward_depth):
        """
        Độ dạt ngang (pixel, dương = sang phải) so với hướng làn tiếp cận tại điểm qua vạch (start_x, start_y),
        tức hướng về điểm tụ. None nếu chưa ước lượng được điểm tụ.
        """
        if self.vanishing_point is None:
            return None
        vp_x, vp_y = self.vanishing_point
        if start_y - vp_y < 20:
            return None
        return dx - (vp_x - start_x) / (start_y - vp_y) * forward_depth

    # ───────────────────────────── Kết luận hướng đi + đối chiếu đèn ─────────────────────────────

    def _resolve(self, track_id, v_info, direction, cross_lights, frame_idx, stood_still=False, note=""):
        """
        Gán hướng đi đã xác định cho xe và đối chiếu với tín hiệu đèn TẠI THỜI ĐIỂM VƯỢT VẠCH (cross_lights).
        Không dùng đèn hiện tại: xe vào giao lộ khi đèn xanh / vàng rồi đèn mới chuyển đỏ là hợp lệ.
        """
        v_info["direction"] = direction
        v_info["resolved_frame"] = frame_idx

        if direction == "STRAIGHT_OR_LEFT":
            # Chỉ gọi khi đi thẳng và rẽ trái cùng kết quả theo đèn tại thời điểm vượt vạch
            if not straight_allowed(cross_lights):
                self._mark_violation(track_id, v_info, "Vi pham: Di thang / re trai khi den do" + note)
            else:
                v_info["status"] = VehicleStatus.LEGAL_PASS
                v_info["reason"] = "Di thang / re trai hop le (den xanh)" + note
                self.straight_legal_count += 1
            return

        if direction == "STRAIGHT":
            if not straight_allowed(cross_lights):
                if stood_still:
                    reason = f"Vi pham: Qua stopline va dung yen qua {self.observe_seconds:.1f}s khi den do"
                else:
                    reason = "Vi pham: Di thang khi den do"
                self._mark_violation(track_id, v_info, reason + note)
            else:
                v_info["status"] = VehicleStatus.LEGAL_PASS
                v_info["reason"] = "Di thang hop le (den xanh)" + note
                self.straight_legal_count += 1
            return

        side = "phai" if direction == "RIGHT" else "trai"
        if turn_allowed(cross_lights, direction):
            v_info["status"] = VehicleStatus.LEGITIMATE
            v_info["reason"] = f"Re {side} hop le (co den re {side})" + note
            if track_id not in self.legitimate_ids:
                self.legitimate_ids.add(track_id)
                self.legitimate_turn_count += 1
                if direction == "RIGHT":
                    self.legitimate_right_count += 1
                else:
                    self.legitimate_left_count += 1
        else:
            self._mark_violation(track_id, v_info, f"Vi pham: Re {side} khi den do" + note)

    def _mark_violation(self, track_id, v_info, reason):
        v_info["status"] = VehicleStatus.VIOLATION
        v_info["reason"] = reason
        if track_id not in self.violation_ids:
            self.violation_ids.add(track_id)
            self.violation_count += 1

    def draw_visuals(self, frame, frame_idx, extra_lines=None):
        """extra_lines: [(text, bgr_color)] thêm vào cuối bảng HUD (VD: số xe không đội mũ bảo hiểm)."""
        extra_lines = extra_lines or []
        """
        Vẽ toàn bộ hình ảnh trực quan:
        1. Vết quỹ đạo xe (Trajectory paths)
        2. Bounding box phân màu theo trạng thái (Xanh / Vàng theo dõi / Đỏ vi phạm / Xanh ngọc hợp lệ)
        3. Dashboard HUD thống kê trên cùng góc phải
        """
        h, w = frame.shape[:2]

        # 1. Vẽ các đường quỹ đạo di chuyển (chỉ vẽ xe được công nhận nhận diện)
        for track_id, info in self.vehicles.items():
            if not info.get("is_recognized", False):
                continue

            pts = list(self.trajectories[track_id])
            if len(pts) < 2:
                continue

            status = info["status"]
            if status == VehicleStatus.VIOLATION:
                path_color = (0, 0, 255)       # Đỏ: vi phạm
                thickness = 2
            elif status == VehicleStatus.LEGITIMATE:
                path_color = (0, 255, 128)     # Xanh ngọc: hợp lệ rẽ
                thickness = 1
            elif status == VehicleStatus.OBSERVING:
                path_color = (0, 215, 255)     # Vàng/Cam: đang theo dõi
                thickness = 1
            else:
                path_color = (200, 200, 200)   # Xám: bình thường
                thickness = 1

            for i in range(1, len(pts)):
                pt1 = (pts[i - 1][0], pts[i - 1][1])
                pt2 = (pts[i][0], pts[i][1])
                cv2.line(frame, pt1, pt2, path_color, thickness)

        # 2. Vẽ Bounding box và Nhãn trạng thái cho các xe đang xuất hiện (chỉ vẽ xe được công nhận nhận diện)
        for track_id, info in self.vehicles.items():
            if not info.get("is_recognized", False):
                continue

            # Chỉ vẽ nếu xe vừa xuất hiện gần đây (trong 2 frame)
            last_f = info.get("last_frame", 0)
            if frame_idx - last_f > 2:
                continue


            bbox = info.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            status = info["status"]
            cls_name = info.get("class_name", "xe")

            direction = info.get("direction", "UNKNOWN")
            # Map direction sang nhãn tiếng Việt
            dir_label = {"RIGHT": "Re phai", "LEFT": "Re trai", "STRAIGHT": "Di thang",
                         "STRAIGHT_OR_LEFT": "Thang/Re trai"}.get(direction, "")

            if status == VehicleStatus.VIOLATION:
                box_color = (0, 0, 255)
                dir_str = f" [{dir_label}]" if dir_label else ""
                tag = f"ID:{track_id} {cls_name}{dir_str} - VI PHAM VUOT DEN DO"
                bg_color = (0, 0, 200)
            elif status == VehicleStatus.LEGITIMATE:
                box_color = (0, 255, 128)
                dir_str = dir_label if dir_label else "Re"
                tag = f"ID:{track_id} {cls_name} - {dir_str} (HOP LE)"
                bg_color = (0, 160, 80)
            elif status == VehicleStatus.LEGAL_PASS:
                box_color = (0, 220, 100)
                dir_str = dir_label if dir_label else "Di thang"
                tag = f"ID:{track_id} {cls_name} - {dir_str} (HOP LE)"
                bg_color = (0, 140, 60)
            elif status == VehicleStatus.OBSERVING:
                box_color = (0, 215, 255)
                elapsed = frame_idx - (info["start_frame"] or frame_idx)
                time_left = max(0.0, (self.max_observe_frames - elapsed) / self.fps)
                dir_str = f" [{dir_label}]" if dir_label else ""
                tag = f"ID:{track_id} {cls_name}{dir_str} - Theo doi ({time_left:.1f}s)"
                bg_color = (0, 140, 200)
            else:
                box_color = (0, 255, 0)
                tag = f"ID:{track_id} {cls_name}"
                bg_color = (0, 180, 0)

            # Vẽ bounding box mỏng tinh tế (thickness=1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

            # Vẽ nhãn nhỏ gọn thanh thoát, không chiếm nhiều diện tích
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            label_y = max(y1 - 4, th + 3)
            cv2.rectangle(
                frame,
                (x1, label_y - th - 3),
                (min(w - 1, x1 + tw + 6), label_y + 3),
                bg_color,
                -1,
            )
            cv2.putText(
                frame,
                tag,
                (x1 + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Bảng thông tin góc trên bên phải (cùng phong cách thanh bên: nền tối, chữ có dấu, số canh phải)
        self._draw_hud(frame, frame_idx, extra_lines)
        return frame

    def _draw_hud(self, frame, frame_idx, extra_lines):
        """
        extra_lines: các hàng thêm [(nhãn, giá trị, màu chấm BGR)] — VD: ("Không đội mũ bảo hiểm", 5, màu hồng).
        Vẫn nhận kiểu cũ (chuỗi, màu): khi đó cả chuỗi là nhãn, không có giá trị.
        """
        from utils.text_render import TextLayer

        h, w = frame.shape[:2]
        c_t1, c_t2, c_t3 = (243, 239, 237), (179, 168, 164), (143, 130, 125)
        c_line, c_signal, c_ok = (48, 44, 41), (77, 72, 229), (138, 195, 76)

        # Chỉ tính xe đang xuất hiện gần đây
        num_observing = sum(
            1 for v in self.vehicles.values()
            if v["status"] == VehicleStatus.OBSERVING and v.get("is_recognized", False)
            and frame_idx - v.get("last_frame", 0) <= 5
        )
        rows = [
            ("Đang theo dõi quỹ đạo", str(num_observing), c_t3),
            ("Vượt đèn đỏ", str(self.violation_count), c_signal),
            (f"Rẽ hợp lệ (phải {self.legitimate_right_count}, trái {self.legitimate_left_count})",
             str(self.legitimate_turn_count), c_ok),
        ]
        for extra in extra_lines:
            if len(extra) == 3:
                rows.append((extra[0], str(extra[1]), extra[2]))
            else:
                rows.append((extra[0], "", extra[1]))

        pad, row_h, head_h = 14, 24, 36
        panel_w = 280
        panel_h = head_h + row_h * len(rows) + 8
        px2, py1 = w - 16, 16
        px1, py2 = px2 - panel_w, py1 + panel_h

        # Nền tối gần như đặc (video vẫn thấp thoáng), viền mảnh, bo góc
        roi = frame[py1:py2, px1:px2]
        dark = np.full_like(roi, (13, 9, 8))
        cv2.addWeighted(dark, 0.86, roi, 0.14, 0, roi)
        r = 10
        for (cx, cy), ang in (((px1 + r, py1 + r), 180), ((px2 - r, py1 + r), 270), ((px1 + r, py2 - r), 90), ((px2 - r, py2 - r), 0)):
            cv2.ellipse(frame, (cx, cy), (r, r), ang, 0, 90, c_line, 1, cv2.LINE_AA)
        cv2.line(frame, (px1 + r, py1), (px2 - r, py1), c_line, 1)
        cv2.line(frame, (px1 + r, py2), (px2 - r, py2), c_line, 1)
        cv2.line(frame, (px1, py1 + r), (px1, py2 - r), c_line, 1)
        cv2.line(frame, (px2, py1 + r), (px2, py2 - r), c_line, 1)

        T = TextLayer()
        T.add(px1 + pad, py1 + 23, "Phân tích quỹ đạo", c_t1, 13, "sans-medium")
        T.add(px2 - pad, py1 + 23, f"ngưỡng {self.observe_seconds:g} s", c_t3, 12, align="right")
        cv2.line(frame, (px1 + pad, py1 + head_h - 2), (px2 - pad, py1 + head_h - 2), c_line, 1)

        for i, (label, value, color) in enumerate(rows):
            base = py1 + head_h + 17 + i * row_h
            cv2.circle(frame, (px1 + pad + 3, base - 4), 3, color, -1, cv2.LINE_AA)
            T.add(px1 + pad + 14, base, label, c_t2, 12)
            T.add(px2 - pad, base, value, c_t1, 13, "mono", align="right")

        T.flush(frame, px1, px2 + 1)
