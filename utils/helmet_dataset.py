# utils/helmet_dataset.py
# Thu thập dữ liệu huấn luyện lại mô hình helmet TỪ VIỆC DUYỆT VI PHẠM trên web.
#
# 1. Khi phát hiện vi phạm "không đội mũ", api/analyze.py gọi save_sample(): lưu khung hình SẠCH (đúng ảnh mô hình
#    nhìn thấy: 1280x720 đã tăng nét, chưa vẽ gì) + mọi khung đầu mô hình helmet phát hiện ở frame đó
#    + khung đầu của người bị kết luận vi phạm  ->  static/evidence/train_<task>_<track>_helmet.jpg / .json
# 2. Cán bộ duyệt hồ sơ (api/violations.py gọi apply_review()):
#      - "Đã xác nhận"                          -> đầu người vi phạm gán nhãn no_helmet (mô hình đúng)
#      - "Từ chối" + lý do "Có đội mũ"           -> đầu người vi phạm gán nhãn helmet   (mô hình nhầm — dữ liệu quý nhất)
#      - trạng thái khác / từ chối vì lý do khác -> bỏ mẫu khỏi bộ dữ liệu
#    Mẫu được ghi theo định dạng YOLO vào dataset/helmet_feedback/{images,labels}/.
#    Các đầu KHÁC trong khung hình dùng nhãn sẵn của mô hình (tin cậy cao); đầu mô hình không chắc chắn bị tô xám
#    để khi train không bị coi là "nền" (YOLO không có nhãn "bỏ qua vùng này").
# 3. tools/helmet_train.py gộp thư mục này với dữ liệu khác để train tiếp mô hình.
import json
import os

import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_DIR = os.path.join(BASE_DIR, "static", "evidence")
DATASET_DIR = os.getenv("HELMET_FEEDBACK_DIR", os.path.join(BASE_DIR, "dataset", "helmet_feedback"))

CLASS_NAMES = ["helmet", "no_helmet"]          # đúng thứ tự lớp của weights/helmet_26s.pt
PSEUDO_LABEL_CONF = 0.45                       # đầu khác trong khung: >= ngưỡng này dùng nhãn của mô hình
MASK_MIN_CONF = 0.25                           # dưới PSEUDO_LABEL_CONF nhưng >= ngưỡng này: tô xám (không chắc)

REASON_HAS_HELMET = "has_helmet"               # lý do từ chối: người lái có đội mũ, mô hình nhận nhầm
REASON_LABELS = {REASON_HAS_HELMET: "Có đội mũ — mô hình nhận nhầm (đã thêm vào dữ liệu huấn luyện)"}


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ───────────────────────────── mẫu lưu lúc phân tích ─────────────────────────────
def sample_paths(evidence_url: str | None) -> tuple[str, str] | None:
    """'/static/evidence/evid_<task>_<tid>_helmet.jpg' -> (đường dẫn ảnh, đường dẫn json) của mẫu huấn luyện."""
    if not evidence_url:
        return None
    name = os.path.basename(evidence_url)
    if not name.startswith("evid_"):
        return None
    stem = "train_" + os.path.splitext(name)[0][len("evid_"):]
    return os.path.join(EVIDENCE_DIR, stem + ".jpg"), os.path.join(EVIDENCE_DIR, stem + ".json")


def helmet_detections(helmet_result) -> list:
    """Kết quả YOLO helmet -> [(x1, y1, x2, y2), nhãn, conf] (dữ liệu thuần, lưu được ra json)."""
    out = []
    boxes = helmet_result.boxes
    if boxes is None:
        return out
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        out.append(((x1, y1, x2, y2), helmet_result.names[int(box.cls[0])], round(float(box.conf[0]), 4)))
    return out


def save_sample(evidence_fname: str, clean_frame, detections: list, violator_heads: list) -> None:
    paths = sample_paths(f"/static/evidence/{evidence_fname}")
    if paths is None or clean_frame is None or not violator_heads:
        return
    img_path, json_path = paths
    cv2.imwrite(img_path, clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = clean_frame.shape[:2]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "width": w, "height": h,
            "detections": [[*box, label, conf] for box, label, conf in detections],
            "violator_heads": [list(map(int, b)) for b in violator_heads],
        }, f)


# ───────────────────────────── ghi nhãn khi duyệt ─────────────────────────────
def _dataset_name(v) -> str:
    return f"{v.task_id}_{v.track_id}"


def _dataset_files(name: str) -> tuple[str, str, str]:
    return (os.path.join(DATASET_DIR, "images", name + ".jpg"),
            os.path.join(DATASET_DIR, "labels", name + ".txt"),
            os.path.join(DATASET_DIR, "meta", name + ".json"))


def is_helmet_violation(v) -> bool:
    vt = getattr(v, "violation_type", None)
    return getattr(vt, "code", None) == "NO_HELMET"


def has_sample(v) -> bool:
    paths = sample_paths(v.evidence_image)
    return bool(paths and os.path.isfile(paths[0]) and os.path.isfile(paths[1]))


def dataset_label(v) -> str | None:
    """Nhãn đã ghi vào bộ dữ liệu cho người vi phạm ('helmet' / 'no_helmet'), None nếu chưa có."""
    meta = _dataset_files(_dataset_name(v))[2]
    if not os.path.isfile(meta):
        return None
    try:
        with open(meta, encoding="utf-8") as f:
            return json.load(f).get("violator_label")
    except (OSError, ValueError):
        return None


def label_for_review(status: str, reason_code: str | None) -> str | None:
    if status == "Đã xác nhận":
        return "no_helmet"
    if status == "Từ chối" and reason_code == REASON_HAS_HELMET:
        return "helmet"
    return None


def apply_review(v, status: str, reason_code: str | None) -> str | None:
    """Ghi / xoá mẫu trong bộ dữ liệu theo kết quả duyệt. Trả về nhãn đã ghi (hoặc None)."""
    if not is_helmet_violation(v):
        return None
    name = _dataset_name(v)
    img_out, lbl_out, meta_out = _dataset_files(name)
    label = label_for_review(status, reason_code)
    paths = sample_paths(v.evidence_image)
    if label is None or not paths or not os.path.isfile(paths[0]) or not os.path.isfile(paths[1]):
        for p in (img_out, lbl_out, meta_out):
            if os.path.isfile(p):
                os.remove(p)
        return None

    img_path, json_path = paths
    with open(json_path, encoding="utf-8") as f:
        sample = json.load(f)
    image = cv2.imread(img_path)
    if image is None:
        return None
    h, w = image.shape[:2]
    violators = [tuple(b) for b in sample["violator_heads"]]

    # Đầu khác: gộp nhãn trùng (giữ tin cậy cao hơn), bỏ đầu trùng người vi phạm
    others = []
    for x1, y1, x2, y2, lb, conf in sorted(sample["detections"], key=lambda d: -d[5]):
        box = (x1, y1, x2, y2)
        if any(box_iou(box, vb) >= 0.3 for vb in violators) or any(box_iou(box, o[0]) >= 0.5 for o in others):
            continue
        others.append((box, lb, conf))

    lines = []
    for box in violators:
        lines.append((CLASS_NAMES.index(label), box))
    for box, lb, conf in others:
        if conf >= PSEUDO_LABEL_CONF and lb in CLASS_NAMES:
            lines.append((CLASS_NAMES.index(lb), box))
        elif conf >= MASK_MIN_CONF:
            # Không chắc -> tô xám để mô hình không học sai (không gán nhãn, cũng không thành "nền")
            x1, y1, x2, y2 = box
            image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 114

    for p in (img_out, lbl_out, meta_out):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(img_out, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(lbl_out, "w", encoding="utf-8") as f:
        for cls, (x1, y1, x2, y2) in lines:
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            f.write(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump({"violation_code": v.violation_code, "violator_label": label, "status": status,
                   "reason": reason_code, "source_image": os.path.basename(img_path)}, f, ensure_ascii=False)
    classes = os.path.join(DATASET_DIR, "classes.txt")
    if not os.path.isfile(classes):
        with open(classes, "w", encoding="utf-8") as f:
            f.write("\n".join(CLASS_NAMES) + "\n")
    return label


def remove_sample_files(v) -> None:
    """Xoá mẫu thô trong static/evidence khi xoá vi phạm (mẫu đã ghi vào dataset/ được GIỮ LẠI)."""
    paths = sample_paths(v.evidence_image)
    for p in paths or ():
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass
