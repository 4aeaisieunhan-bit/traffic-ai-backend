# tools/helmet_eval.py
# SO SÁNH mô hình helmet cũ / mới trên CÙNG một bộ test (ảnh mô hình chưa từng thấy khi train).
#
#   python tools/helmet_eval.py --data dataset/helmet_merged/data.yaml --weights weights/helmet_26s.pt runs/detect/helmet_ft/weights/best.pt
#
# Chỉ số quan trọng nhất với hệ thống phạt nguội:
#   - "Mũ -> báo KHÔNG MŨ": tỷ lệ đầu CÓ đội mũ bị mô hình gán no_helmet  => phạt nhầm (càng thấp càng tốt)
#   - "Bắt đúng không mũ"  : tỷ lệ đầu KHÔNG mũ được mô hình gán no_helmet => bỏ sót (càng cao càng tốt)
# Đo ở đúng ngưỡng hệ thống đang dùng (no_helmet >= 0.55, helmet >= 0.35 — utils/helmet_violation.py).
import argparse
import glob
import os
import sys
from typing import Any

import cv2
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLASS_NAMES = ["helmet", "no_helmet"]          # đúng thứ tự lớp của weights/helmet_26s.pt


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


THRESH = {"helmet": 0.35, "no_helmet": 0.55}


def load_yaml(path: str) -> dict[str, Any]:
    """Đọc data.yaml -> dict (file rỗng / không phải dict -> {})."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _split_images(data_yaml: str, split: str) -> list[str]:
    cfg = load_yaml(data_yaml)
    base = cfg.get("path") or os.path.dirname(os.path.abspath(data_yaml))
    entries = cfg.get(split) or []
    entries = entries if isinstance(entries, list) else [entries]
    images = []
    for e in entries:
        d = os.path.normpath(e if os.path.isabs(e) else os.path.join(base, e))
        images += sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
    return images


def _read_labels(img_path: str, w: int, h: int) -> list:
    lbl = os.path.normpath(img_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").rsplit(".", 1)[0] + ".txt"
    out = []
    if os.path.isfile(lbl):
        with open(lbl, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) != 5:
                    continue
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                out.append((CLASS_NAMES[c], (int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                                             int((cx + bw / 2) * w), int((cy + bh / 2) * h))))
    return out


def head_confusion(weights: str, images: list[str], min_head: int = 7) -> dict:
    """Ghép mỗi đầu thật (nhãn) với dự đoán IoU >= 0.4 ở ngưỡng hệ thống -> bảng nhầm lẫn helmet / no_helmet."""
    from ultralytics import YOLO
    from models.yolo_types import INFER_KW, as_results
    model = YOLO(weights)
    m = {(g, p): 0 for g in CLASS_NAMES for p in (*CLASS_NAMES, "missed")}
    false_alarm = 0   # no_helmet ở chỗ không có đầu nào
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = [g for g in _read_labels(path, w, h) if g[1][3] - g[1][1] >= min_head]
        r = as_results(model.predict(img, conf=0.2, verbose=False, **INFER_KW))[0]
        preds = []
        if r.boxes is not None:
            # Đọc thẳng các tensor (Boxes không khai báo __iter__ nên Pylance không cho lặp trực tiếp)
            dets = zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
            for xyxy, conf, cls in sorted(dets, key=lambda d: -d[1]):
                lb = r.names[int(cls)]
                box = tuple(int(v) for v in xyxy)
                if conf >= THRESH.get(lb, 1) and all(box_iou(box, p[1]) < 0.5 for p in preds):
                    preds.append((lb, box))
        used = set()
        for g_lb, g_box in gts:
            best, best_iou = None, 0.4
            for i, (p_lb, p_box) in enumerate(preds):
                iou = box_iou(g_box, p_box)
                if i not in used and iou >= best_iou:
                    best, best_iou = i, iou
            if best is None:
                m[(g_lb, "missed")] += 1
            else:
                used.add(best)
                m[(g_lb, preds[best][0])] += 1
        false_alarm += sum(1 for i, (lb, _) in enumerate(preds) if i not in used and lb == "no_helmet")
    hel = sum(m[("helmet", p)] for p in (*CLASS_NAMES, "missed"))
    bare = sum(m[("no_helmet", p)] for p in (*CLASS_NAMES, "missed"))
    return {
        "matrix": m, "helmet_total": hel, "no_helmet_total": bare, "false_alarm": false_alarm,
        "helmet_as_bare": m[("helmet", "no_helmet")] / hel if hel else 0.0,
        "bare_recall": m[("no_helmet", "no_helmet")] / bare if bare else 0.0,
    }


def evaluate(data_yaml: str, weights_list: list[str]) -> list[dict]:
    from ultralytics import YOLO
    from models.yolo_types import DEVICE
    split, images = "test", _split_images(data_yaml, "test")
    if not images:
        split, images = "val", _split_images(data_yaml, "val")
        print("⚠️  Bộ test chưa có ảnh -> tạm đánh giá trên tập val (kém tin cậy hơn, nên gán nhãn thêm).")
    if not images:
        sys.exit(f"Không thấy ảnh test / val trong {data_yaml}")
    print(f"Bộ {split}: {len(images)} ảnh\n")
    results = []
    for wpath in weights_list:
        met = YOLO(wpath).val(data=data_yaml, split=split, imgsz=640, device=DEVICE, verbose=False, plots=False,
                              project=os.path.join(ROOT, "runs", "eval"), name="val", exist_ok=True)
        conf = head_confusion(wpath, images)
        results.append({"weights": wpath, "map50": float(met.box.map50), "map": float(met.box.map), **conf})

    print(f"{'Mô hình':<55} {'mAP50':>6} {'mAP50-95':>9} {'Mũ->KHÔNG MŨ':>13} {'Bắt đúng ko mũ':>15} {'Báo ko mũ thừa':>15}")
    for r in results:
        print(f"{os.path.relpath(r['weights'], ROOT)[:55]:<55} {r['map50']:>6.3f} {r['map']:>9.3f} "
              f"{r['helmet_as_bare']:>12.1%} {r['bare_recall']:>14.1%} {r['false_alarm']:>15}")
    r0 = results[0]
    print(f"\n(Bộ test có {r0['helmet_total']} đầu đội mũ, {r0['no_helmet_total']} đầu không mũ)")
    if len(results) >= 2:
        old, new = results[0], results[-1]
        better = (new["helmet_as_bare"] < old["helmet_as_bare"]
                  and new["bare_recall"] >= old["bare_recall"] - 0.02
                  and new["false_alarm"] <= old["false_alarm"])
        print("\n✅ NÊN dùng mô hình mới: ít phạt nhầm hơn, không bỏ sót thêm." if better else
              "\n❌ CHƯA nên thay: mô hình mới không tốt hơn rõ ràng (cần thêm dữ liệu / kiểm tra nhãn).")
    return results


def main():
    ap = argparse.ArgumentParser(description="So sánh các mô hình helmet trên bộ test")
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset", "helmet_merged", "data.yaml"))
    ap.add_argument("--weights", nargs="+", required=True, help="mô hình CŨ đặt trước, mô hình MỚI đặt sau")
    args = ap.parse_args()
    evaluate(args.data, args.weights)


if __name__ == "__main__":
    main()
