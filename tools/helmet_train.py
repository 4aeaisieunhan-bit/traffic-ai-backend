# tools/helmet_train.py
# TRAIN TIẾP mô hình helmet từ weights/helmet_26s.pt với dữ liệu đã gán nhãn, rồi tự so sánh cũ / mới.
#
#   python tools/helmet_train.py                                   (dùng dữ liệu duyệt trên web + dữ liệu đã sửa)
#   python tools/helmet_train.py --base-data D:/datasets/Mu_Bao_Hiem_QA-1/data.yaml   (NÊN: trộn dữ liệu gốc)
#
# Nguồn dữ liệu (định dạng YOLO: <thư mục>/images/*.jpg + <thư mục>/labels/*.txt):
#   dataset/helmet_feedback   tự sinh khi cán bộ xác nhận / từ chối "Có đội mũ" vi phạm mũ trên web
#   dataset/helmet_reviewed   ảnh từ tools/helmet_mine.py SAU KHI bạn đã sửa nhãn
#   --base-data               bộ dữ liệu gốc đã dùng train helmet_26s.pt (trộn vào để mô hình không "quên"
#                             những gì đã học — chỉ train trên vài trăm ảnh mới dễ làm mô hình kém đi ở ca khác)
#
# Chia train / val / test theo NHÓM (cùng 1 video / 1 lần phân tích ở cùng 1 phần) để bộ test không "nhìn trộm"
# frame gần giống trong train. Bộ test cố định giữa các lần chạy => so sánh các mô hình công bằng.
# Mô hình mới ở runs/detect/<name>/weights/best.pt — KHÔNG tự thay mô hình đang chạy. Nếu kết quả tốt:
#   copy best.pt vào weights/ và đặt HELMET_WEIGHTS=weights/<tên file>.pt trong .env rồi khởi động lại backend.
import argparse
import glob
import hashlib
import os
import re
import shutil
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.helmet_dataset import CLASS_NAMES   # noqa: E402

DEFAULT_SOURCES = [os.path.join(ROOT, "dataset", "helmet_feedback"), os.path.join(ROOT, "dataset", "helmet_reviewed")]


def _group(name: str) -> str:
    """Nhóm chia tập: 'video_f000450' -> video + đoạn ~3 giây; '<task>_<track>' (web) -> task."""
    m = re.match(r"(.+)_f(\d+)$", name)
    if m:
        return f"{m.group(1)}#{int(m.group(2)) // 90}"
    return name.split("_")[0]


def _split(group: str, test_frac: float, val_frac: float) -> str:
    x = int(hashlib.md5(group.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if x < test_frac else "val" if x < test_frac + val_frac else "train"


def build_dataset(sources: list[str], out: str, base_data: str | None, test_frac: float, val_frac: float) -> str:
    if os.path.isdir(out):
        shutil.rmtree(out)
    counts = {"train": 0, "val": 0, "test": 0}
    for src in sources:
        for img in sorted(glob.glob(os.path.join(src, "images", "*.jpg")) + glob.glob(os.path.join(src, "images", "*.png"))):
            name = os.path.splitext(os.path.basename(img))[0]
            lbl = os.path.join(src, "labels", name + ".txt")
            if not os.path.isfile(lbl):
                continue                      # ảnh chưa có nhãn -> bỏ qua
            part = _split(_group(name), test_frac, val_frac)
            tag = os.path.basename(os.path.normpath(src))
            for kind, path, ext in (("images", img, os.path.splitext(img)[1]), ("labels", lbl, ".txt")):
                dst_dir = os.path.join(out, part, kind)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(path, os.path.join(dst_dir, f"{tag}__{name}{ext}"))
            counts[part] += 1
        print(f"  nguồn {src}: đã gộp")
    total = sum(counts.values())
    if total == 0:
        sys.exit("Chưa có ảnh nào đã gán nhãn. Duyệt vi phạm mũ trên web hoặc chạy tools/helmet_mine.py rồi sửa nhãn trước.")
    for part in counts:
        os.makedirs(os.path.join(out, part, "images"), exist_ok=True)

    cfg = {"path": out, "train": ["train/images"], "val": ["val/images"], "test": "test/images",
           "names": {i: n for i, n in enumerate(CLASS_NAMES)}}
    if base_data:
        from tools.helmet_eval import load_yaml
        base = load_yaml(base_data)
        base_root = base.get("path") or os.path.dirname(os.path.abspath(base_data))
        names = base.get("names")
        names = list(names.values()) if isinstance(names, dict) else names
        if names != CLASS_NAMES:
            sys.exit(f"Lớp của bộ dữ liệu gốc {names} khác {CLASS_NAMES} — cần cùng thứ tự lớp.")
        for key in ("train", "val"):
            entries = base.get(key) or []
            for e in entries if isinstance(entries, list) else [entries]:
                cfg[key].append(e if os.path.isabs(e) else os.path.normpath(os.path.join(base_root, e)))
        print(f"  + bộ dữ liệu gốc: {base_data}")
    path = os.path.join(out, "data.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"Ảnh mới: train {counts['train']} · val {counts['val']} · test {counts['test']}  ->  {path}")
    if counts["test"] < 20:
        print("⚠️  Bộ test còn rất ít ảnh -> kết quả so sánh chưa đáng tin, nên gán nhãn thêm.")
    if counts["val"] == 0 and not base_data:
        cfg["val"] = cfg["train"]   # quá ít dữ liệu: tạm dùng train làm val để Ultralytics chạy được
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def main():
    ap = argparse.ArgumentParser(description="Train tiếp mô hình helmet với dữ liệu đã gán nhãn")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--base-data", default=None, help="data.yaml của bộ dữ liệu gốc (khuyên dùng)")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "helmet_26s.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset", "helmet_merged"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="GPU 6GB: 8 là an toàn")
    ap.add_argument("--freeze", type=int, default=10, help="giữ nguyên N lớp đầu (backbone) để không quên kiến thức cũ")
    ap.add_argument("--lr0", type=float, default=0.002, help="learning rate nhỏ: tinh chỉnh, không học lại từ đầu")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--name", default="helmet_ft")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    data_yaml = build_dataset(args.sources, args.out, args.base_data, args.test_frac, args.val_frac)

    from ultralytics import YOLO
    from models.yolo_types import DEVICE
    model = YOLO(args.weights)
    model.train(
        data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=DEVICE,
        freeze=args.freeze, lr0=args.lr0, lrf=0.1, warmup_epochs=1, patience=15,
        hsv_v=0.5,          # thay đổi độ sáng mạnh hơn: mũ đen / tóc đen trong nhiều điều kiện ánh sáng
        fliplr=0.5, mosaic=0.5, close_mosaic=5,
        project=os.path.join(ROOT, "runs", "detect"), name=args.name, exist_ok=False, plots=True,
    )
    best = os.path.join(str(model.trainer.save_dir), "weights", "best.pt") if model.trainer else None
    print(f"\nMô hình mới: {best}")
    if best and os.path.isfile(best) and not args.no_eval:
        from tools.helmet_eval import evaluate
        print("\n══ So sánh mô hình cũ / mới trên bộ test ══")
        evaluate(data_yaml, [args.weights, best])


if __name__ == "__main__":
    main()
