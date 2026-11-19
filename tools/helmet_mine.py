# tools/helmet_mine.py
# KHAI THÁC DỮ LIỆU để train lại mô hình helmet: quét video, giữ lại các khung hình có đầu "đáng ngờ"
# (mô hình báo no_helmet, hoặc phân vân giữa helmet / no_helmet — hay gặp với mũ đen), kèm NHÃN SẴN của mô hình.
# Bạn chỉ cần SỬA nhãn sai (chủ yếu: mũ đen bị gán no_helmet -> đổi thành helmet) thay vì vẽ từ đầu.
#
#   python tools/helmet_mine.py "video 18s.mp4" 21s_vidnew.mov
#   python tools/helmet_mine.py videos/*.mp4 --every 0.5 --max-images 400
#
# Kết quả (định dạng YOLO, mở được bằng X-AnyLabeling / CVAT "Ultralytics YOLO" / Label Studio):
#   dataset/helmet_mining/images/*.jpg      khung hình 1280x720 đã tăng nét (đúng ảnh mô hình nhìn thấy khi chạy)
#   dataset/helmet_mining/labels/*.txt      nhãn sẵn: <lớp> <cx> <cy> <w> <h> (0 = helmet, 1 = no_helmet)
#   dataset/helmet_mining/classes.txt
#   dataset/helmet_mining/review/*.jpg      ảnh phóng to từng đầu đáng ngờ để xem nhanh
# Sau khi sửa xong: lưu ảnh + nhãn YOLO đã sửa vào dataset/helmet_reviewed/{images,labels}/ (tools/helmet_train.py
# chỉ dùng thư mục đã duyệt, KHÔNG dùng nhãn sẵn chưa sửa).
import argparse
import glob
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.frame_enhance import enhance_frame          # noqa: E402
from tools.helmet_eval import CLASS_NAMES, box_iou         # noqa: E402

PROCESS_SIZE = (1280, 720)


def main():
    ap = argparse.ArgumentParser(description="Khai thác khung hình có đầu đáng ngờ để gán nhãn lại mô hình helmet")
    ap.add_argument("videos", nargs="+", help="đường dẫn video (hỗ trợ ký tự đại diện *.mp4)")
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset", "helmet_mining"))
    ap.add_argument("--weights", default=os.getenv("HELMET_WEIGHTS", os.path.join(ROOT, "weights", "helmet_26s.pt")))
    ap.add_argument("--every", type=float, default=0.5, help="lấy 1 khung hình mỗi N giây (mặc định 0.5)")
    ap.add_argument("--min-head", type=int, default=12, help="bỏ qua đầu thấp hơn N px (quá nhỏ để gán nhãn chắc)")
    ap.add_argument("--label-conf", type=float, default=0.25, help="ngưỡng tin cậy của nhãn sẵn")
    ap.add_argument("--max-images", type=int, default=0, help="giới hạn số ảnh (0 = không giới hạn)")
    args = ap.parse_args()

    from ultralytics import YOLO
    from models.yolo_types import INFER_KW, as_results
    model = YOLO(args.weights)

    videos = [p for pattern in args.videos for p in (glob.glob(pattern) or [pattern])]
    for sub in ("images", "labels", "review"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")

    saved = heads_total = 0
    W, H = PROCESS_SIZE
    for video in videos:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"⚠️  Không mở được {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps * args.every)))
        stem = os.path.splitext(os.path.basename(video))[0].replace(" ", "_")
        idx = kept = 0
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            frame = raw if (raw.shape[1], raw.shape[0]) == (W, H) else cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA)
            frame = enhance_frame(frame, sharpen_amount=0.30, contrast_clip=1.4)
            r = as_results(model.predict(frame, conf=0.2, verbose=False, **INFER_KW))[0]
            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                    dets.append(((x1, y1, x2, y2), r.names[int(b.cls[0])], float(b.conf[0])))

            # Đầu đáng ngờ: mọi no_helmet đủ lớn (mũ đen hay bị báo no_helmet với độ tin cậy cao)
            # + đầu mà mô hình gắn cả 2 nhãn (phân vân)
            suspicious = []
            for box, lb, conf in dets:
                if box[3] - box[1] < args.min_head:
                    continue
                if lb == "no_helmet" or any(
                    lb2 != lb and box_iou(box, b2) >= 0.35 for b2, lb2, _ in dets
                ):
                    suspicious.append((box, lb, conf))
            if not suspicious:
                continue

            # Nhãn sẵn: mỗi đầu 1 nhãn (giữ nhãn tin cậy hơn)
            labels = []
            for box, lb, conf in sorted(dets, key=lambda d: -d[2]):
                if conf < args.label_conf or any(box_iou(box, o[0]) >= 0.5 for o in labels):
                    continue
                labels.append((box, lb, conf))

            name = f"{stem}_f{idx:06d}"
            cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(os.path.join(args.out, "labels", name + ".txt"), "w", encoding="utf-8") as f:
                for (x1, y1, x2, y2), lb, _ in labels:
                    f.write(f"{CLASS_NAMES.index(lb)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
            for k, ((x1, y1, x2, y2), lb, conf) in enumerate(suspicious):
                pad = max(x2 - x1, y2 - y1)
                crop = frame[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
                if crop.size:
                    crop = cv2.resize(crop, None, fx=160 / crop.shape[0], fy=160 / crop.shape[0], interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(args.out, "review", f"{name}_{k}_{lb}_{conf:.2f}.jpg"), crop)
            kept += 1
            saved += 1
            heads_total += len(suspicious)
            if args.max_images and saved >= args.max_images:
                break
        cap.release()
        print(f"✔ {video}: {kept} khung hình có đầu đáng ngờ")
        if args.max_images and saved >= args.max_images:
            break

    print(f"\nĐã lưu {saved} ảnh ({heads_total} đầu đáng ngờ) vào {args.out}")
    print("Bước tiếp: mở thư mục bằng công cụ gán nhãn, sửa nhãn sai, lưu bản đã sửa vào dataset/helmet_reviewed/")


if __name__ == "__main__":
    main()
