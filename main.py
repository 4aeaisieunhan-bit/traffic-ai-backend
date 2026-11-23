# main.py — chạy phân tích 1 video từ dòng lệnh: ghi output.mp4 + in bảng tổng kết.
# Xử lý từng khung hình dùng CHUNG pipeline với giao diện web (utils/pipeline.py).
#   Cách chạy:  python main.py "video 18s.mp4"                 (ghi output.mp4)
#               python main.py "video 18s.mp4" ket_qua.mp4     (chọn tên file kết quả)
import sys
import time
from collections import defaultdict

import cv2

from utils.labels import VIOLATION_TYPES, classify_reason
from utils.pipeline import TrafficPipeline


def violation_label(v: dict) -> str:
    """Tên loại vi phạm (giống bảng violation_types) từ 1 mục sidebar.violations_history.
    Dùng utils/labels.py (không import database.*) để main.py chạy độc lập, không kết nối SQL Server."""
    code = "NO_HELMET" if v.get("kind") == "NO_HELMET" else classify_reason(v.get("reason"))
    return VIOLATION_TYPES[code][0]


# ─────────────────────────────────────────────────────────────────────────────
# HÀM IN BÁO CÁO TỔNG KẾT SAU KHI XỬ LÝ VIDEO
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(
    video_path: str,
    output_path: str,
    total_frames: int,
    fps: float,
    elapsed_sec: float,
    violation_tracker,
    sidebar,
    model_stats: dict,
    helmet_detector=None,
    scanned_vehicles: int = 0,
):
    """In bảng tóm tắt kết quả xử lý video và thống kê từng mô hình AI.

    Args:
        video_path:       Đường dẫn video đầu vào.
        output_path:      Đường dẫn video đầu ra.
        total_frames:     Tổng số frame đã xử lý.
        fps:              Tốc độ khung hình của video.
        elapsed_sec:      Thời gian xử lý thực tế (giây).
        violation_tracker: Đối tượng TrafficViolationTracker.
        sidebar:          Đối tượng ViolationSidebar.
        model_stats:      Dict chứa stats từng model: {tên_model: {"weight": str,
                          "detections": int, "conf_sum": float, "conf_count": int}}
        scanned_vehicles: Số xe khác nhau đã theo dõi (số ID ByteTrack duy nhất).
    """
    duration_sec = total_frames / fps if fps > 0 else 0
    sep = "═" * 70

    print(f"\n{sep}")
    print("  🚦  BÁO CÁO TỔNG KẾT XỬ LÝ VIDEO TRAFFIC AI")
    print(sep)

    # ── 1. Thông tin video ────────────────────────────────────────────────────
    print("\n📹  THÔNG TIN VIDEO")
    print(f"   Đầu vào   : {video_path}")
    print(f"   Đầu ra    : {output_path}")
    print(f"   Tổng frame : {total_frames:,}")
    print(f"   FPS        : {fps:.2f}")
    print(f"   Thời lượng : {duration_sec:.1f}s")
    speed = f"{total_frames / elapsed_sec:.1f} FPS" if elapsed_sec > 0 else "N/A"
    print(f"   Thời gian xử lý: {elapsed_sec:.1f}s  (tốc độ: {speed})")

    # ── 2. Kết quả phát hiện vi phạm ─────────────────────────────────────────
    violations = getattr(sidebar, "violations_history", [])
    total_violations = len(violations)

    # Đếm theo loại vi phạm: mỗi vi phạm trong sidebar có "kind" (RED_LIGHT / NO_HELMET)
    # và "reason" (VD: "Vi pham: Qua stopline va dung yen ..." -> lỗi dừng sai vạch)
    _vtype = violation_label

    vtype_counts: dict[str, int] = defaultdict(int)
    for v in violations:
        vtype_counts[_vtype(v)] += 1

    legitimate_turns = getattr(violation_tracker, "legitimate_turn_count", 0)
    red_violations   = getattr(violation_tracker, "violation_count", 0)

    print("\n🚨  KẾT QUẢ PHÁT HIỆN VI PHẠM")
    print(f"   Phương tiện đã quét  : {scanned_vehicles}")
    print(f"   Rẽ hợp lệ            : {legitimate_turns}")
    print(f"   Vi phạm vượt đèn đỏ  : {red_violations}")
    if helmet_detector is not None:
        print(f"   Không đội mũ bảo hiểm: {helmet_detector.violation_count}")
    print(f"   Tổng vi phạm (sidebar): {total_violations}")

    if vtype_counts:
        print("\n   📋  Phân loại vi phạm:")
        for vtype, count in sorted(vtype_counts.items(), key=lambda x: -x[1]):
            print(f"       • {vtype:<35}: {count}")

    # Danh sách biển số + timestamp vi phạm
    if violations:
        print("\n   🔖  Danh sách xe vi phạm:")
        print(f"   {'STT':<5} {'Biển số':<14} {'Loại xe':<10} {'Loại vi phạm':<34} {'Mốc video'}")
        print("   " + "─" * 85)
        # violations_history: mới nhất ở đầu -> in theo thứ tự thời gian
        for idx, v in enumerate(reversed(violations), 1):
            print(f"   {idx:<5} {v.get('plate_text', 'N/A'):<14} {str(v.get('cls_name', 'N/A'))[:9]:<10} "
                  f"{_vtype(v):<34} {v.get('time_str', 'N/A')}")

    # ── 3. Thống kê từng mô hình AI ──────────────────────────────────────────
    print(f"\n🤖  THỐNG KÊ MÔ HÌNH AI")
    print(f"   {'Mô hình':<22} {'File weights':<35} {'Detections':>12} {'Avg Conf':>10}")
    print("   " + "─" * 82)
    for name, stats in model_stats.items():
        det   = stats["detections"]
        avg_c = (stats["conf_sum"] / stats["conf_count"]) if stats["conf_count"] > 0 else 0.0
        print(f"   {name:<22} {stats['weight']:<35} {det:>12,} {avg_c:>9.3f}")

    print(f"\n{sep}\n")

def run(video_path: str, output_path: str = "output.mp4") -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video {video_path}")
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = raw_fps if 0 < raw_fps <= 120 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pipeline = TrafficPipeline(fps)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, pipeline.output_size)

    frame_idx = 0
    start_time = time.time()
    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                break
            frame_idx += 1
            result = pipeline.process(raw, frame_idx)
            out.write(result.canvas)
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                print(
                    f"Xử lý frame {frame_idx}/{total_frames} | Vi phạm vượt đèn: {pipeline.tracker.violation_count}"
                    f" | Không đội mũ: {pipeline.helmet_detector.violation_count}"
                    f" | Rẽ hợp lệ: {pipeline.tracker.legitimate_turn_count}",
                    flush=True,
                )
    finally:
        cap.release()
        out.release()

    # Thống kê mô hình theo định dạng print_summary
    model_stats = {
        key: {"weight": st["weight"], "detections": st["detections"],
              "conf_sum": pipeline.stats[key]["conf_sum"], "conf_count": st["detections"]}
        for key, st in pipeline.model_stats().items()
    }
    print_summary(
        video_path=video_path,
        output_path=output_path,
        total_frames=frame_idx,
        fps=fps,
        elapsed_sec=time.time() - start_time,
        violation_tracker=pipeline.tracker,
        sidebar=pipeline.sidebar,
        model_stats=model_stats,
        helmet_detector=pipeline.helmet_detector,
        scanned_vehicles=len(pipeline.seen_track_ids),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Cách chạy: python main.py "<đường dẫn video>" [video_ket_qua.mp4]')
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output.mp4")
