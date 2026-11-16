# models/yolo_types.py
# 1) Chọn thiết bị chạy AI dùng chung cho mọi mô hình YOLO (GPU NVIDIA nếu có, không thì CPU).
# 2) Ultralytics khai báo kiểu trả về của predict()/track() rất rộng (list | Iterator, Results | Tensor),
#    khiến trình soạn thảo báo lỗi khi dùng result[0].boxes. as_results() chỉ "nói rõ kiểu" cho
#    trình kiểm tra kiểu, KHÔNG thay đổi dữ liệu (predict với stream=False luôn trả list[Results]).
import os
from typing import Any, cast

import numpy as np
import torch
from ultralytics.engine.results import Results


def _select_device() -> tuple[str | int, bool]:
    """
    AI_DEVICE trong .env / biến môi trường: 'auto' (mặc định) | 'cpu' | 'cuda' | '0', '1'...
    GPU: dùng FP16 (half) -> nhanh hơn ~2 lần, độ chính xác gần như không đổi.
    """
    wanted = os.getenv("AI_DEVICE", "auto").strip().lower()
    if wanted == "cpu" or not torch.cuda.is_available():
        if wanted not in ("auto", "cpu"):
            print(f"⚠️  AI_DEVICE={wanted} nhưng không có GPU CUDA khả dụng -> dùng CPU", flush=True)
        return "cpu", False
    index = 0 if wanted in ("auto", "cuda") else int(wanted.replace("cuda:", ""))
    return index, True


DEVICE, HALF = _select_device()
# Truyền vào mọi lệnh model.predict(...) / model.track(...).
# Ultralytics 8.4 thay tham số 'half' (đã lỗi thời) bằng 'quantize': 16 = FP16, None = FP32.
INFER_KW: dict[str, Any] = {"device": DEVICE, "quantize": 16 if HALF else None}

if DEVICE == "cpu":
    DEVICE_NAME = "CPU"
    print("ℹ️  AI chạy trên CPU (chậm). Có GPU NVIDIA thì cài PyTorch bản CUDA để tăng tốc.", flush=True)
else:
    DEVICE_NAME = f"GPU {torch.cuda.get_device_name(DEVICE)}"
    print(f"🚀 AI chạy trên {DEVICE_NAME} (FP16)", flush=True)


def as_results(output: Any) -> list[Results]:
    """Kết quả predict()/track() (stream=False) -> list[Results]."""
    return cast(list[Results], list(output))


def to_numpy(value: Any) -> np.ndarray:
    """Tensor (CPU/GPU) hoặc ndarray -> ndarray."""
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
