import os

from ultralytics import YOLO
from ultralytics.engine.results import Results
from models.yolo_types import INFER_KW, as_results

# Đổi mô hình mũ bảo hiểm: đặt HELMET_WEIGHTS=<đường dẫn file .pt> trong .env
HELMET_WEIGHTS = os.getenv("HELMET_WEIGHTS", "weights_demo/mubaohiem.pt")
model = YOLO(HELMET_WEIGHTS)


def detect_helmet(image) -> list[Results]:
    result = model.predict(
        source=image,
        conf=0.25,
        verbose=False,
        **INFER_KW,
    )
    return as_results(result)
