from ultralytics import YOLO
from ultralytics.engine.results import Results
from models.weights import HELMET_WEIGHTS   # đổi mô hình: HELMET_WEIGHTS=<file .pt> trong .env
from models.yolo_types import INFER_KW, as_results

model = YOLO(HELMET_WEIGHTS)


def detect_helmet(image) -> list[Results]:
    result = model.predict(
        source=image,
        conf=0.25,
        verbose=False,
        **INFER_KW,
    )
    return as_results(result)
