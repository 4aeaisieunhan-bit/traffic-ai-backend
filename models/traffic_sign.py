from ultralytics import YOLO
from ultralytics.engine.results import Results
from models.yolo_types import INFER_KW, as_results 
model = YOLO("weights/OCR-26s-best.pt")

def detect_traffic_sign(image) -> list[Results]:
    result = model.predict(
        source = image,
        conf = 0.25,
        verbose=False,
        **INFER_KW,

    )
    return as_results(result)
