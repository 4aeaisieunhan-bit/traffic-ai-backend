from ultralytics import YOLO
from ultralytics.engine.results import Results
from models.yolo_types import INFER_KW, as_results 
model = YOLO("weights/vehicle-26s-best.pt")
def detect_vehicle(image, track=True) -> list[Results]:
    if track:
        result = model.track(
            source=image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False,
            **INFER_KW,
        )
    else:
        result = model.predict(
            source=image,
            conf=0.25,
            verbose=False,
            **INFER_KW,
        )
    return as_results(result)


def track_vehicle(image) -> list[Results]:
    return detect_vehicle(image, track=True)
