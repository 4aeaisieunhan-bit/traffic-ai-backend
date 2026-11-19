from ultralytics import YOLO
from ultralytics.engine.results import Results
from models.weights import LICENSE_PLATE_WEIGHTS
from models.yolo_types import INFER_KW, as_results 
model = YOLO(LICENSE_PLATE_WEIGHTS) 
def detect_license_plate(image) -> list[Results]:
    result = model.predict(
        source = image,
        conf = 0.25,
        verbose=False,
        **INFER_KW, 
    )
    return as_results(result)
