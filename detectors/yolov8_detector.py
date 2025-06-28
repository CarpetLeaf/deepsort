# detectors/yolov8_detector.py
from ultralytics import YOLO
from .base_detector import BaseDetector
import numpy as np

class YOLOv8Detector(BaseDetector):
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.3, device='cpu'):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.model.to(device)
        self.conf_threshold = confidence_threshold

    def detect(self, frame: np.ndarray):
        results = self.model.predict(source=frame, conf=self.conf_threshold, classes=[0], verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                detections.append((x1, y1, w, h, conf))
        return detections
