from ultralytics import YOLO
from .base_detector import BaseDetector
import numpy as np

class YOLOv5Detector(BaseDetector):
    def __init__(self, model_path='yolov5n.pt', confidence_threshold=0.25, device='cpu'):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.model.to(device)
        self.conf_threshold = confidence_threshold

    def detect(self, frame: np.ndarray):
        results = self.model(frame, conf=self.conf_threshold, classes=[0])
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                detections.append((x1, y1, w, h, conf))
        return detections