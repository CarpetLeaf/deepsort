# detectors/ssd_detector.py
import torch
import torchvision
import numpy as np
from .base_detector import BaseDetector

from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

class SSDDetector(BaseDetector):
    def __init__(self, confidence_threshold=0.5, device='cpu'):
        self.device = torch.device(device)
        weights_path = "resources/models/ssdlite320.pth"
        weights = torch.load(weights_path, map_location=self.device)
        self.model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        self.model.load_state_dict(weights)
        self.model.to(self.device).eval()
        self.conf_threshold = confidence_threshold


    def detect(self, frame: np.ndarray):
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor()])
        img = transform(frame).to(self.device)
        outputs = self.model([img])[0]
        boxes = outputs['boxes'].cpu().detach().numpy()
        scores = outputs['scores'].cpu().detach().numpy()
        labels = outputs['labels'].cpu().detach().numpy()
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if label == 1 and score >= self.conf_threshold:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                detections.append((x1, y1, w, h, score))
        return detections