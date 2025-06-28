from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray):
        """
        Detect objects in the frame.
        Returns list of tuples: (x, y, w, h, confidence).
        """
        pass