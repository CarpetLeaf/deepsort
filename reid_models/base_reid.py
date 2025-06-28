# reid_models/base_reid.py
from abc import ABC, abstractmethod
import numpy as np

class BaseReID(ABC):
    @abstractmethod
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """
        Given a cropped person image, return a 1D feature vector.
        """
        pass