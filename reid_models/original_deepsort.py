import torch
import numpy as np
from .base_reid import BaseReID
from torchvision import transforms
from PIL import Image

class OriginalReID(BaseReID):
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = checkpoint['model'].to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def extract_features(self, img):
        im = Image.fromarray(img[:,:,::-1])
        tensor = self.transform(im).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        return feat.cpu().numpy().flatten()