# reid_models/resnet18_reid.py
import torch
import numpy as np
from .base_reid import BaseReID
from torchvision import models, transforms
from PIL import Image

class ResNet18ReID(BaseReID):
    def __init__(self, weights_path=None, device='cpu'):
        self.device = torch.device(device)
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Identity()
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))


        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((256,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def extract_features(self, img):
        im = Image.fromarray(img[:,:,::-1])
        tensor = self.transform(im).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy().flatten()