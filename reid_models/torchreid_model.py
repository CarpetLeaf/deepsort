import torch
import numpy as np
from .base_reid import BaseReID
import torchreid
from torchvision import transforms
from PIL import Image

class TorchReID(BaseReID):
    def __init__(self, model_name='osnet_x0_25', device='cpu'):
        self.device = torch.device(device)
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        )
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