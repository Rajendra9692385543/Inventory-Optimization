# embeddings/embedding_model.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Union, List

class ResNetEmbedder:
    """
    ResNet50-based embedder. Returns L2-normalized float32 numpy vectors of size 2048.
    """
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(pretrained=True)
        modules = list(model.children())[:-1]  # remove final fc
        self.model = nn.Sequential(*modules).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _image_to_tensor(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)  # (1,3,224,224)

    @torch.no_grad()
    def embed(self, image: Union[Image.Image, str]) -> np.ndarray:
        x = self._image_to_tensor(image)
        feat = self.model(x)  # (1, 2048, 1, 1)
        feat = feat.reshape(feat.size(0), -1)  # (1, 2048)
        arr = feat.cpu().numpy().astype('float32')[0]
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    def embed_batch(self, images: List[Union[Image.Image, str]]) -> np.ndarray:
        tensors = [self._image_to_tensor(img) for img in images]
        batch = torch.cat(tensors, dim=0)
        with torch.no_grad():
            feats = self.model(batch).reshape(batch.size(0), -1).cpu().numpy().astype('float32')
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
        return feats
