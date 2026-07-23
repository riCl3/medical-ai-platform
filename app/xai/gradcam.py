from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torchvision import models, transforms

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "model.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def _load_model(device: torch.device) -> Tuple[nn.Module, nn.Module]:
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()
    target_layer = model.layer4[-1].conv2
    return model, target_layer


def run_gradcam(
    image_path: str,
    device: str = "cpu",
    target_class: int = None,
) -> Tuple[int, float, Image.Image]:
    device_obj = torch.device(device)
    model, target_layer = _load_model(device_obj)

    img_pil = Image.open(image_path).convert("RGB")
    input_tensor = _transform(img_pil).unsqueeze(0).to(device_obj)

    cam = GradCAM(model=model, target_layers=[target_layer])

    with torch.no_grad():
        logit = model(input_tensor).squeeze()
        prob = torch.sigmoid(logit).item()

    if target_class is None:
        target_class = int(prob > 0.5)

    targets = [BinaryClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    pred_label = CLASS_NAMES[target_class]
    confidence = prob if target_class == 1 else 1 - prob

    return pred_label, confidence, Image.fromarray(overlay)
