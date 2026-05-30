"""
utils.py — Ortak Yardımcı Fonksiyonlar
========================================
- Model yükleme (checkpoint destekli)
- Görüntü ön işleme (ImageNet normalize dahil)
- Bina varlık kontrolü (MobileNetV3 ile)
- Grad-CAM ısı haritası üretimi
- EXIF'ten GPS koordinatı okuma
"""

import io
import struct
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
)

# ── Sabitler ──────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

HASAR_SEVIYELERI = {
    "yikilmis":    {"tr": "Yıkılmış",     "renk": "#c0392b", "seviye": 5},
    "agir_hasarli":{"tr": "Ağır Hasarlı", "renk": "#e74c3c", "seviye": 4},
    "orta_hasarli":{"tr": "Orta Hasarlı", "renk": "#e67e22", "seviye": 3},
    "hafif_hasarli":{"tr": "Hafif Hasarlı","renk": "#f39c12", "seviye": 2},
    "hasarsiz":    {"tr": "Hasarsız",      "renk": "#27ae60", "seviye": 1},
}

# ── Transform ─────────────────────────────────────────────────────────────────

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """PIL Image → (1, 3, 224, 224) tensor"""
    tf  = get_inference_transform()
    img = pil_image.convert("RGB")
    return tf(img).unsqueeze(0)

# ── Model yükleme ─────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device = None):
    """
    Checkpoint'ten model yükler.
    Eski format (sadece state_dict) ve yeni format (dict) desteklenir.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    # Yeni format: dict with 'model_state' and 'classes'
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        classes     = checkpoint["classes"]
        state_dict  = checkpoint["model_state"]
        num_classes = len(classes)
    else:
        # Eski format: direkt state_dict, 2 sınıf varsayımı
        state_dict  = checkpoint
        classes     = ["yikilmis_veya_hasarli", "hasarsiz"]
        num_classes = 2

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device), classes, device

# ── Tahmin ────────────────────────────────────────────────────────────────────

def predict(model, tensor: torch.Tensor, classes: list, device: torch.device):
    """
    Döndürür: (sinif_adi, guvenskor_0_1, {sinif: olasilik} dict)
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx    = probs.argmax().item()
    pred_class  = classes[pred_idx]
    confidence  = probs[pred_idx].item()
    prob_dict   = {cls: round(probs[i].item(), 4) for i, cls in enumerate(classes)}

    return pred_class, confidence, prob_dict

# ── Bina varlık kontrolü ──────────────────────────────────────────────────────

_bina_model = None  # lazy load

# ImageNet sınıfları — yapı/bina ile ilişkili indeksler
BINA_IMAGENET_IDXLER = set([
    449,  # boathouse
    483,  # castle
    663,  # mobile home
    668,  # mosque
    782,  # stupa
    600,  # jail
    684,  # palace
    670,  # muzzle  (kale duvarı gibi görünüm)
    698,  # palace
    747,  # prison
    762,  # school
    763,  # screen
    777,  # steel arch bridge  (mimari)
    779,  # suspension bridge
    833,  # theater curtain
    840,  # tile roof
    857,  # triumphal arch
    873,  # wall clock (duvar)
    910,  # window screen
    912,  # wooden spoon  — kaldır gerekirse
]) | set(range(440, 530))  # birçok yapı ImageNet 440-530 arasında


def is_building(pil_image: Image.Image, threshold: float = 0.10) -> bool:
    """
    MobileNetV3-Small kullanarak görüntünün bina/enkaz içerip içermediğini kontrol eder.
    threshold: bina sınıflarının toplam olasılığı bu değerin üzerindeyse True döner.
    """
    global _bina_model
    if _bina_model is None:
        _bina_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = tf(pil_image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(_bina_model(tensor), dim=1)[0]

    bina_skoru = sum(probs[i].item() for i in BINA_IMAGENET_IDXLER if i < len(probs))
    return bina_skoru >= threshold

# ── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """
    ResNet18'in son conv katmanından (layer4) Grad-CAM üretir.
    Kullanım:
        gcam   = GradCAM(model)
        heatmap = gcam(tensor, target_class)  # numpy (H, W) 0-1 arası
        gcam.remove_hooks()
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []
        self._register()

    def _register(self):
        target_layer = self.model.layer4[-1]

        self._hooks.append(
            target_layer.register_forward_hook(
                lambda m, inp, out: setattr(self, "activations", out.detach())
            )
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(
                lambda m, grad_in, grad_out: setattr(self, "gradients", grad_out[0].detach())
            )
        )

    def __call__(self, tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.zero_grad()
        tensor = tensor.requires_grad_(True)
        logits = self.model(tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        logits[0, target_class].backward()

        weights      = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam          = (weights * self.activations).sum(dim=1)[0]       # (H, W)
        cam          = torch.clamp(cam, min=0)
        cam          = cam.cpu().numpy()
        cam         -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam  # (7, 7) — gerekirse PIL ile 224×224'e büyütülür

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def gradcam_overlay(model, pil_image: Image.Image, target_class: int = None) -> Image.Image:
    """
    Orijinal görüntü üzerine Grad-CAM ısı haritası bindirir.
    Döndürür: RGBA PIL Image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    tf     = get_inference_transform()
    tensor = tf(pil_image.convert("RGB")).unsqueeze(0)

    gcam    = GradCAM(model)
    heatmap = gcam(tensor, target_class)
    gcam.remove_hooks()

    # 224×224'e büyüt
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(
        pil_image.size, Image.BILINEAR
    )

    # Renk haritası uygula (Jet)
    colored = np.array(cm.jet(np.array(heatmap_img) / 255.0))[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    overlay = Image.blend(
        pil_image.convert("RGB"),
        Image.fromarray(colored),
        alpha=0.45,
    )
    return overlay

# ── GPS / EXIF ────────────────────────────────────────────────────────────────

def get_gps_from_exif(pil_image: Image.Image):
    """
    Görüntü EXIF verisinden GPS koordinatlarını okur.
    Döndürür: (lat, lon) tuple veya (None, None)
    """
    try:
        exif = pil_image._getexif()
        if exif is None:
            return None, None

        GPS_TAG = 34853  # GPSInfo tag ID
        gps_info = exif.get(GPS_TAG)
        if gps_info is None:
            return None, None

        def to_decimal(vals, ref):
            d = float(vals[0])
            m = float(vals[1])
            s = float(vals[2])
            dec = d + m / 60 + s / 3600
            if ref in ("S", "W"):
                dec = -dec
            return dec

        lat = to_decimal(gps_info.get(2, (0, 0, 0)), gps_info.get(1, "N"))
        lon = to_decimal(gps_info.get(4, (0, 0, 0)), gps_info.get(3, "E"))
        return round(lat, 6), round(lon, 6)

    except Exception:
        return None, None
