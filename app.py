import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch import nn
from PIL import Image

st.title("Deprem Sonrası Bina Hasar Tespiti")

# === MODEL YÜKLEME ===
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("deprem_modeli.pth", map_location="cpu"))
model.eval()

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["Hasarlı", "Sağlam"]

# === BİNA KONTROLÜ (BASİT AMA ETKİLİ) ===
def bina_mi(image):
    # Çok açık / çok karanlık / aşırı düz görüntüler elenir
    gray = image.convert("L")
    pixels = list(gray.getdata())
    std = torch.tensor(pixels, dtype=torch.float).std().item()
    return std > 15   # eşik (deneysel ama iş görür)

uploaded_file = st.file_uploader("Bir bina fotoğrafı yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Fotoğraf", use_container_width=True)

    if not bina_mi(image):
        st.error("❌ Bina tespit edilemedi. Lütfen bina fotoğrafı yükleyin.")
    else:
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, 1).item()

        st.success(f"🏢 Tahmin Sonucu: **{classes[pred]}**")
