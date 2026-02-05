import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet18
from PIL import Image

st.title("Deprem Sonrası Bina Hasar Tespiti")

# =======================
# HASAR TESPİT MODELİ
# =======================
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("deprem_modeli.pth", map_location="cpu"))
model.eval()

classes = ["Hasarlı", "Sağlam"]

hasar_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =======================
# BİNA ALGILAMA MODELİ
# =======================
@st.cache_resource
def load_building_detector():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.eval()
    return m

building_model = load_building_detector()

imagenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def bina_var_mi(image):
    img = imagenet_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = building_model(img)
        probs = torch.softmax(outputs, dim=1)
        top5 = torch.topk(probs, 5)

    bina_kelimeleri = [
        "building", "house", "palace", "church",
        "mosque", "tower", "apartment", "castle"
    ]

    labels = models.ResNet18_Weights.DEFAULT.meta["categories"]

    for idx in top5.indices[0]:
        label = labels[idx]
        for kelime in bina_kelimeleri:
            if kelime in label.lower():
                return True

    return False

# =======================
# ARAYÜZ
# =======================
uploaded_file = st.file_uploader(
    "Bir bina fotoğrafı yükleyin",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Fotoğraf", use_container_width=True)

    if not bina_var_mi(image):
        st.error("❌ Bu fotoğrafta bina tespit edilemedi. Lütfen bina fotoğrafı yükleyin.")
        st.stop()

    img = hasar_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item() * 100

    st.success(
        f"🏢 Tahmin Sonucu: **{classes[pred]}** (%{confidence:.1f} güven)"
    )

    st.info(
        "⚠️ Bu sistem kesin hasar tespiti yapmaz. "
        "Deprem sonrası hızlı risk ön değerlendirmesi amacıyla geliştirilmiştir."
    )
