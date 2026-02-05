import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =====================
# SAYFA AYARLARI
# =====================
st.set_page_config(page_title="Deprem Hasar Tespiti", layout="centered")
st.title("🏚️ Deprem Sonrası Yapı Risk Analizi")
st.write("Bu sistem **kesin hasar tespiti yapmaz**, hızlı **risk ön değerlendirmesi** sunar.")

# =====================
# MODEL YÜKLEME
# =====================
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  
    model.load_state_dict(
        torch.load("deprem_modeli.pth", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================
# DOSYA YÜKLEME
# =====================
uploaded_file = st.file_uploader(
    "Bir bina / enkaz fotoğrafı yükleyin",
    type=["jpg", "jpeg", "png"]
)

# =====================
# TAHMİN
# =====================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görüntü", use_container_width=True)

    img = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)[0]
    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()


# 🔴 ENKAZ / KARARSIZLIK FİLTRESİ
if confidence < 0.75:
    pred = 1  # yüksek risk


if pred == 0:
    st.success(
        f"🟢 **Düşük Riskli Yapı**\n\n"
        f"Güven Skoru: **%{confidence*100:.1f}**"
    )
else:
    st.error(
        f"🔴 **Yüksek Riskli / Hasarlı Yapı**\n\n"
        f"Güven Skoru: **%{confidence*100:.1f}**"
    )

    # =====================
    # SONUÇ YORUMLAMA
    # =====================
    if pred == 0:
        st.success(
            f"🟢 **Düşük Riskli Yapı**\n\n"
            f"Güven Skoru: **%{confidence*100:.1f}**"
        )
    else:
        st.error(
            f"🔴 **Yüksek Riskli / Hasarlı Yapı**\n\n"
            f"Güven Skoru: **%{confidence*100:.1f}**"
        )

    st.info(
        "ℹ️ Bu sonuç, saha ekipleri için **önceliklendirme amacıyla** üretilmiştir. "
        "Kesin karar için mühendis incelemesi gereklidir."
    )
