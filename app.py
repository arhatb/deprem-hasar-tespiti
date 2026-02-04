import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch import nn
from PIL import Image

st.title("Deprem Sonrası Bina Hasar Tespiti")

# Model yükleme
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("deprem_modeli.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

uploaded_file = st.file_uploader("Bir bina fotoğrafı yükle", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Yüklenen Görüntü", use_container_width=True)


    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1)

    if pred.item() == 0:
        st.error("🔴 HASARLI BİNA")
    else:
        st.success("🟢 SAĞLAM BİNA")
