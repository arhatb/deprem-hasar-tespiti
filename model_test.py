import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

# Modeli yükle
model = resnet18(pretrained=True)
model.eval()

# Görüntü hazırlama
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open("bina.jpg")
img = transform(img)
img = img.unsqueeze(0)

# Tahmin
with torch.no_grad():
    output = model(img)

print("Model çıktısı boyutu:", output.shape)
