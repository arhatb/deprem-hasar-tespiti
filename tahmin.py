import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch import nn
from PIL import Image

# Modeli yükle
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("deprem_modeli.pth"))
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
    pred = torch.argmax(output, dim=1)

if pred.item() == 0:
    print("Tahmin: HASARLI")
else:
    print("Tahmin: SAGLAM")
