import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch import nn
from PIL import Image

check_model = mobilenet_v2(pretrained=True).eval()

def is_building(image_path):
    img = Image.open(image_path).convert('RGB')
    # ResNet/MobileNet için standart transform
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        output = check_model(img_t)
    _, index = torch.max(output, 1)
    
    # ImageNet bina sınıf aralıkları (Basit kontrol)
    return 400 <= index.item() <= 900

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
