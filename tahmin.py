import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, mobilenet_v2 # mobilenet_v2 buraya eklendi
from torch import nn
from PIL import Image

# 1. Bina Kontrol Modeli
check_model = mobilenet_v2(pretrained=True).eval()

def is_building(image_path):
    img = Image.open(image_path).convert('RGB')
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
    return 400 <= index.item() <= 900

# 2. Hasar Tespit Modeli
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("deprem_modeli.pth"))
model.eval()

# 3. Çalıştırma Bölümü
image_path = "bina.jpg" 

if not is_building(image_path):
    print("Hata: Görüntüde bina tespit edilemedi. Lütfen geçerli bir fotoğraf yükleyin.")
else:
    # --- DİKKAT: Bu kısımlar 'else' bloğunun içinde (sağda) olmalı ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert('RGB') # Sabit isim yerine image_path kullanıldı
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1)

    if pred.item() == 0:
        print("Tahmin: HASARLI")
    else:
        print("Tahmin: SAGLAM")
