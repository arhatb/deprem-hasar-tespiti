import torch
from torchvision import models, transforms
from PIL import Image

def is_building(image_path):
    # Hafif ve hızlı bir model (MobileNet)
    model = models.mobilenet_v2(pretrained=True).eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
    
    # ImageNet üzerinde bina ile ilgili sınıfların indexleri (Örn: 449: boathouse, 482: castle vb.)
    # Basitçe en yüksek skorlu sınıfın 'bina' olup olmadığına bakılır
    _, index = torch.max(output, 1)
    
    # Bina ile ilgili genel ImageNet index aralığı (Örn: 400-900 arası mimari ağırlıklıdır)
    if index.item() < 400: 
        return False
    return True
