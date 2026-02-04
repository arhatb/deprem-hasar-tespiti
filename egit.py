import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch import nn, optim

print("Eğitim başladı")

# Data augmentation + normalizasyon
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder("data", transform=transform)

print("Toplam veri sayısı:", len(dataset))
print("Sınıflar:", dataset.classes)

loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 10

for epoch in range(epochs):
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} tamamlandı, loss: {loss.item():.4f}")

torch.save(model.state_dict(), "deprem_modeli.pth")
print("Model kaydedildi")
