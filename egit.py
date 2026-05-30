"""
egit.py — Deprem Hasar Tespiti Model Eğitimi
=============================================
Düzeltmeler:
  - ImageNet normalizasyonu eklendi (kritik hata giderildi)
  - Train / Validation ayrımı (%80 / %20)
  - Her epoch'ta Accuracy, Precision, Recall, F1 hesaplanıyor
  - En iyi val_f1 skoruna sahip model kaydediliyor (best checkpoint)
  - Sınıf dengesizliği WeightedRandomSampler ile gideriliyor
  - pretrained=True → weights=ResNet18_Weights.DEFAULT (deprecation uyarısı kaldırıldı)

Klasör yapısı:
  data/
    yikilmis/        ← yıkılmış / enkaz
    agir_hasarli/    ← ağır hasarlı
    orta_hasarli/    ← orta hasarlı
    hafif_hasarli/   ← hafif hasarlı
    hasarsiz/        ← hasarsız bina

Kullanım:
  python egit.py
  python egit.py --epochs 20 --lr 0.0003 --batch 16
"""

import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

# ── Sabitler ──────────────────────────────────────────────────────────────────

SINIFLAR = ["yikilmis", "agir_hasarli", "orta_hasarli", "hafif_hasarli", "hasarsiz"]
NUM_CLASSES = len(SINIFLAR)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Argümanlar ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="data",              help="Veri klasörü")
    p.add_argument("--model",  default="deprem_modeli.pth", help="Kaydedilecek model yolu")
    p.add_argument("--epochs", type=int,   default=15)
    p.add_argument("--lr",     type=float, default=0.0005)
    p.add_argument("--batch",  type=int,   default=16)
    p.add_argument("--val",    type=float, default=0.20,    help="Validation oranı")
    return p.parse_args()

# ── Transform ─────────────────────────────────────────────────────────────────

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),   # ← kritik düzeltme
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf

# ── Veri yükleme ──────────────────────────────────────────────────────────────

def build_loaders(data_dir, val_ratio, batch_size):
    train_tf, val_tf = get_transforms()

    full_ds = datasets.ImageFolder(data_dir, transform=train_tf)
    n_val   = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Validation setine ayrı transform uygula
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=val_tf)

    # Sınıf dengesizliği düzeltme — WeightedRandomSampler
    targets      = [full_ds.targets[i] for i in train_ds.indices]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"\nVeri seti  : {len(full_ds)} görüntü")
    print(f"Eğitim     : {n_train} | Validasyon: {n_val}")
    print(f"Sınıflar   : {full_ds.classes}\n")
    return train_loader, val_loader, full_ds.classes

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes, device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Sadece son katmanı değiştir; önceki katmanlar ImageNet ağırlıklarıyla başlasın
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# ── Metrik hesaplama ─────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds):
    avg = "macro"
    return {
        "accuracy":  round(accuracy_score(all_labels, all_preds) * 100, 2),
        "f1":        round(f1_score(all_labels, all_preds, average=avg, zero_division=0) * 100, 2),
        "precision": round(precision_score(all_labels, all_preds, average=avg, zero_division=0) * 100, 2),
        "recall":    round(recall_score(all_labels, all_preds, average=avg, zero_division=0) * 100, 2),
    }

# ── Epoch döngüsü ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = round(total_loss / len(loader.dataset), 4)
    return metrics

# ── Ana eğitim döngüsü ────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    train_loader, val_loader, classes = build_loaders(args.data, args.val, args.batch)

    model     = build_model(len(classes), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1      = 0.0
    best_epoch   = 0
    history      = []

    print(f"{'Epoch':>6} | {'Train Loss':>10} {'Train F1':>9} | {'Val Loss':>9} {'Val F1':>8} {'Val Acc':>8}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        tr  = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val = run_epoch(model, val_loader,   criterion, None,      device, training=False)
        scheduler.step()

        history.append({"epoch": epoch, "train": tr, "val": val})

        marker = " ★" if val["f1"] > best_f1 else ""
        print(f"{epoch:>6} | {tr['loss']:>10.4f} {tr['f1']:>8.2f}% | "
              f"{val['loss']:>9.4f} {val['f1']:>7.2f}% {val['accuracy']:>7.2f}%{marker}")

        # Best checkpoint kaydet
        if val["f1"] > best_f1:
            best_f1    = val["f1"]
            best_epoch = epoch
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "classes":     classes,
                "val_f1":      best_f1,
                "val_acc":     val["accuracy"],
            }, args.model)

    print(f"\n✓ En iyi model: Epoch {best_epoch} — Val F1: {best_f1:.2f}%")
    print(f"✓ Model kaydedildi: {args.model}")
    return history

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.data):
        print(f"HATA: '{args.data}' klasörü bulunamadı.")
        print("Klasör yapısı şöyle olmalı:")
        for cls in SINIFLAR:
            print(f"  data/{cls}/")
        raise SystemExit(1)

    train(args)
