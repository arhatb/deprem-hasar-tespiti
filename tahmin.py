"""
tahmin.py — Komut Satırı Tahmin Aracı
=======================================
Tek bir görüntü veya klasör üzerinde tahmin yapar.

Kullanım:
  python tahmin.py bina.jpg
  python tahmin.py goruntu_klasoru/
  python tahmin.py bina.jpg --model deprem_modeli.pth --gradcam
"""

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

from utils import load_model, preprocess_image, predict, is_building, HASAR_SEVIYELERI

DESTEKLENEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Argümanlar ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Deprem hasar tespiti — komut satırı aracı")
    p.add_argument("girdi",         help="Görüntü dosyası veya klasör yolu")
    p.add_argument("--model",       default="deprem_modeli.pth")
    p.add_argument("--gradcam",     action="store_true", help="Grad-CAM ısı haritasını kaydet")
    p.add_argument("--no-bina",     action="store_true", help="Bina varlık kontrolünü atla")
    return p.parse_args()

# ── Tek görüntü işleme ────────────────────────────────────────────────────────

def analiz_et(image_path: str, model, classes, device, args) -> dict:
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"dosya": image_path, "hata": str(e)}

    # Bina kontrolü
    if not args.no_bina and not is_building(pil_img):
        return {
            "dosya":  image_path,
            "sonuc":  "BİNA YOK",
            "detay":  "Görüntüde bina tespit edilemedi.",
        }

    tensor = preprocess_image(pil_img)
    pred_cls, confidence, prob_dict = predict(model, tensor, classes, device)
    info = HASAR_SEVIYELERI.get(pred_cls, {"tr": pred_cls, "renk": "#888", "seviye": 0})

    sonuc = {
        "dosya":      image_path,
        "sinif":      pred_cls,
        "sinif_tr":   info["tr"],
        "seviye":     info["seviye"],
        "guven":      round(confidence * 100, 1),
        "olasiliklar": {cls: f"%{v*100:.1f}" for cls, v in prob_dict.items()},
    }

    # Grad-CAM
    if args.gradcam:
        from utils import gradcam_overlay
        overlay_path = Path(image_path).stem + "_gradcam.png"
        target_idx   = classes.index(pred_cls) if pred_cls in classes else None
        overlay      = gradcam_overlay(model, pil_img, target_idx)
        overlay.save(overlay_path)
        sonuc["gradcam"] = overlay_path

    return sonuc

# ── Sonuç yazdırma ────────────────────────────────────────────────────────────

RENKLER = {5: "\033[91m", 4: "\033[91m", 3: "\033[93m", 2: "\033[93m", 1: "\033[92m"}
RESET   = "\033[0m"

def yazdir(sonuc: dict):
    if "hata" in sonuc:
        print(f"[HATA] {sonuc['dosya']}: {sonuc['hata']}")
        return
    if sonuc.get("sonuc") == "BİNA YOK":
        print(f"[BİNA YOK] {sonuc['dosya']}: {sonuc['detay']}")
        return

    seviye = sonuc["seviye"]
    renk   = RENKLER.get(seviye, "")
    sembol = "●" * seviye + "○" * (5 - seviye)

    print(f"\n{'─'*55}")
    print(f"Dosya   : {sonuc['dosya']}")
    print(f"Sonuç   : {renk}{sonuc['sinif_tr']}{RESET}  [{sembol}]  Seviye {seviye}/5")
    print(f"Güven   : %{sonuc['guven']}")
    print("Olasılıklar:")
    for cls, val in sonuc["olasiliklar"].items():
        info = HASAR_SEVIYELERI.get(cls, {})
        print(f"  {info.get('tr', cls):<15}: {val}")
    if "gradcam" in sonuc:
        print(f"Grad-CAM: {sonuc['gradcam']}")
    print(f"{'─'*55}")
    print("\n⚠️  Bu sonuç kesin hasar tespiti değildir. Mühendis incelemesi gereklidir.\n")

# ── Ana ───────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()

    if not os.path.exists(args.model):
        print(f"HATA: Model bulunamadı: {args.model}")
        print("Önce 'python egit.py' çalıştırın.")
        sys.exit(1)

    model, classes, device = load_model(args.model)
    print(f"Model yüklendi ({len(classes)} sınıf, cihaz: {device})")

    girdi = args.girdi

    if os.path.isfile(girdi):
        sonuc = analiz_et(girdi, model, classes, device, args)
        yazdir(sonuc)

    elif os.path.isdir(girdi):
        dosyalar = [
            str(p) for p in Path(girdi).rglob("*")
            if p.suffix.lower() in DESTEKLENEN
        ]
        print(f"\n{len(dosyalar)} görüntü bulundu.\n")
        for dosya in sorted(dosyalar):
            sonuc = analiz_et(dosya, model, classes, device, args)
            yazdir(sonuc)
    else:
        print(f"HATA: '{girdi}' bulunamadı.")
        sys.exit(1)

if __name__ == "__main__":
    main()
