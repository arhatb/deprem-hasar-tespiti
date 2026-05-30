"""
app.py — Deprem Sonrası Yapı Risk Analizi
==========================================
Çalıştırma:
  streamlit run app.py

Özellikler:
  - 5 sınıflı hasar tespiti (yıkılmış → hasarsız)
  - Grad-CAM ısı haritası görselleştirme
  - Sınıf olasılık grafiği
  - Toplu görüntü analizi (ZIP yükleme)
  - EXIF GPS koordinatı okuma
  - CSV rapor indirme
  - Sorumluluk reddi / yasal uyarı
"""

import io
import os
import zipfile
import csv
import tempfile

import streamlit as st
import torch
from PIL import Image
import numpy as np

from utils import (
    load_model,
    preprocess_image,
    predict,
    is_building,
    gradcam_overlay,
    get_gps_from_exif,
    HASAR_SEVIYELERI,
)

# ── Sayfa yapılandırması ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Deprem Hasar Tespiti",
    page_icon="🏚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.hasar-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 18px;
    font-weight: 600;
    color: white;
    margin-bottom: 8px;
}
.uyari-kutu {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 14px;
    color: #555;
    margin-top: 16px;
}
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# ── Model yükleme (cache) ─────────────────────────────────────────────────────

MODEL_PATH = "deprem_modeli.pth"

@st.cache_resource(show_spinner="Model yükleniyor…")
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    return load_model(MODEL_PATH)

model, classes, device = get_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Ayarlar")

    goster_gradcam = st.toggle("Grad-CAM ısı haritası", value=True,
                               help="Modelin hangi bölgeye odaklandığını gösterir")
    bina_kontrol   = st.toggle("Bina varlık kontrolü", value=True,
                               help="Görüntüde bina olup olmadığını önce kontrol et")

    st.divider()
    st.caption("**Sınıflar:**")
    for cls, info in HASAR_SEVIYELERI.items():
        st.markdown(
            f"<span style='color:{info['renk']};font-weight:600'>"
            f"{'●' * info['seviye']}</span> {info['tr']}",
            unsafe_allow_html=True,
        )

    st.divider()
    if model:
        st.success(f"✓ Model yüklendi")
        if classes:
            st.caption(f"Sınıf sayısı: {len(classes)}")
    else:
        st.error(f"Model bulunamadı: `{MODEL_PATH}`")
        st.info("Önce `python egit.py` çalıştırın.")

# ── Başlık ────────────────────────────────────────────────────────────────────

st.title("🏚️ Deprem Sonrası Yapı Risk Analizi")
st.caption(
    "Bu sistem **kesin hasar tespiti yapmaz** — deprem sonrası hızlı "
    "**ön değerlendirme ve önceliklendirme** amacıyla geliştirilmiştir."
)

if model is None:
    st.warning("⚠️ Model dosyası bulunamadı. Lütfen önce modeli eğitin: `python egit.py`")
    st.stop()

# ── Sekmeler ─────────────────────────────────────────────────────────────────

tab_tek, tab_toplu = st.tabs(["📷 Tek Görüntü", "📦 Toplu Analiz (ZIP)"])

# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 1 — TEK GÖRÜNTÜ
# ═══════════════════════════════════════════════════════════════════════════════

with tab_tek:
    uploaded = st.file_uploader(
        "Bir bina / enkaz fotoğrafı yükleyin",
        type=["jpg", "jpeg", "png", "webp"],
        key="tek",
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        col_img, col_sonuc = st.columns([1, 1], gap="large")

        with col_img:
            st.image(pil_img, caption="Yüklenen görüntü", use_container_width=True)

            # GPS
            lat, lon = get_gps_from_exif(pil_img)
            if lat and lon:
                st.caption(f"📍 GPS: {lat}, {lon}")
                st.map({"lat": [lat], "lon": [lon]}, zoom=14)

        with col_sonuc:
            with st.spinner("Analiz yapılıyor…"):
                # Bina kontrolü
                if bina_kontrol and not is_building(pil_img):
                    st.warning(
                        "⚠️ Görüntüde bina tespit edilemedi. "
                        "Bina veya enkaz içeren bir fotoğraf yükleyin."
                    )
                    st.stop()

                tensor     = preprocess_image(pil_img)
                pred_cls, confidence, prob_dict = predict(model, tensor, classes, device)

            # Sonuç
            info = HASAR_SEVIYELERI.get(pred_cls, {
                "tr": pred_cls, "renk": "#888", "seviye": 0
            })

            renk = info["renk"]
            isim = info["tr"]
            st.markdown(
                f"<div class='hasar-badge' style='background:{renk}'>{isim}</div>",
                unsafe_allow_html=True,
            )

            seviye = info.get("seviye", 0)
            st.progress(seviye / 5, text=f"Risk seviyesi: {seviye}/5")
            st.metric("Güven skoru", f"%{confidence*100:.1f}")

            # Olasılık grafiği
            st.subheader("Sınıf olasılıkları")
            prob_labels = []
            prob_values = []
            prob_colors = []
            for cls_name in classes:
                meta = HASAR_SEVIYELERI.get(cls_name, {})
                prob_labels.append(meta.get("tr", cls_name))
                prob_values.append(round(prob_dict.get(cls_name, 0) * 100, 1))
                prob_colors.append(meta.get("renk", "#888"))

            for label, val, color in zip(prob_labels, prob_values, prob_colors):
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
                    f"<span style='min-width:120px;font-size:13px'>{label}</span>"
                    f"<div style='flex:1;background:#eee;border-radius:4px;height:16px'>"
                    f"<div style='width:{val}%;background:{color};height:100%;border-radius:4px'></div>"
                    f"</div><span style='min-width:42px;font-size:13px;text-align:right'>%{val}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Grad-CAM
            if goster_gradcam:
                st.subheader("Grad-CAM — Odak bölgesi")
                with st.spinner("Isı haritası hesaplanıyor…"):
                    target_idx = classes.index(pred_cls) if pred_cls in classes else None
                    overlay    = gradcam_overlay(model, pil_img, target_idx)
                st.image(overlay, caption="Kırmızı = modelin dikkat ettiği bölge",
                         use_container_width=True)

            # Uyarı
            st.markdown("""
<div class='uyari-kutu'>
ℹ️ Bu sonuç <strong>kesin hasar tespiti değildir</strong>. Saha ekipleri için
<strong>önceliklendirme</strong> amacıyla üretilmiştir. Kesin karar için lisanslı
inşaat mühendisi incelemesi zorunludur. Sistem hukuki sorumluluk doğurmaz.
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 2 — TOPLU ANALİZ
# ═══════════════════════════════════════════════════════════════════════════════

with tab_toplu:
    st.info("Birden fazla fotoğrafı ZIP dosyası olarak yükleyin. "
            "Sonuçlar tablo ve CSV olarak gösterilir.")

    zip_file = st.file_uploader(
        "ZIP dosyası yükleyin (jpg/png içeren)",
        type=["zip"],
        key="toplu",
    )

    if zip_file:
        sonuclar = []

        with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
            goruntu_listesi = [
                f for f in zf.namelist()
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                and not f.startswith("__MACOSX")
            ]

        if not goruntu_listesi:
            st.warning("ZIP içinde geçerli görüntü bulunamadı.")
        else:
            st.write(f"**{len(goruntu_listesi)} görüntü** analiz edilecek.")
            progress = st.progress(0)
            durum    = st.empty()

            zip_bytes = zip_file.getvalue()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for i, fname in enumerate(goruntu_listesi):
                    durum.text(f"İşleniyor: {fname}")
                    try:
                        raw      = zf.read(fname)
                        pil_img  = Image.open(io.BytesIO(raw)).convert("RGB")
                        tensor   = preprocess_image(pil_img)
                        pred_cls, conf, prob_dict = predict(model, tensor, classes, device)
                        info     = HASAR_SEVIYELERI.get(pred_cls, {})
                        lat, lon = get_gps_from_exif(pil_img)

                        sonuclar.append({
                            "Dosya":       fname,
                            "Hasar Sınıfı": info.get("tr", pred_cls),
                            "Risk Seviyesi": info.get("seviye", "-"),
                            "Güven (%)":   round(conf * 100, 1),
                            "GPS Lat":     lat or "",
                            "GPS Lon":     lon or "",
                        })
                    except Exception as e:
                        sonuclar.append({
                            "Dosya":        fname,
                            "Hasar Sınıfı": "HATA",
                            "Risk Seviyesi": "-",
                            "Güven (%)":    "-",
                            "GPS Lat":      "",
                            "GPS Lon":      "",
                        })

                    progress.progress((i + 1) / len(goruntu_listesi))

            durum.text("✓ Analiz tamamlandı.")

            # Tablo
            st.dataframe(sonuclar, use_container_width=True)

            # Özet istatistik
            gecerli = [s for s in sonuclar if s["Hasar Sınıfı"] != "HATA"]
            if gecerli:
                st.subheader("Özet")
                cols = st.columns(4)
                toplam   = len(gecerli)
                yuksek   = sum(1 for s in gecerli if s["Risk Seviyesi"] in (4, 5))
                orta     = sum(1 for s in gecerli if s["Risk Seviyesi"] == 3)
                dusuk    = sum(1 for s in gecerli if s["Risk Seviyesi"] in (1, 2))
                cols[0].metric("Toplam analiz", toplam)
                cols[1].metric("🔴 Yüksek risk", yuksek)
                cols[2].metric("🟠 Orta risk",   orta)
                cols[3].metric("🟢 Düşük risk",  dusuk)

            # CSV indirme
            csv_buf = io.StringIO()
            if sonuclar:
                writer = csv.DictWriter(csv_buf, fieldnames=sonuclar[0].keys())
                writer.writeheader()
                writer.writerows(sonuclar)

            st.download_button(
                "⬇️ CSV olarak indir",
                data=csv_buf.getvalue().encode("utf-8-sig"),
                file_name="hasar_tespiti_raporu.csv",
                mime="text/csv",
            )

            st.markdown("""
<div class='uyari-kutu'>
ℹ️ Toplu analiz sonuçları <strong>kesin hasar tespiti değildir</strong>. 
Önceliklendirme amacıyla üretilmiştir. Her yapı için ayrıca lisanslı 
mühendis incelemesi yapılması zorunludur.
</div>""", unsafe_allow_html=True)
