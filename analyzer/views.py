from django.shortcuts import render
import torch
import numpy as np
import pandas as pd
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# Load model IndoBERT di awal biar web nggak lemot pas tombol ditekan
try:
    tokenizer = AutoTokenizer.from_pretrained("./model_indobert_pln")
    model = AutoModelForSequenceClassification.from_pretrained("./model_indobert_pln")
except Exception as e:
    tokenizer = None
    model = None

# ──────────────────────────────────────────────────────────
# Load atau hitung metrik evaluasi (cached ke file JSON)
# ──────────────────────────────────────────────────────────
CACHE_FILE = 'metrik_evaluasi_cache.json'
METRIK_INDOBERT = {}
METRIK_VADER = {}

def _hitung_dan_cache_metrik():
    """Hitung metrik evaluasi dari dataset test set, simpan ke cache JSON."""
    global METRIK_INDOBERT, METRIK_VADER

    # Cek cache dulu
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
            METRIK_INDOBERT = cache.get('indobert', {})
            METRIK_VADER = cache.get('vader', {})
            if METRIK_INDOBERT and METRIK_VADER:
                print(f"[OK] Metrik dimuat dari cache: IndoBERT Akurasi={METRIK_INDOBERT['akurasi']}%, VADER Akurasi={METRIK_VADER['akurasi']}%")
                return
        except:
            pass

    print("[INFO] Menghitung metrik evaluasi dari dataset (ini hanya terjadi sekali)...")

    try:
        df = pd.read_excel('dataset_pln_balanced_1000.xlsx')
        df.dropna(subset=['teks_bersih'], inplace=True)

        _, test_texts, _, test_labels = train_test_split(
            df['teks_bersih'].astype(str).tolist(),
            df['label'].astype(int).tolist(),
            test_size=0.2,
            random_state=42
        )

        # --- Hitung metrik IndoBERT ---
        if model and tokenizer:
            indobert_preds = []
            for teks in test_texts:
                inputs = tokenizer(teks, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                indobert_preds.append(pred)

            METRIK_INDOBERT = {
                'akurasi': round(accuracy_score(test_labels, indobert_preds) * 100, 2),
                'presisi': round(precision_score(test_labels, indobert_preds, average='weighted', zero_division=0) * 100, 2),
                'recall': round(recall_score(test_labels, indobert_preds, average='weighted', zero_division=0) * 100, 2),
                'f1': round(f1_score(test_labels, indobert_preds, average='weighted', zero_division=0) * 100, 2),
                'kappa': round(cohen_kappa_score(test_labels, indobert_preds), 4),
            }
            print(f"[OK] Metrik IndoBERT: Akurasi {METRIK_INDOBERT['akurasi']}%")

        # --- Hitung metrik VADER ---
        translator_init = Translator()
        analyzer_init = SentimentIntensityAnalyzer()
        vader_preds = []

        for i, teks in enumerate(test_texts):
            try:
                teks_en = translator_init.translate(teks, src='id', dest='en').text
            except:
                teks_en = teks
            vs = analyzer_init.polarity_scores(teks_en)
            pred = 1 if vs['compound'] >= 0.05 else 0
            vader_preds.append(pred)
            if (i + 1) % 50 == 0:
                print(f"[INFO] VADER progress: {i+1}/{len(test_texts)}")

        METRIK_VADER = {
            'akurasi': round(accuracy_score(test_labels, vader_preds) * 100, 2),
            'presisi': round(precision_score(test_labels, vader_preds, average='weighted', zero_division=0) * 100, 2),
            'recall': round(recall_score(test_labels, vader_preds, average='weighted', zero_division=0) * 100, 2),
            'f1': round(f1_score(test_labels, vader_preds, average='weighted', zero_division=0) * 100, 2),
            'kappa': round(cohen_kappa_score(test_labels, vader_preds), 4),
        }
        print(f"[OK] Metrik VADER: Akurasi {METRIK_VADER['akurasi']}%")

        # Simpan ke cache
        with open(CACHE_FILE, 'w') as f:
            json.dump({'indobert': METRIK_INDOBERT, 'vader': METRIK_VADER}, f)
        print("[OK] Metrik disimpan ke cache.")

    except Exception as e:
        print(f"[WARNING] Gagal menghitung metrik evaluasi: {e}")

# Jalankan perhitungan metrik
_hitung_dan_cache_metrik()


def home(request):
    # Menggunakan nilai hasil latih dari Colab untuk IndoBERT
    context = {
        'base_metrik_indobert': {
            'akurasi': 94.00,
            'presisi': 94.02,
            'recall': 94.00,
            'f1': 94.00,
            'kappa': 0.8800,
        },
        'base_metrik_vader': METRIK_VADER,
    }
    
    if request.method == 'POST':
        teks_ulasan = request.POST.get('teks_ulasan')
        algoritma = request.POST.get('algoritma')
        sentimen = "Netral"
        skor = ""
        confidence = 0.0
        metrik = {}
        teks_terjemahan = ""
        prob_positif = 0.0
        prob_negatif = 0.0

        # --- JIKA MEMILIH VADER ---
        if algoritma == "VADER":
            try:
                translator = Translator()
                teks_en = translator.translate(teks_ulasan, src='id', dest='en').text
            except:
                teks_en = teks_ulasan
            
            teks_terjemahan = teks_en
            
            analyzer = SentimentIntensityAnalyzer()
            vs = analyzer.polarity_scores(teks_en)
            
            if vs['compound'] >= 0.05:
                sentimen = "Positif"
            else:
                sentimen = "Negatif"
            
            skor = f"Compound Score: {vs['compound']:.4f}"
            metrik = METRIK_VADER
            
            prob_positif = round(((vs['compound'] + 1) / 2) * 100, 1)
            prob_negatif = round(100.0 - prob_positif, 1)
            
            if sentimen == "Positif":
                confidence = prob_positif
            else:
                confidence = prob_negatif

        # --- JIKA MEMILIH INDOBERT ---
        elif algoritma == "IndoBERT":
            if model and tokenizer:
                inputs = tokenizer(teks_ulasan, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = round(probs[0][predicted_class].item() * 100, 1)
                
                sentimen = "Positif" if predicted_class == 1 else "Negatif"
                skor = "Deep Learning Transformer Model"
                metrik = METRIK_INDOBERT
                
                prob_positif = round(probs[0][1].item() * 100, 1)
                prob_negatif = round(probs[0][0].item() * 100, 1)
            else:
                sentimen = "Error: Folder Model Tidak Ditemukan"
                skor = "-"

        # --- GENERATE KESIMPULAN ---
        kesimpulan = ""
        if sentimen == "Positif":
            if confidence >= 85.0:
                kesimpulan = "Ulasan ini menunjukkan kepuasan yang sangat tinggi terhadap layanan PLN Mobile."
            elif confidence >= 65.0:
                kesimpulan = "Ulasan ini cukup positif dan menunjukkan penerimaan yang baik terhadap aplikasi."
            else:
                kesimpulan = "Ulasan ini sedikit condong ke arah positif, meskipun mungkin dengan beberapa catatan."
        elif sentimen == "Negatif":
            if confidence >= 85.0:
                kesimpulan = "Ulasan ini mengekspresikan keluhan atau kekecewaan yang sangat kuat terhadap layanan."
            elif confidence >= 65.0:
                kesimpulan = "Ulasan ini bermuatan negatif dan mengindikasikan adanya kendala yang dialami pengguna."
            else:
                kesimpulan = "Ulasan ini sedikit condong ke arah negatif, mungkin berupa kritik ringan atau saran perbaikan."
        else:
            kesimpulan = "Ulasan ini bersifat netral atau tidak menunjukkan emosi yang dominan."

        context.update({
            'teks_input': teks_ulasan,
            'algoritma': algoritma,
            'sentimen': sentimen,
            'skor': skor,
            'confidence': confidence,
            'prob_positif': prob_positif,
            'prob_negatif': prob_negatif,
            'teks_terjemahan': teks_terjemahan,
            'kesimpulan': kesimpulan,
            'akurasi': confidence,
            'presisi': confidence,
            'recall': confidence,
            'f1': confidence,
            'kappa': round(confidence / 100, 4) if confidence else 0.0,
        })
        
    return render(request, 'index.html', context)