"""
Script cepat untuk menghitung metrik evaluasi IndoBERT dan VADER,
lalu disimpan ke metrik_evaluasi_cache.json supaya server Django langsung siap.
"""
import torch
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

print("=" * 60)
print("  HITUNG METRIK EVALUASI MODEL")
print("=" * 60)

# Load dataset
df = pd.read_excel('dataset_pln_balanced_1000.xlsx')
df.dropna(subset=['teks_bersih'], inplace=True)

_, test_texts, _, test_labels = train_test_split(
    df['teks_bersih'].astype(str).tolist(),
    df['label'].astype(int).tolist(),
    test_size=0.2,
    random_state=42
)

print(f"Jumlah data test: {len(test_texts)}")

# === IndoBERT ===
print("\n--- Evaluasi IndoBERT ---")
tokenizer = AutoTokenizer.from_pretrained("./model_indobert_pln")
model = AutoModelForSequenceClassification.from_pretrained("./model_indobert_pln")

indobert_preds = []
for i, teks in enumerate(test_texts):
    inputs = tokenizer(teks, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    indobert_preds.append(pred)
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{len(test_texts)}")

metrik_indobert = {
    'akurasi': round(accuracy_score(test_labels, indobert_preds) * 100, 2),
    'presisi': round(precision_score(test_labels, indobert_preds, average='weighted', zero_division=0) * 100, 2),
    'recall': round(recall_score(test_labels, indobert_preds, average='weighted', zero_division=0) * 100, 2),
    'f1': round(f1_score(test_labels, indobert_preds, average='weighted', zero_division=0) * 100, 2),
    'kappa': round(cohen_kappa_score(test_labels, indobert_preds), 4),
}
print(f"  Akurasi: {metrik_indobert['akurasi']}%")
print(f"  Presisi: {metrik_indobert['presisi']}%")
print(f"  Recall:  {metrik_indobert['recall']}%")
print(f"  F1:      {metrik_indobert['f1']}%")
print(f"  Kappa:   {metrik_indobert['kappa']}")

# === VADER ===
print("\n--- Evaluasi VADER ---")
translator = Translator()
analyzer = SentimentIntensityAnalyzer()
vader_preds = []

for i, teks in enumerate(test_texts):
    try:
        teks_en = translator.translate(teks, src='id', dest='en').text
    except:
        teks_en = teks
    vs = analyzer.polarity_scores(teks_en)
    pred = 1 if vs['compound'] >= 0.05 else 0
    vader_preds.append(pred)
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(test_texts)}")

metrik_vader = {
    'akurasi': round(accuracy_score(test_labels, vader_preds) * 100, 2),
    'presisi': round(precision_score(test_labels, vader_preds, average='weighted', zero_division=0) * 100, 2),
    'recall': round(recall_score(test_labels, vader_preds, average='weighted', zero_division=0) * 100, 2),
    'f1': round(f1_score(test_labels, vader_preds, average='weighted', zero_division=0) * 100, 2),
    'kappa': round(cohen_kappa_score(test_labels, vader_preds), 4),
}
print(f"  Akurasi: {metrik_vader['akurasi']}%")
print(f"  Presisi: {metrik_vader['presisi']}%")
print(f"  Recall:  {metrik_vader['recall']}%")
print(f"  F1:      {metrik_vader['f1']}%")
print(f"  Kappa:   {metrik_vader['kappa']}")

# Simpan cache
cache = {'indobert': metrik_indobert, 'vader': metrik_vader}
with open('metrik_evaluasi_cache.json', 'w') as f:
    json.dump(cache, f, indent=2)

print("\n" + "=" * 60)
print("  SELESAI! Cache disimpan ke metrik_evaluasi_cache.json")
print("=" * 60)
