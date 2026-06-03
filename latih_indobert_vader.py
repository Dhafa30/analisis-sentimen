import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, cohen_kappa_score, confusion_matrix, 
                             ConfusionMatrixDisplay)

# =====================================================================
# FUNGSI UNTUK MENGHITUNG METRIK & MENGGAMBAR CONFUSION MATRIX
# =====================================================================
def evaluasi_model(nama_model, y_test, y_pred):
    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print(f"\n📊 --- Hasil Evaluasi {nama_model} ---")
    print(f"Akurasi       : {akurasi * 100:.2f}%")
    print(f"Presisi       : {presisi * 100:.2f}%")
    print(f"Recall        : {recall * 100:.2f}%")
    print(f"F1-Score      : {f1 * 100:.2f}%")
    print(f"Cohen's Kappa : {kappa:.4f}")
    
    # Bikin gambar Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negatif (0)', 'Positif (1)'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {nama_model}')
    
    # Simpan gambar kualitas HD (dpi=300)
    nama_file_gambar = f'confusion_matrix_{nama_model.lower()}.png'
    plt.savefig(nama_file_gambar, dpi=300, bbox_inches='tight') 
    plt.close() 
    print(f"📸 Gambar disimpan: {nama_file_gambar}")

# =====================================================================
# TAHAP 1: MEMBACA DATASET BERSIH
# =====================================================================
print("1. Membaca dataset dari Excel...")
df = pd.read_excel('dataset_pln_balanced_1000.xlsx')
df.dropna(subset=['teks_bersih'], inplace=True)

# Membagi data (80% Training, 20% Testing)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['teks_bersih'].astype(str).tolist(), 
    df['label'].astype(int).tolist(), 
    test_size=0.2, 
    random_state=42
)

# =====================================================================
# TAHAP 2: EVALUASI VADER (LEXICON)
# =====================================================================
print("\n2. Mengeksekusi algoritma VADER pada Data Uji (Testing)...")
print("⏳ Sabar cuy, lagi translate ke Bahasa Inggris...")
translator = Translator()
analyzer = SentimentIntensityAnalyzer()
vader_preds = []

# Hanya tes di data test (200 data) agar perbandingan adil dengan model ML
for teks in test_texts:
    try:
        # VADER butuh bahasa Inggris
        teks_en = translator.translate(teks, src='id', dest='en').text
    except:
        teks_en = teks # Backup kalau Google API limit
    
    vs = analyzer.polarity_scores(teks_en)
    pred = 1 if vs['compound'] >= 0.05 else 0
    vader_preds.append(pred)

evaluasi_model("VADER", test_labels, vader_preds)

# =====================================================================
# TAHAP 3: PERSIAPAN & TRAINING INDOBERT
# =====================================================================
print("\n3. Persiapan Training IndoBERT...")
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

class PLNDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = PLNDataset(train_encodings, train_labels)
test_dataset = PLNDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3, # Bisa dinaikkan kalau mau akurasi lebih tajam
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("\n4. Mulai Training IndoBERT...")
print("⚠️ Peringatan: Proses ini lumayan berat. Pastikan laptop dicolok charger ya cuy!")
trainer.train()

# =====================================================================
# TAHAP 4: EVALUASI INDOBERT & SAVE MODEL
# =====================================================================
print("\n5. Menghitung Evaluasi IndoBERT...")
predictions = trainer.predict(test_dataset)
indobert_preds = np.argmax(predictions.predictions, axis=-1)
evaluasi_model("IndoBERT", test_labels, indobert_preds)

print("\n6. Membungkus model ke dalam folder untuk Web Django...")
model.save_pretrained("./model_indobert_pln")
tokenizer.save_pretrained("./model_indobert_pln")

print("🎉 SELESAI! Folder 'model_indobert_pln' dan gambar Confusion Matrix udah aman!")