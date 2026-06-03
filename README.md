# Analisis Sentimen PLN Mobile

Aplikasi web berbasis Django untuk melakukan analisis sentimen terhadap ulasan aplikasi PLN Mobile. Proyek ini membandingkan dua pendekatan utama dalam Natural Language Processing (NLP):
1. **IndoBERT** (Deep Learning / Transformer Model)
2. **VADER** (Lexicon-based Model)

## 🌟 Fitur Utama
- **Antarmuka Modern (Premium UI)**: Desain UI responsif menggunakan efek *Glassmorphism*, palet warna yang premium (kombinasi navy blue, cyan, dan amber), serta animasi partikel di latar belakang.
- **Klasifikasi Sentimen Biner**: Mengklasifikasikan ulasan pengguna menjadi **Positif** atau **Negatif**.
- **Perbandingan Model**: Pengguna dapat memilih antara model IndoBERT (bahasa Indonesia asli) atau VADER (menerjemahkan teks ke bahasa Inggris terlebih dahulu).
- **Teks Terjemahan Otomatis**: Khusus untuk VADER, aplikasi akan menampilkan teks hasil terjemahan bahasa Inggris yang digunakan untuk prediksi skor.
- **Metrik Evaluasi Real-time**: Menampilkan performa model berdasarkan *test set* berupa metrik Akurasi, Presisi, Recall, F1-Score, dan Cohen's Kappa.
- **Confidence Level**: Menunjukkan tingkat keyakinan (probabilitas) dari model terhadap hasil prediksinya dalam bentuk persentase dan *progress bar* visual.

## 🛠️ Teknologi yang Digunakan
- **Backend Framework**: Django (Python)
- **Machine Learning / NLP**: 
  - `transformers` (Hugging Face) dan `PyTorch` untuk menjalankan model IndoBERT.
  - `vaderSentiment` untuk analisis sentimen berbasis leksikon.
  - `scikit-learn` untuk perhitungan metrik evaluasi.
- **Utilitas**: 
  - `googletrans` (untuk auto-translate VADER).
  - `pandas` (untuk load dataset).
- **Frontend**: HTML5, Vanilla CSS3 (dengan animasi *keyframes* dan *backdrop-filter*).

## 📂 Struktur Direktori Utama
- `pln_sentiment/`: Folder utama konfigurasi Django (settings, urls).
- `analyzer/`: Aplikasi Django yang berisi logika pemrosesan sentimen (`views.py`) dan rute (`urls.py`).
  - `templates/index.html`: Kode antarmuka (frontend) aplikasi.
- `model_indobert_pln/`: Folder yang berisi *weights* dan konfigurasi model IndoBERT yang telah di-finetune.
- `dataset_pln_balanced_1000.xlsx`: Dataset yang digunakan untuk menghitung matriks evaluasi di awal.
- `metrik_evaluasi_cache.json`: File cache untuk menyimpan hasil kalkulasi metrik agar server tidak perlu menghitung ulang setiap kali *restart*.

## 🚀 Cara Menjalankan Aplikasi Lokal

1. **Pastikan Python sudah terinstal** (disarankan versi 3.9+).
2. **Instal seluruh *dependencies*** yang dibutuhkan:
   ```bash
   pip install django torch numpy pandas transformers vaderSentiment googletrans==4.0.0-rc1 scikit-learn openpyxl
   ```
3. **Jalankan server Django**:
   Pastikan Anda berada di dalam folder yang sama dengan `manage.py`, lalu ketikkan:
   ```bash
   python manage.py runserver
   ```
4. Buka peramban (browser) dan akses alamat: **`http://127.0.0.1:8000/`**

> **Catatan:** Pada saat pertama kali dijalankan, server mungkin membutuhkan waktu beberapa detik untuk *loading* model IndoBERT ke memori dan menghitung metrik evaluasi awal (jika cache belum ada).

## 🔄 Alur Proses (Flow) Aplikasi
1. Pengguna memasukkan teks keluhan/ulasan PLN Mobile ke dalam form dan memilih model.
2. Ketika tombol ditekan, data dikirim ke `views.py` melalui metode `POST`.
3. **Jika IndoBERT dipilih:** Teks langsung di-tokenisasi dan dimasukkan ke dalam model *Transformer*. Model mengembalikan probabilitas, dan probabilitas tertinggi menentukan apakah sentimen Positif (kelas 1) atau Negatif (kelas 0).
4. **Jika VADER dipilih:** Teks diterjemahkan ke bahasa Inggris menggunakan `googletrans`. Teks bahasa Inggris tersebut dinilai oleh *SentimentIntensityAnalyzer*. Skor *compound* >= 0.05 dianggap Positif, selebihnya Negatif.
5. Prediksi, *confidence score*, teks terjemahan (untuk VADER), dan metrik evaluasi dikirim kembali dan dirender ke dalam `index.html`.

## 📜 Lisensi & Atribusi
- Model IndoBERT dilatih dari dataset publik ulasan aplikasi PLN Mobile.
- VADER Sentiment adalah model *open-source* berbasis aturan (lexicon-rule based).
