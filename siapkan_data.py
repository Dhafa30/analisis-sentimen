import pandas as pd
import re
from google_play_scraper import Sort, reviews
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split

# =====================================================================
# TAHAP 1: SCRAPING DATA PLN MOBILE (BALANCED DATASET 1000 ULASAN)
# =====================================================================
print("1. Sedang menyedot ulasan PLN Mobile dari Play Store...")
print("Target: 250 (Bintang 1), 250 (Bintang 2), 250 (Bintang 4), 250 (Bintang 5)...")

# Kita pasang target kuota masing-masing rating
kuota = {1: 250, 2: 250, 4: 250, 5: 250}
data_terkumpul = {1: [], 2: [], 4: [], 5: []}
token_lanjut = None

# Sistem akan terus mencari ke belakang sampai semua kuota terpenuhi
while True:
    res, token_lanjut = reviews(
        'com.icon.pln123',
        lang='id', 
        country='id',
        sort=Sort.NEWEST, 
        count=1000,
        continuation_token=token_lanjut
    )
    
    for r in res:
        skor = r['score']
        # Masukkan ke keranjang jika rating sesuai (1,2,4,5) dan kuota belum penuh
        if skor in kuota and len(data_terkumpul[skor]) < kuota[skor]:
            data_terkumpul[skor].append({
                'waktu_ulasan': r['at'],
                'nama_pengguna': r['userName'],
                'teks_mentah': r['content'],
                'rating': skor
            })
            
    # Hentikan pencarian jika semua keranjang sudah terisi masing-masing 250
    if all(len(data_terkumpul[skor]) == kuota[skor] for skor in kuota):
        break
    
    # Berhenti jika server Google Play Store sudah kehabisan ulasan (jaga-jaga)
    if not token_lanjut:
        break

# Satukan semua keranjang menjadi satu tabel
semua_data = []
for skor in data_terkumpul:
    semua_data.extend(data_terkumpul[skor])

df = pd.DataFrame(semua_data)

# Buat Label Sentimen: Bintang 4 & 5 = Positif (1), Bintang 1 & 2 = Negatif (0)
df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

print(f"✅ Scraping sukses! Terkumpul tepat {len(df)} data ulasan yang seimbang.\n")

# =====================================================================
# TAHAP 2: PREPROCESSING (PEMBERSIHAN TEKS & STEMMING SASTRAWI)
# =====================================================================
print("2. Memulai proses Preprocessing (Cleaning & Stemming)...")
print("⏳ Sabar cuy, proses Sastrawi ini makan waktu lumayan lama. Tinggal ngopi dulu aja...")

# Siapkan mesin pemotong imbuhan
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def bersihkan_teks(teks):
    teks = str(teks).lower() # Kecilkan huruf
    teks = re.sub(r'http\S+|www\S+|https\S+', '', teks, flags=re.MULTILINE) # Hapus link
    teks = re.sub(r'[^a-z\s]', ' ', teks) # Hapus angka dan simbol
    teks = re.sub(r'\n', ' ', teks) # Hapus enter
    teks = re.sub(r'\s+', ' ', teks).strip() # Rapikan spasi
    teks = stemmer.stem(teks) # Kembalikan ke kata dasar
    return teks

# Proses pembersihan diaplikasikan ke semua data
df['teks_bersih'] = df['teks_mentah'].apply(bersihkan_teks)

# Buang data yang jadi kosong setelah dibersihkan
df.dropna(subset=['teks_bersih'], inplace=True)
df = df[df['teks_bersih'].str.strip() != '']

print("✅ Preprocessing selesai!\n")

# =====================================================================
# TAHAP 3: EXPORT KE EXCEL & SPLIT DATASET
# =====================================================================
print("3. Membagi dataset dan menyimpan ke Excel untuk lampiran skripsi...")

# Susun urutan kolom biar rapi pas masuk Excel
df_final = df[['waktu_ulasan', 'nama_pengguna', 'rating', 'label', 'teks_mentah', 'teks_bersih']]

# Simpan ke format Excel
nama_file = 'dataset_pln_balanced_1000.xlsx'
df_final.to_excel(nama_file, index=False)

# Membagi data untuk mesin belajar (80% Training, 20% Testing)
train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42)

print(f"✅ Data berhasil disimpan dan dibagi!")
print(f"📊 Total Data Training: {len(train_df)} baris")
print(f"📊 Total Data Uji (Testing): {len(test_df)} baris\n")
print(f"🎉 SELESAI! Silakan cek file '{nama_file}' di folder kiri lu.")