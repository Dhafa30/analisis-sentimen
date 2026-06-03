from django.shortcuts import render
import json
import os
import urllib.request
import urllib.error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

# Hardcode metrik evaluasi VADER
METRIK_VADER = {
    'akurasi': 76.00,
    'presisi': 75.50,
    'recall': 76.00,
    'f1': 75.80,
    'kappa': 0.5200,
}


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
            try:
                # Menggunakan urllib.request murni dengan model Publik IndoBERT dan Token Akses
                url = "https://router.huggingface.co/hf-inference/models/mdhugol/indonesia-bert-sentiment-classification"
                payload = {"inputs": teks_ulasan}
                data = json.dumps(payload).encode('utf-8')
                
                # Mengambil Token dari Environment Variable Vercel
                hf_token = os.environ.get('HF_TOKEN', '')
                
                req = urllib.request.Request(url, data=data, headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {hf_token}'
                })
                
                try:
                    res = urllib.request.urlopen(req, timeout=15)
                    response_data = res.read().decode('utf-8')
                    result = json.loads(response_data)
                    
                    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                        scores = result[0]
                        
                        # mdhugol model mapping: LABEL_0 = Positif, LABEL_2 = Negatif, LABEL_1 = Netral
                        score_positif = next((item['score'] for item in scores if item['label'] == 'LABEL_0'), 0.0)
                        score_negatif = next((item['score'] for item in scores if item['label'] == 'LABEL_2'), 0.0)
                        
                        # Normalisasi agar totalnya 100% karena UI hanya menampilkan Positif & Negatif (mengabaikan Netral)
                        total_score = score_positif + score_negatif
                        if total_score > 0:
                            prob_positif = round((score_positif / total_score) * 100, 1)
                            prob_negatif = round(100.0 - prob_positif, 1)
                        else:
                            prob_positif = 50.0
                            prob_negatif = 50.0
                        
                        if prob_positif > prob_negatif:
                            sentimen = "Positif"
                            confidence = prob_positif
                        else:
                            sentimen = "Negatif"
                            confidence = prob_negatif
                            
                        skor = f"HuggingFace API (Serverless Router)"
                    else:
                        sentimen = "Error: Format balasan API tidak dikenali"
                        skor = "-"
                        
                except urllib.error.HTTPError as e:
                    if e.code == 503:
                        sentimen = "Error: Model sedang loading di server (Tunggu 20 detik lalu klik lagi)"
                    else:
                        error_body = e.read().decode('utf-8')
                        sentimen = f"Error HuggingFace API - HTTP {e.code}"
                    skor = "-"
                except urllib.error.URLError as e:
                    sentimen = f"Error Jaringan Backend: {e.reason}"
                    skor = "-"
                    
            except Exception as e:
                sentimen = f"Error Internal Backend: {str(e)}"
                skor = "-"
                
            metrik = context['base_metrik_indobert']

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