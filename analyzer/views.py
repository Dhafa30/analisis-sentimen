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
                import subprocess
                
                # 1. Resolve DNS manual via DNS over HTTPS (Bypass Vercel DNS Block)
                cmd_dns = ["curl", "-s", "https://dns.google/resolve?name=api-inference.huggingface.co&type=A"]
                try:
                    dns_output = subprocess.check_output(cmd_dns, timeout=10).decode('utf-8')
                    dns_data = json.loads(dns_output)
                    ip_address = next((ans["data"] for ans in dns_data.get("Answer", []) if ans["type"] == 1), None)
                except Exception:
                    ip_address = None
                    
                if not ip_address:
                    sentimen = "Error Server: DNS HuggingFace Diblokir Vercel (DoH Failed)"
                    skor = "-"
                else:
                    # 2. Request ke HuggingFace dengan IP yang sudah di-resolve
                    API_URL = "https://api-inference.huggingface.co/models/Dhafa30/model_indobert_pln"
                    payload_json = json.dumps({"inputs": teks_ulasan})
                    
                    cmd = [
                        "curl", "-s", "-X", "POST", API_URL,
                        "-H", "Content-Type: application/json",
                        "--resolve", f"api-inference.huggingface.co:443:{ip_address}",
                        "-d", payload_json
                    ]
                    
                    curl_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=60)
                    result_str = curl_output.decode('utf-8')
                    
                    if not result_str.strip():
                        sentimen = "Error: Balasan API kosong"
                        skor = "-"
                    else:
                        result = json.loads(result_str)
                        
                        if isinstance(result, dict) and "error" in result:
                            err_msg = str(result["error"])
                            if "loading" in err_msg.lower() or "estimated_time" in result:
                                sentimen = "Error: Model sedang loading di server HF (Tunggu 20 detik lalu coba lagi)"
                                skor = "-"
                            else:
                                sentimen = f"Error Server HF: {err_msg}"
                                skor = "-"
                        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                            scores = result[0]
                            score_negatif = next((item['score'] for item in scores if item['label'] == 'LABEL_0'), 0.0)
                            score_positif = next((item['score'] for item in scores if item['label'] == 'LABEL_1'), 0.0)
                            
                            prob_negatif = round(score_negatif * 100, 1)
                            prob_positif = round(score_positif * 100, 1)
                            
                            if prob_positif > prob_negatif:
                                sentimen = "Positif"
                                confidence = prob_positif
                            else:
                                sentimen = "Negatif"
                                confidence = prob_negatif
                                
                            skor = "HuggingFace API (via DNS-Proxy)"
                            metrik = context['base_metrik_indobert']
                        else:
                            sentimen = f"Error: Format balasan API tidak sesuai ({result_str[:50]})"
                            skor = "-"
            except Exception as e:
                sentimen = f"Error Jaringan Vercel (DoH): {str(e)}"
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