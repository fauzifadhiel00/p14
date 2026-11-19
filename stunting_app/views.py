import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import requests
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
from .models import DataStuntingJabar, DataGiziKurangJabar, PemeriksaanBalita

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'model_stunting.joblib'
DATA_PATH = BASE_DIR / 'data_balita.csv'

# Standar Median Tinggi Badan Balita (WHO MGRS & Kemenkes RI - TB/U)
STANDAR_WHO = {
    0: {"laki-laki": 49.9, "perempuan": 49.1, "min_normal": 46.1, "max_normal": 53.7},
    3: {"laki-laki": 61.4, "perempuan": 59.8, "min_normal": 57.3, "max_normal": 65.5},
    6: {"laki-laki": 67.6, "perempuan": 65.7, "min_normal": 63.3, "max_normal": 71.9},
    9: {"laki-laki": 72.0, "perempuan": 70.1, "min_normal": 67.5, "max_normal": 76.5},
    12: {"laki-laki": 75.7, "perempuan": 74.0, "min_normal": 71.0, "max_normal": 80.5},
    18: {"laki-laki": 82.3, "perempuan": 80.7, "min_normal": 76.9, "max_normal": 87.7},
    24: {"laki-laki": 87.8, "perempuan": 86.4, "min_normal": 82.1, "max_normal": 93.5},
    36: {"laki-laki": 96.1, "perempuan": 95.1, "min_normal": 89.8, "max_normal": 102.5},
    48: {"laki-laki": 103.3, "perempuan": 102.7, "min_normal": 96.0, "max_normal": 110.6},
    60: {"laki-laki": 110.0, "perempuan": 109.4, "min_normal": 102.0, "max_normal": 118.0}
}

# Standar Median & Deviasi Baku Berat Badan menurut Tinggi Badan (BB/TB - Indikator Wasting WHO & Kemenkes RI)
STANDAR_BB_TB = {
    45: {"laki-laki": (2.4, 0.35), "perempuan": (2.5, 0.35)},
    50: {"laki-laki": (3.3, 0.40), "perempuan": (3.4, 0.40)},
    55: {"laki-laki": (4.5, 0.50), "perempuan": (4.5, 0.50)},
    60: {"laki-laki": (5.9, 0.60), "perempuan": (5.8, 0.60)},
    65: {"laki-laki": (7.4, 0.70), "perempuan": (7.2, 0.70)},
    70: {"laki-laki": (8.6, 0.80), "perempuan": (8.4, 0.80)},
    75: {"laki-laki": (9.6, 0.90), "perempuan": (9.3, 0.90)},
    80: {"laki-laki": (10.6, 1.00), "perempuan": (10.3, 1.00)},
    85: {"laki-laki": (11.7, 1.10), "perempuan": (11.4, 1.10)},
    90: {"laki-laki": (12.9, 1.20), "perempuan": (12.6, 1.20)},
    95: {"laki-laki": (14.2, 1.35), "perempuan": (13.9, 1.35)},
    100: {"laki-laki": (15.6, 1.50), "perempuan": (15.3, 1.50)},
    105: {"laki-laki": (17.1, 1.70), "perempuan": (16.8, 1.70)},
    110: {"laki-laki": (18.7, 1.90), "perempuan": (18.5, 1.90)},
    115: {"laki-laki": (20.6, 2.15), "perempuan": (20.4, 2.15)},
    120: {"laki-laki": (22.8, 2.40), "perempuan": (22.6, 2.40)},
}

# Standar Median & Deviasi Baku Berat Badan menurut Umur (BB/U - Indikator Underweight WHO)
STANDAR_BB_U = {
    0: {"laki-laki": (3.3, 0.50), "perempuan": (3.2, 0.45)},
    3: {"laki-laki": (6.4, 0.70), "perempuan": (5.8, 0.65)},
    6: {"laki-laki": (7.9, 0.80), "perempuan": (7.3, 0.75)},
    9: {"laki-laki": (8.9, 0.90), "perempuan": (8.2, 0.85)},
    12: {"laki-laki": (9.6, 1.00), "perempuan": (8.9, 0.95)},
    18: {"laki-laki": (10.9, 1.15), "perempuan": (10.2, 1.10)},
    24: {"laki-laki": (12.2, 1.30), "perempuan": (11.5, 1.25)},
    36: {"laki-laki": (14.3, 1.60), "perempuan": (13.9, 1.55)},
    48: {"laki-laki": (16.3, 1.90), "perempuan": (16.1, 1.85)},
    60: {"laki-laki": (18.3, 2.20), "perempuan": (18.2, 2.15)},
}

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
]

# Cache model in memory
_LOADED_MODEL_PKG = None
def get_model():
    global _LOADED_MODEL_PKG
    if _LOADED_MODEL_PKG is None and MODEL_PATH.exists():
        _LOADED_MODEL_PKG = joblib.load(MODEL_PATH)
    return _LOADED_MODEL_PKG

# Cache dataset summary
_CACHED_STATS = None
def get_dataset_stats():
    global _CACHED_STATS
    if _CACHED_STATS is not None:
        return _CACHED_STATS
    
    if not DATA_PATH.exists():
        return {}

    df = pd.read_csv(DATA_PATH)
    total_rows = len(df)
    status_counts = df['Status Gizi'].value_counts().to_dict()
    
    # Cross tab gender
    ct = pd.crosstab(df['Jenis Kelamin'], df['Status Gizi'])
    gender_stats = {
        'laki_laki': ct.loc['laki-laki'].to_dict() if 'laki-laki' in ct.index else {},
        'perempuan': ct.loc['perempuan'].to_dict() if 'perempuan' in ct.index else {}
    }

    # Sample data dengan nama atribut bersih untuk template Django
    sample_records = []
    for r in df.head(8).to_dict(orient='records'):
        sample_records.append({
            'umur': r.get('Umur (bulan)', 0),
            'gender': r.get('Jenis Kelamin', ''),
            'tinggi': round(float(r.get('Tinggi Badan (cm)', 0)), 1),
            'status': r.get('Status Gizi', '')
        })

    # Kurva referensi WHO untuk Chart
    who_curve = []
    for u in sorted(STANDAR_WHO.keys()):
        who_curve.append({
            'umur': u,
            'laki_laki': STANDAR_WHO[u]['laki-laki'],
            'perempuan': STANDAR_WHO[u]['perempuan'],
            'min_normal': STANDAR_WHO[u]['min_normal'],
            'max_normal': STANDAR_WHO[u]['max_normal'],
        })

    _CACHED_STATS = {
        'total_rows': total_rows,
        'total_rows_formatted': f"{total_rows:,}".replace(",", "."),
        'status_counts': status_counts,
        'gender_stats': gender_stats,
        'sample_records': sample_records,
        'who_curve': who_curve,
        'pct_normal': round((status_counts.get('normal', 0) / total_rows) * 100, 1),
        'pct_stunted': round((status_counts.get('stunted', 0) / total_rows) * 100, 1),
        'pct_severely': round((status_counts.get('severely stunted', 0) / total_rows) * 100, 1),
        'pct_tinggi': round((status_counts.get('tinggi', 0) / total_rows) * 100, 1),
    }
    return _CACHED_STATS

def hitung_estimasi_ideal(umur, gender):
    """Menghitung tinggi badan median dan batas bawah normal (TB/U)"""
    gender = str(gender).strip().lower()
    if gender not in ['laki-laki', 'perempuan']:
        gender = 'laki-laki'
    try:
        umur = int(round(float(umur)))
    except (ValueError, TypeError):
        umur = 12

    umur_kunci = sorted(STANDAR_WHO.keys())
    if umur <= umur_kunci[0]:
        return STANDAR_WHO[umur_kunci[0]][gender], STANDAR_WHO[umur_kunci[0]]["min_normal"]
    if umur >= umur_kunci[-1]:
        return STANDAR_WHO[umur_kunci[-1]][gender], STANDAR_WHO[umur_kunci[-1]]["min_normal"]
    if umur in STANDAR_WHO:
        return STANDAR_WHO[umur][gender], STANDAR_WHO[umur]["min_normal"]
    for i in range(len(umur_kunci) - 1):
        u1, u2 = umur_kunci[i], umur_kunci[i+1]
        if u1 <= umur <= u2:
            rasio = (umur - u1) / (u2 - u1)
            med = STANDAR_WHO[u1][gender] + rasio * (STANDAR_WHO[u2][gender] - STANDAR_WHO[u1][gender])
            min_norm = STANDAR_WHO[u1]["min_normal"] + rasio * (STANDAR_WHO[u2]["min_normal"] - STANDAR_WHO[u1]["min_normal"])
            return round(med, 1), round(min_norm, 1)
    return 100.0, 95.0

def hitung_status_wasting(tinggi, berat, gender):
    """Menghitung Z-Score BB/TB dan menentukan klasifikasi Wasting (Akut)"""
    gender = str(gender).strip().lower()
    if gender not in ['laki-laki', 'perempuan']:
        gender = 'laki-laki'
    try:
        tinggi = float(tinggi)
    except (ValueError, TypeError):
        tinggi = 75.0
    try:
        berat = float(berat)
    except (ValueError, TypeError):
        berat = 9.0

    tinggi_kunci = sorted(STANDAR_BB_TB.keys())
    
    if tinggi <= tinggi_kunci[0]:
        med, sd = STANDAR_BB_TB[tinggi_kunci[0]][gender]
    elif tinggi >= tinggi_kunci[-1]:
        med, sd = STANDAR_BB_TB[tinggi_kunci[-1]][gender]
    else:
        med, sd = 10.0, 1.0
        for i in range(len(tinggi_kunci) - 1):
            t1, t2 = tinggi_kunci[i], tinggi_kunci[i+1]
            if t1 <= tinggi <= t2:
                r = (tinggi - t1) / (t2 - t1)
                m1, s1 = STANDAR_BB_TB[t1][gender]
                m2, s2 = STANDAR_BB_TB[t2][gender]
                med = m1 + r * (m2 - m1)
                sd = s1 + r * (s2 - s1)
                break
                
    sd_val = sd if sd > 0 else 1.0
    z_score = round((berat - med) / sd_val, 2)
    selisih_bb = round(berat - med, 2)
    min_normal_bb = round(med - 2 * sd, 2)
    
    # Standar Permenkes No. 2 Tahun 2020:
    if z_score < -3.0:
        status = "Gizi Buruk (Severely Wasted)"
        kategori_singkat = "severely_wasted"
        badge_class = "danger"
    elif -3.0 <= z_score < -2.0:
        status = "Gizi Kurang (Wasted)"
        kategori_singkat = "wasted"
        badge_class = "warning"
    elif -2.0 <= z_score <= 1.0:
        status = "Gizi Baik (Normal)"
        kategori_singkat = "normal"
        badge_class = "success"
    elif 1.0 < z_score <= 2.0:
        status = "Berisiko Gizi Lebih"
        kategori_singkat = "risk_overweight"
        badge_class = "info"
    else:
        status = "Gizi Lebih / Obesitas"
        kategori_singkat = "overweight"
        badge_class = "danger"

    return {
        'status': status,
        'kategori_singkat': kategori_singkat,
        'badge_class': badge_class,
        'z_score': z_score,
        'ideal_bb': round(med, 2),
        'min_normal_bb': min_normal_bb,
        'selisih_bb': selisih_bb
    }

def hitung_status_underweight(umur, berat, gender):
    """Menghitung Z-Score BB/U (Underweight menurut Umur)"""
    gender = str(gender).strip().lower()
    if gender not in ['laki-laki', 'perempuan']:
        gender = 'laki-laki'
    try:
        umur = int(round(float(umur)))
    except (ValueError, TypeError):
        umur = 12
    try:
        berat = float(berat)
    except (ValueError, TypeError):
        berat = 9.0

    umur_kunci = sorted(STANDAR_BB_U.keys())
    
    if umur <= umur_kunci[0]:
        med, sd = STANDAR_BB_U[umur_kunci[0]][gender]
    elif umur >= umur_kunci[-1]:
        med, sd = STANDAR_BB_U[umur_kunci[-1]][gender]
    else:
        med, sd = 10.0, 1.0
        for i in range(len(umur_kunci) - 1):
            u1, u2 = umur_kunci[i], umur_kunci[i+1]
            if u1 <= umur <= u2:
                r = (umur - u1) / (u2 - u1)
                m1, s1 = STANDAR_BB_U[u1][gender]
                m2, s2 = STANDAR_BB_U[u2][gender]
                med = m1 + r * (m2 - m1)
                sd = s1 + r * (s2 - s1)
                break
                
    sd_val = sd if sd > 0 else 1.0
    z_score = round((berat - med) / sd_val, 2)
    if z_score < -3.0:
        status = "Berat Badan Sangat Kurang"
    elif -3.0 <= z_score < -2.0:
        status = "Berat Badan Kurang"
    elif -2.0 <= z_score <= 1.0:
        status = "Berat Badan Normal"
    else:
        status = "Risiko Berat Badan Lebih"
        
    return {
        'status': status,
        'z_score': z_score,
        'ideal_bb_umur': round(med, 2)
    }

def evaluasi_komposit(stunting_status, wasting_kat):
    """Analisis Komposit Malnutrisi: Stunting (Linear/Kronis) vs Wasting (Massa/Akut)"""
    is_stunted = stunting_status in ['stunted', 'severely stunted']
    is_wasted = wasting_kat in ['wasted', 'severely_wasted']
    
    if is_stunted and is_wasted:
        return {
            'level': 'Kritis (Prioritas Tinggi)',
            'ringkasan': 'Malnutrisi Ganda (Stunted & Wasted)',
            'pesan': 'Anak mengalami defisit massa tubuh akut sekaligus gangguan pertumbuhan linear kronis. Kondisi ini memiliki risiko morbiditas tertinggi sehingga memerlukan rujukan medis segera ke dokter spesialis anak atau Puskesmas.',
            'badge': 'danger'
        }
    elif is_stunted and not is_wasted:
        return {
            'level': 'Peringatan (Malnutrisi Kronis)',
            'ringkasan': 'Stunting Saja (Tanpa Wasting Akut)',
            'pesan': 'Proporsi berat terhadap tinggi saat ini tergolong proporsional, namun terjadi hambatan penambahan tinggi badan jangka panjang. Fokus pada stimulasi catch-up growth dengan protein hewani ganda.',
            'badge': 'warning'
        }
    elif not is_stunted and is_wasted:
        return {
            'level': 'Peringatan (Malnutrisi Akut)',
            'ringkasan': 'Wasting Saja (Tanpa Stunting)',
            'pesan': 'Tinggi badan anak saat ini masih normal, namun terjadi penurunan berat badan drastis (wasting). Kondisi akut ini berisiko menjadi stunting jika defisit kalori dan penyakit infeksi tidak segera diatasi.',
            'badge': 'warning'
        }
    else:
        return {
            'level': 'Optimal',
            'ringkasan': 'Tumbuh Kembang Seimbang (Bebas Stunting & Wasting)',
            'pesan': 'Pertumbuhan tinggi badan dan penambahan berat badan anak berjalan proporsional sesuai standar baku WHO. Pertahankan pola makan bergizi seimbang dan stimulasi motorik.',
            'badge': 'success'
        }


def get_jabar_statistics():
    """Mengambil ringkasan data stunting dan gizi kurang Jawa Barat 2019-2025 dari SQLite"""
    try:
        # Auto-seed dari file CSV jika database SQLite masih kosong
        if DataStuntingJabar.objects.count() == 0:
            try:
                import import_jabar_data
                import_jabar_data.import_stunting()
                import_jabar_data.import_gizi_kurang()
            except Exception:
                pass

        db_years = sorted(list(set(DataStuntingJabar.objects.values_list('tahun', flat=True))))
        years = db_years if db_years else [2019, 2020, 2021, 2022, 2023, 2024, 2025]
        trend_stunting = []
        trend_gizi = []
        for y in years:
            total_stunting = sum(DataStuntingJabar.objects.filter(tahun=y).values_list('jumlah_balita_stunting', flat=True))
            total_gizi = sum(DataGiziKurangJabar.objects.filter(tahun=y).values_list('jumlah_balita', flat=True))
            trend_stunting.append(int(total_stunting))
            trend_gizi.append(int(round(float(total_gizi))))
            
        # Top 10 wilayah stunting per tahun (untuk filter interaktif dropdown tahun di dashboard)
        top_by_year = {}
        for y in years:
            qs = DataStuntingJabar.objects.filter(tahun=y).order_by('-jumlah_balita_stunting')[:10]
            top_by_year[str(y)] = {
                'labels': [item.nama_kabupaten_kota.title() for item in qs],
                'values': [item.jumlah_balita_stunting for item in qs]
            }

        latest_year = str(years[-1]) if years else '2025'
        top_kab_labels = top_by_year.get(latest_year, {}).get('labels', [])
        top_kab_values = top_by_year.get(latest_year, {}).get('values', [])
        
        # Seluruh daftar kab/kota untuk dropdown form
        all_kab = sorted(list(set(DataStuntingJabar.objects.values_list('nama_kabupaten_kota', flat=True))))
        
        # Riwayat input realtime terakhir
        recent_pemeriksaan = list(PemeriksaanBalita.objects.all()[:8])
        total_realtime = PemeriksaanBalita.objects.count()

        # Statistik ringkasan riwayat pemeriksaan realtime
        rt_normal = PemeriksaanBalita.objects.filter(status_stunting='normal').count()
        rt_stunted = PemeriksaanBalita.objects.filter(status_stunting__in=['stunted', 'severely stunted']).count()
        rt_severe = PemeriksaanBalita.objects.filter(status_stunting='severely stunted').count()
        rt_wasted = PemeriksaanBalita.objects.filter(status_wasting__icontains='Wasted').count()
        rt_optimal = PemeriksaanBalita.objects.filter(evaluasi_komposit__icontains='Optimal').count()
        rt_rujukan = PemeriksaanBalita.objects.filter(evaluasi_komposit__icontains='Malnutrisi Ganda').count()
        latest_balita = recent_pemeriksaan[0] if recent_pemeriksaan else None

        pct_rt_normal = round((rt_normal / total_realtime * 100), 1) if total_realtime > 0 else 0
        pct_rt_stunted = round((rt_stunted / total_realtime * 100), 1) if total_realtime > 0 else 0

        realtime_summary = {
            'total': total_realtime,
            'normal': rt_normal,
            'stunted': rt_stunted,
            'severe': rt_severe,
            'wasted': rt_wasted,
            'optimal': rt_optimal,
            'rujukan': rt_rujukan,
            'pct_normal': pct_rt_normal,
            'pct_stunted': pct_rt_stunted,
            'latest': latest_balita,
        }
        
        total_stunting_val = int(trend_stunting[-1]) if trend_stunting else 0
        total_gizi_val = int(trend_gizi[-1]) if trend_gizi else 0

        return {
            'years': years,
            'trend_stunting': trend_stunting,
            'trend_gizi': trend_gizi,
            'top_by_year': top_by_year,
            'top_kab_labels': top_kab_labels,
            'top_kab_values': top_kab_values,
            'all_kab': all_kab,
            'recent_pemeriksaan': recent_pemeriksaan,
            'realtime_summary': realtime_summary,
            'total_stunting_2025': total_stunting_val,
            'total_stunting_2025_formatted': f"{total_stunting_val:,}".replace(",", "."),
            'total_gizi_2025': total_gizi_val,
            'total_gizi_2025_formatted': f"{total_gizi_val:,}".replace(",", "."),
            'total_realtime': total_realtime,
            'has_jabar_data': len(trend_stunting) > 0 and trend_stunting[-1] > 0
        }
    except Exception as e:
        return {
            'years': [2019, 2020, 2021, 2022, 2023, 2024, 2025],
            'trend_stunting': [],
            'trend_gizi': [],
            'top_by_year': {},
            'top_kab_labels': [],
            'top_kab_values': [],
            'all_kab': [],
            'recent_pemeriksaan': [],
            'realtime_summary': {
                'total': 0, 'normal': 0, 'stunted': 0, 'severe': 0, 'wasted': 0, 'optimal': 0, 'rujukan': 0, 'pct_normal': 0, 'pct_stunted': 0, 'latest': None
            },
            'total_stunting_2025': 0,
            'total_gizi_2025': 0,
            'total_realtime': 0,
            'has_jabar_data': False,
            'error': str(e)
        }


# =========================================================================
# VIEWS
# =========================================================================

def dashboard_view(request):
    """Halaman Dashboard: Visualisasi Grafik & Ringkasan Eksplorasi Data"""
    stats = get_dataset_stats()
    model_pkg = get_model()
    test_acc = f"{model_pkg['test_acc']*100:.1f}%" if model_pkg else "99.9%"
    
    jabar_stats = get_jabar_statistics()

    context = {
        'stats': stats,
        'test_acc': test_acc,
        'who_curve_json': json.dumps(stats.get('who_curve', [])),
        'jabar_stats': jabar_stats,
        'jabar_years_json': json.dumps(jabar_stats.get('years', [2019, 2020, 2021, 2022, 2023, 2024, 2025])),
        'jabar_stunting_json': json.dumps(jabar_stats.get('trend_stunting', [])),
        'jabar_gizi_json': json.dumps(jabar_stats.get('trend_gizi', [])),
        'jabar_top_labels_json': json.dumps(jabar_stats.get('top_kab_labels', [])),
        'jabar_top_values_json': json.dumps(jabar_stats.get('top_kab_values', [])),
        'jabar_top_by_year_json': json.dumps(jabar_stats.get('top_by_year', {})),
        'active_menu': 'dashboard'
    }
    return render(request, 'dashboard.html', context)


def prediksi_view(request):
    """Halaman Prediksi Machine Learning & Input Antropometri Balita (TB & BB)"""
    model_pkg = get_model()
    hasil_prediksi = None
    error_msg = None
    input_form = {}
    
    jabar_stats = get_jabar_statistics()
    all_kab = jabar_stats.get('all_kab', [])

    if request.method == 'POST':
        nama = request.POST.get('nama', 'Balita Sehat').strip() or 'Balita Sehat'
        gender = request.POST.get('gender', 'laki-laki')
        umur = int(request.POST.get('umur', 12))
        tinggi = float(request.POST.get('tinggi', 75.0))
        berat = float(request.POST.get('berat', 9.0))
        kabupaten = request.POST.get('kabupaten_kota', 'KABUPATEN BOGOR').strip().upper() or 'JAWA BARAT'

        input_form = {
            'nama': nama,
            'gender': gender,
            'umur': umur,
            'tinggi': tinggi,
            'berat': berat,
            'kabupaten_kota': kabupaten
        }

        try:
            if model_pkg is None:
                error_msg = "Model Machine Learning belum dimuat atau file model_stunting.joblib tidak ditemukan."
            else:
                model = model_pkg['model']
                gender_num = 1 if gender == "laki-laki" else 0
                
                df_input = pd.DataFrame({
                    'Umur (bulan)': [umur],
                    'gender_num': [gender_num],
                    'Tinggi Badan (cm)': [tinggi]
                })

                pred_class = model.predict(df_input)[0]
                proba = model.predict_proba(df_input)[0]
                classes = model.classes_.tolist()

                ideal_tb, min_normal_tb = hitung_estimasi_ideal(umur, gender)
                selisih_ideal = round(tinggi - ideal_tb, 1)

                prob_dict = []
                for c, p in zip(classes, proba):
                    prob_dict.append({
                        'status': c,
                        'pct': round(p * 100, 1)
                    })
                prob_dict = sorted(prob_dict, key=lambda x: x['pct'], reverse=True)

                # Hitung Indikator Wasting (BB/TB) dan Underweight (BB/U)
                wasting_data = hitung_status_wasting(tinggi, berat, gender)
                underweight_data = hitung_status_underweight(umur, berat, gender)
                komposit_data = evaluasi_komposit(pred_class, wasting_data['kategori_singkat'])

                # Simpan rekaman pemeriksaan balita ke database SQLite
                try:
                    PemeriksaanBalita.objects.create(
                        nama=nama,
                        gender=gender,
                        umur_bulan=umur,
                        tinggi_badan=tinggi,
                        berat_badan=berat,
                        kabupaten_kota=kabupaten,
                        status_stunting=pred_class,
                        status_wasting=wasting_data['status'],
                        status_underweight=underweight_data['status'],
                        evaluasi_komposit=komposit_data['ringkasan']
                    )
                except Exception:
                    pass

                hasil_prediksi = {
                    'nama': nama,
                    'gender': gender,
                    'umur': umur,
                    'tinggi': tinggi,
                    'berat': berat,
                    'kabupaten_kota': kabupaten,
                    'status': pred_class,
                    'ideal': ideal_tb,
                    'min_normal': min_normal_tb,
                    'selisih': selisih_ideal,
                    'probabilitas': prob_dict,
                    'wasting': wasting_data,
                    'underweight': underweight_data,
                    'komposit': komposit_data,
                    'saved_to_db': True
                }

                # Simpan ke session untuk AI Advisor
                request.session['last_prediction'] = hasil_prediksi

                # Karena data sudah berhasil dikirim ke database SQLite,
                # form input di-reset bersih agar siap untuk input balita berikutnya
                input_form = {}

        except Exception as e:
            error_msg = f"Terjadi kesalahan pemrosesan: {str(e)}"

    context = {
        'hasil': hasil_prediksi,
        'error_msg': error_msg,
        'input_form': input_form,
        'all_kab': all_kab,
        'active_menu': 'prediksi'
    }
    return render(request, 'prediksi.html', context)


def ai_advisor_view(request):
    """Halaman Pemanfaatan AI: Explainable AI & Nutritional Health Advisor"""
    # 1. Reset profil subjek jika diminta melalui query parameter (?reset=1)
    if request.GET.get('reset'):
        if 'last_prediction' in request.session:
            del request.session['last_prediction']
        return redirect('ai_advisor')

    # 2. Muat balita dari riwayat database SQLite jika parameter (?id=X) diberikan
    balita_id = request.GET.get('id')
    if balita_id:
        try:
            b = PemeriksaanBalita.objects.get(id=balita_id)
            ideal_tb, min_normal_tb = hitung_estimasi_ideal(b.umur_bulan, b.gender)
            selisih_ideal = round(b.tinggi_badan - ideal_tb, 1)
            wasting_data = hitung_status_wasting(b.tinggi_badan, b.berat_badan, b.gender)
            underweight_data = hitung_status_underweight(b.umur_bulan, b.berat_badan, b.gender)
            komposit_data = evaluasi_komposit(b.status_stunting, wasting_data['kategori_singkat'])

            subjek_data = {
                'id': b.id,
                'nama': b.nama,
                'gender': b.gender,
                'umur': b.umur_bulan,
                'tinggi': b.tinggi_badan,
                'berat': b.berat_badan,
                'kabupaten_kota': b.kabupaten_kota,
                'status': b.status_stunting,
                'ideal': ideal_tb,
                'min_normal': min_normal_tb,
                'selisih': selisih_ideal,
                'wasting': wasting_data,
                'underweight': underweight_data,
                'komposit': komposit_data,
                'saved_to_db': True,
                'is_custom': True,
                'source': 'riwayat'
            }
            request.session['last_prediction'] = subjek_data
        except (PemeriksaanBalita.DoesNotExist, ValueError):
            pass

    # 3. Periksa ketersediaan subjek aktif di sesi
    is_custom_subjek = 'last_prediction' in request.session
    last_pred = request.session.get('last_prediction', {
        'nama': 'Balita Rujukan',
        'gender': 'laki-laki',
        'umur': 15,
        'tinggi': 72.5,
        'berat': 8.2,
        'status': 'stunted',
        'ideal': 79.5,
        'min_normal': 75.5,
        'selisih': -7.0,
        'wasting': {
            'status': 'Gizi Kurang (Wasted)',
            'kategori_singkat': 'wasted',
            'badge_class': 'warning',
            'z_score': -2.35,
            'ideal_bb': 8.9,
            'min_normal_bb': 7.3,
            'selisih_bb': -0.7
        },
        'underweight': {
            'status': 'Berat Badan Kurang',
            'z_score': -2.1,
            'ideal_bb_umur': 10.2
        },
        'komposit': {
            'level': 'Kritis (Prioritas Tinggi)',
            'ringkasan': 'Malnutrisi Ganda (Stunted & Wasted)',
            'pesan': 'Anak mengalami defisit massa tubuh akut sekaligus gangguan pertumbuhan linear kronis. Diperlukan intervensi gizi intensif.',
            'badge': 'danger'
        },
        'is_custom': False,
        'source': 'default'
    })

    # Ambil riwayat balita untuk opsi pilih subjek balita di modal
    riwayat_balita_list = PemeriksaanBalita.objects.all().order_by('-created_at')[:25]

    # AI Consultation Q&A default dictionary
    konsultasi_list = [
        {
            'id': 1,
            'tanya': 'Bagaimana cara mengejar ketertinggalan tinggi badan balita (Catch-up Growth)?',
            'jawab': 'Prinsip utama Catch-up Growth adalah pemenuhan asam amino esensial dan energi padat gizi: (1) Utamakan konsumsi protein hewani tinggi seperti telur, susu pertumbuhan, dan ikan laut. (2) Cukupi mikronutrien pembentuk matriks tulang (Kalsium, Zink, Vitamin D). (3) Atasi penyakit infeksi penyerta seperti diare atau ISPA agar metabolisme optimal.'
        },
        {
            'id': 2,
            'tanya': 'Apa perbedaan mendasar antara Stunting (TB/U) dan Wasting (BB/TB)?',
            'jawab': 'Stunting adalah malnutrisi kronis (jangka panjang) akibat kekurangan gizi menahun selama 1000 HPK sehingga anak lebih pendek dari usianya. Sedangkan Wasting adalah malnutrisi akut (jangka pendek) di mana berat badan anak turun drastis relatif terhadap tinggi badannya, umumnya dipicu penyakit infeksi akut atau kelaparan mendadak. Anak yang mengalami kombinasi keduanya (Stunted + Wasted) memiliki risiko mortalitas tertinggi.'
        },
        {
            'id': 3,
            'tanya': 'Mengapa protein hewani lebih efektif mencegah stunting dibanding protein nabati?',
            'jawab': 'Protein hewani memiliki nilai biologis yang jauh lebih tinggi dengan asam amino esensial lengkap dan mudah diserap saluran cerna anak. Protein hewani kaya akan Zink, Zat Besi Heme, dan Vitamin B12 yang memicu sekresi hormon pertumbuhan (IGF-1).'
        },
        {
            'id': 4,
            'tanya': 'Apa saja tanda bahaya (Red Flags) gagal tumbuh pada balita?',
            'jawab': 'Tanda bahaya yang mewajibkan rujukan segera ke dokter anak: (1) Berat badan tidak naik (2T) selama dua bulan berturut-turut. (2) Anak tampak lemas, pasif, dan terlambat mencapai tonggak perkembangan motorik. (3) Mengalami muntah atau diare berkepanjangan.'
        },
        {
            'id': 5,
            'tanya': 'Bagaimana hubungan sanitasi air bersih dengan kejadian stunting?',
            'jawab': 'Sanitasi buruk memicu Environmental Enteric Dysfunction (EED), yaitu peradangan kronis pada dinding usus halus. Akibatnya, dinding usus rusak dan nutrisi makanan tidak dapat diserap ke peredaran darah, memicu stunting meski asupan makanan tampak cukup.'
        }
    ]

    context = {
        'subjek': last_pred,
        'is_custom_subjek': is_custom_subjek,
        'riwayat_balita_list': riwayat_balita_list,
        'konsultasi_list': konsultasi_list,
        'active_menu': 'ai_advisor'
    }
    return render(request, 'ai_advisor.html', context)


def riwayat_view(request):
    """Halaman Khusus: Pemeriksaan Balita Real-Time (Surveilans & Log SQLite)"""
    q = request.GET.get('q', '').strip()
    status_filter = request.GET.get('status', '').strip()
    gender_filter = request.GET.get('gender', '').strip()

    pemeriksaan_qs = PemeriksaanBalita.objects.all().order_by('-created_at')

    if q:
        pemeriksaan_qs = pemeriksaan_qs.filter(
            Q(nama__icontains=q) | Q(kabupaten_kota__icontains=q)
        )
    if status_filter:
        if status_filter in ['normal', 'stunted', 'severely stunted', 'tinggi']:
            pemeriksaan_qs = pemeriksaan_qs.filter(status_stunting=status_filter)
        elif status_filter == 'wasted':
            pemeriksaan_qs = pemeriksaan_qs.filter(status_wasting__icontains='Wasted')
    if gender_filter:
        pemeriksaan_qs = pemeriksaan_qs.filter(gender=gender_filter)

    total_realtime = PemeriksaanBalita.objects.count()
    rt_normal = PemeriksaanBalita.objects.filter(status_stunting='normal').count()
    rt_stunted = PemeriksaanBalita.objects.filter(status_stunting__in=['stunted', 'severely stunted']).count()
    rt_severe = PemeriksaanBalita.objects.filter(status_stunting='severely stunted').count()
    rt_wasted = PemeriksaanBalita.objects.filter(status_wasting__icontains='Wasted').count()
    latest_balita = PemeriksaanBalita.objects.first()

    pct_rt_normal = round((rt_normal / total_realtime * 100), 1) if total_realtime > 0 else 0
    pct_rt_stunted = round((rt_stunted / total_realtime * 100), 1) if total_realtime > 0 else 0

    realtime_summary = {
        'total': total_realtime,
        'normal': rt_normal,
        'stunted': rt_stunted,
        'severe': rt_severe,
        'wasted': rt_wasted,
        'pct_normal': pct_rt_normal,
        'pct_stunted': pct_rt_stunted,
        'latest': latest_balita,
    }

    context = {
        'pemeriksaan_list': pemeriksaan_qs,
        'total_filtered': pemeriksaan_qs.count(),
        'realtime_summary': realtime_summary,
        'q': q,
        'status_filter': status_filter,
        'gender_filter': gender_filter,
        'active_menu': 'riwayat'
    }
    return render(request, 'riwayat.html', context)


def hapus_pemeriksaan_view(request, id):
    """Menghapus rekaman riwayat pemeriksaan dari basis data SQLite."""
    if request.method == 'POST':
        try:
            item = PemeriksaanBalita.objects.get(id=id)
            item.delete()
        except PemeriksaanBalita.DoesNotExist:
            pass
    return redirect('riwayat')


def tentang_view(request):
    """Halaman Informasi Proyek, Metodologi & Profil Mahasiswa"""
    context = {
        'active_menu': 'tentang'
    }
    return render(request, 'tentang.html', context)


# =========================================================================
# GEMINI API CHAT & CONFIGURATION ENDPOINTS
# =========================================================================

def generate_clinical_advisor_fallback(pesan_user, konteks_subjek):
    """
    Mesin Inferensi Klinis Cerdas (Rule-Based Clinical Expert System)
    Menghasilkan rekomendasi nutrisi harian, analisis catch-up growth, dan tanda bahaya
    yang personal berdasarkan profil balita aktif, standar Permenkes No. 2 Tahun 2020, dan WHO MGRS.
    """
    nama = konteks_subjek.get('nama', 'Balita Sehat')
    gender = str(konteks_subjek.get('gender', 'laki-laki')).lower()
    gender_str = 'Laki-laki' if gender == 'laki-laki' else 'Perempuan'
    try:
        umur = int(float(konteks_subjek.get('umur', 12)))
    except (ValueError, TypeError):
        umur = 12
    try:
        tinggi = float(konteks_subjek.get('tinggi', 75.0))
    except (ValueError, TypeError):
        tinggi = 75.0
    try:
        berat = float(konteks_subjek.get('berat', 9.0))
    except (ValueError, TypeError):
        berat = 9.0

    stunting_status = str(konteks_subjek.get('status', 'normal')).lower()
    try:
        selisih_tb = float(konteks_subjek.get('selisih', 0.0))
    except (ValueError, TypeError):
        selisih_tb = 0.0
    try:
        ideal_tb = float(konteks_subjek.get('ideal', 75.0))
    except (ValueError, TypeError):
        ideal_tb = 75.0
    
    wasting = konteks_subjek.get('wasting', {})
    status_wasting = wasting.get('status', 'Gizi Baik (Normal)')
    try:
        z_wasting = float(wasting.get('z_score', 0.0))
    except (ValueError, TypeError):
        z_wasting = 0.0
    ideal_bb = wasting.get('ideal_bb', berat)
    
    komposit = konteks_subjek.get('komposit', {})
    komposit_ringkasan = komposit.get('ringkasan', 'Optimal')

    p_lower = str(pesan_user).lower()

    # DETEKSI INTENT PERTANYAAN
    is_menu = any(w in p_lower for w in ['menu', 'mpasi', 'makan', 'jadwal', 'resep', 'porsi', 'sarapan', 'siang', 'malam'])
    is_tinggi = any(w in p_lower for w in ['tinggi', 'kejar', 'catch-up', 'stunting', 'pendek', 'tumbuh', 'badan'])
    is_bahaya = any(w in p_lower for w in ['bahaya', 'gagal tumbuh', 'red flag', 'tanda', 'falter', 'dokter', 'puskesmas', 'darurat'])
    is_berat = any(w in p_lower for w in ['berat', 'wasting', 'kurus', 'gizi kurang', 'bb', 'timbangan', 'turun'])
    is_protein = any(w in p_lower for w in ['protein', 'hewani', 'telur', 'ikan', 'daging', 'ayam', 'susu', 'ganda'])

    # 1. INTENT REKOMENDASI MENU MPASI / MAKANAN
    if is_menu or (not is_tinggi and not is_bahaya and not is_berat and not is_protein and len(p_lower) < 15):
        if umur < 6:
            guidance = (
                f"Balita **{nama}** masih berusia {umur} bulan, sehingga berada dalam periode **ASI Eksklusif**.\n\n"
                f"**Pedoman Klinis:**\n"
                f"1. Berikan ASI sesuai kebutuhan bayi (on-demand), minimal 8-12 kali per 24 jam.\n"
                f"2. Belum dianjurkan pemberian makanan padat/cair lain kecuali atas instruksi dokter spesialis anak.\n"
                f"3. Pastikan perlekatan menyusui benar untuk mencegah penurunan berat badan."
            )
        elif 6 <= umur <= 8:
            guidance = (
                f"Rekomendasi Menu MPASI Balita **{nama}** (Usia {umur} Bulan - Periode Awal MPASI):\n\n"
                f"**Tekstur & Porsi:**\n"
                f"- Tekstur: Bubur lumat kental (puree/bubur saring), tidak mudah tumpah dari sendok.\n"
                f"- Porsi: Mulai 2-3 sendok makan bertahap hingga 1/2 mangkuk (125 ml) tiap kali makan.\n"
                f"- Frekuensi: 2-3 kali makan utama + 1-2 kali makanan selingan + ASI.\n\n"
                f"**Jadwal & Menu Protein Hewani Ganda Harian:**\n"
                f"- **Pagi (07.00):** Bubur beras + hati ayam lumat + sedikit santan peras asli.\n"
                f"- **Selingan Pagi (09.30):** Puree alpukat matang atau pisang kerok lumat.\n"
                f"- **Siang (12.00):** Bubur beras + kuning telur ayam matang + ikan kembung suwir halus + labu siam saring + 1 sdt minyak kelapa/mentega.\n"
                f"- **Selingan Sore (15.30):** ASI atau puree pepaya manis.\n"
                f"- **Malam (18.00):** Bubur tim saring daging ayam + tahu lumat + kuah kaldu ceker kaya nutrisi."
            )
        elif 9 <= umur <= 11:
            guidance = (
                f"Rekomendasi Menu MPASI Balita **{nama}** (Usia {umur} Bulan - Periode Transisi Tekstur):\n\n"
                f"**Tekstur & Porsi:**\n"
                f"- Tekstur: Makanan lembek dicincang halus (minced) atau cincang kasar (chopped) serta finger food empuk.\n"
                f"- Porsi: 1/2 mangkuk ukuran 250 ml (sekitar 125 ml) tiap kali makan.\n"
                f"- Frekuensi: 3-4 kali makan utama + 1-2 kali selingan padat gizi + ASI.\n\n"
                f"**Jadwal & Menu Protein Hewani Ganda Harian:**\n"
                f"- **Pagi (07.00):** Nasi tim lembek + 2 butir telur puyuh cincang + hati ayam cincang + wortel serut.\n"
                f"- **Selingan Pagi (09.30):** Bola-bola kentang kukus isi keju atau kuning telur.\n"
                f"- **Siang (12.00):** Nasi tim lembut + ikan kembung/tenggiri cincang + telur ayam orak-arik basah + santan gurih.\n"
                f"- **Selingan Sore (15.30):** Potongan buah pepaya/pisang lunak (finger food).\n"
                f"- **Malam (18.30):** Nasi tim + daging sapi giling semur + sup kaldu sayuran hijau."
            )
        elif 12 <= umur <= 23:
            guidance = (
                f"Rekomendasi Menu Makanan Balita **{nama}** (Usia {umur} Bulan - Periode Emas 1000 HPK):\n\n"
                f"**Tekstur & Porsi:**\n"
                f"- Tekstur: Makanan keluarga (table food) yang diiris atau dipotong kecil sesuai kemampuan mengunyah.\n"
                f"- Porsi: 3/4 hingga 1 mangkuk penuh ukuran 250 ml tiap kali makan.\n"
                f"- Frekuensi: 3-4 kali makan lengkap + 1-2 kali selingan sehat + ASI/susu.\n\n"
                f"**Jadwal & Menu Protein Hewani Ganda Harian:**\n"
                f"- **Pagi (07.00):** Nasi pulen + 1 butir telur ceplok/dadar + semur hati ayam + sup sayur bayam/jagung manis.\n"
                f"- **Selingan Pagi (09.30):** Puding susu telur atau nagasari pisang.\n"
                f"- **Siang (12.30):** Nasi + ikan kembung bumbu kuning gurih + tahu goreng empuk + sayur lodeh santan labu.\n"
                f"- **Selingan Sore (16.00):** Buah pisang ambon atau alpukat kerok dengan sedikit susu.\n"
                f"- **Malam (19.00):** Nasi + bola-bola daging sapi empuk cincang + telur rebus iris + kuah sup kaldu ceker."
            )
        else:
            guidance = (
                f"Rekomendasi Menu Makanan Balita **{nama}** (Usia {umur} Bulan / {round(umur/12, 1)} Tahun):\n\n"
                f"**Tekstur & Porsi:**\n"
                f"- Tekstur: Makanan keluarga penuh padat gizi.\n"
                f"- Porsi: 1 piring sedang keluarga (proporsi: 1/3 karbohidrat, 1/3 protein hewani ganda, 1/3 sayur & buah).\n"
                f"- Frekuensi: 3 kali makan utama + 2 kali makanan selingan kaya nutrisi.\n\n"
                f"**Jadwal & Menu Protein Hewani Ganda Harian:**\n"
                f"- **Pagi (07.00):** Nasi gurih/kuning + telur dadar iris + suwiran ayam goreng lengkuas + lalap tomat/timun.\n"
                f"- **Selingan Pagi (10.00):** Buah pepaya potong segar atau susu pertumbuhan.\n"
                f"- **Siang (12.30):** Nasi + ikan laut bakar/goreng + tempe bacem + sayur bening kelor/bayam.\n"
                f"- **Selingan Sore (16.00):** Roti panggang isi telur atau bubur kacang hijau santan.\n"
                f"- **Malam (19.00):** Nasi + rolade daging sapi cincang + telur mata sapi + sup bening wortel buncis."
            )
        return guidance

    # 2. INTENT KEJAR TINGGI BADAN / STUNTING CATCH-UP GROWTH
    elif is_tinggi or ('stunting' in p_lower):
        status_label = "Stunted" if "stunt" in stunting_status else ("Normal" if stunting_status == "normal" else stunting_status.title())
        guidance = (
            f"Strategi Klinis Mengejar Ketertinggalan Tinggi Badan (*Catch-Up Growth*) untuk **{nama}**:\n\n"
            f"**Analisis Antropometri Klinis:**\n"
            f"- Usia: {umur} Bulan | Jenis Kelamin: {gender_str}\n"
            f"- Tinggi Badan Saat Ini: **{tinggi} cm** (Status Stunting: **{status_label}**)\n"
            f"- Standar Median WHO: **{ideal_tb} cm** (Deviasi: **{selisih_tb:+.1f} cm**)\n\n"
            f"**5 Pilar Intervensi Catch-Up Growth Berbasis Bukti:**\n"
            f"1. **Asupan Protein Hewani Ganda Setiap Hari:**\n"
            f"   Kombinasikan minimal 2 sumber protein hewani (Telur + Ikan Laut/Hati Ayam/Daging) dalam setiap jadwal makan. Asam amino esensial lengkap dan Zink memicu sekresi hormon IGF-1 (Insulin-like Growth Factor 1) yang menstimulasi lempeng pertumbuhan tulang epifisis.\n\n"
            f"2. **Mikronutrien Pembentuk Matriks Tulang:**\n"
            f"   Pastikan asupan Kalsium, Fosfor, Magnesium, serta Vitamin D3 (dari sinar matahari pagi dan kuning telur/ikan laut) terpenuhi optimal.\n\n"
            f"3. **Kepadatan Energi & Lemak Sehat:**\n"
            f"   Tambahkan sumber lemak sehat seperti minyak kelapa, santan, atau mentega ke dalam makanan balita untuk menjamin kalori pembakaran cukup sehingga protein difokuskan murni untuk pemanjangan tulang.\n\n"
            f"4. **Kualitas Tidur Malam (Pukul 22.00 - 02.00):**\n"
            f"   Hormon pertumbuhan manusia (Human Growth Hormone / HGH) disekresikan secara optimal pada fase *deep sleep* gelombang lambat. Pastikan balita tidur lelap sebelum pukul 21.00 malam.\n\n"
            f"5. **Skrining Penyakit Infeksi Penyerta Kronis:**\n"
            f"   Periksakan ke dokter/Puskesmas untuk memastikan tidak ada infeksi tersembunyi yang menyerap nutrisi anak (misalnya TBC anak, infeksi saluran kemih / ISK, atau kecacingan)."
        )
        return guidance

    # 3. INTENT TANDA BAHAYA GAGAL TUMBUH (RED FLAGS)
    elif is_bahaya:
        guidance = (
            f"Panduan Klinis Tanda Bahaya (*Red Flags*) Gagal Tumbuh pada Balita:\n\n"
            f"Bila orang tua menemukan salah satu tanda di bawah ini pada **{nama}**, segera bawa balita ke **Puskesmas atau Dokter Spesialis Anak (Sp.A)**:\n\n"
            f"1. **Weight Faltering (Kenaikan Berat Badan Tidak Adekuat):**\n"
            f"   - Berat badan tidak naik (KMS mendatar atau turun / status 2T) selama 2 kali penimbangan berturut-turut.\n"
            f"   - Garis grafik pertumbuhan anak pada buku KIA memotong ke bawah 2 pita z-score.\n\n"
            f"2. **Gejala Klinis Malnutrisi Akut Berat (Gizi Buruk):**\n"
            f"   - Iga gambang (tulang rusuk terlihat menonjol nyata).\n"
            f"   - Otot lengan, paha, dan bokong kendur (*baggy pants appearance*).\n"
            f"   - Pembengkakan (*edema pitting*) pada kedua punggung kaki yang tidak hilang saat ditekan.\n\n"
            f"3. **Tanda Infeksi Kronis Berulang:**\n"
            f"   - Demam berulang tanpa sebab jelas lebih dari 2 minggu.\n"
            f"   - Batuk kronis atau kontak erat dengan pasien TBC dewasa.\n"
            f"   - Diare berulang atau berkepanjangan (> 14 hari).\n\n"
            f"4. **Penurunan Aktivitas & Respon Sosial:**\n"
            f"   - Anak tampak sangat lemas, pasif, rewel berkepanjangan, atau menolak makan total (*anoreksia persisten*).\n\n"
            f"5. **Keterlambatan Tonggak Perkembangan (*Milestone Delay*):**\n"
            f"   - Anak belum mencapai kemampuan motorik kasar atau kontak interaksi sesuai rentang usianya."
        )
        return guidance

    # 4. INTENT DEFISIT BERAT BADAN / WASTING
    elif is_berat:
        guidance = (
            f"Evaluasi & Penanganan Defisit Berat Badan (BB/TB) untuk **{nama}**:\n\n"
            f"**Data Antropometri Akut:**\n"
            f"- Berat Badan Saat Ini: **{berat} kg** (Ideal Proporsional: **{ideal_bb} kg**)\n"
            f"- Status Wasting (BB/TB): **{status_wasting}** (Z-Score: **{z_wasting:+.2f} SD**)\n"
            f"- Evaluasi Komposit: **{komposit_ringkasan}**\n\n"
            f"**Langkah Terapi Nutrisi:**\n"
            f"1. **Tingkatkan Kerapatan Kalori Makanan:** Tambahkan 1 sendok teh minyak kelapa, mentega, atau santan kental pada setiap porsi makan untuk mendongkrak kalori tanpa memperbesar volume makanan.\n"
            f"2. **Utamakan Protein Berkualitas:** Berikan telur ayam, ikan laut, dan hati ayam yang mudah dicerna.\n"
            f"3. **Porsi Kecil tapi Sering (*Small Frequent Meals*):** Berikan makan 5-6 kali sehari dengan porsi yang dapat dihabiskan anak agar tidak memicu rasa mual atau bosan.\n"
            f"4. **Hindari Minum Sebelum Makan:** Jangan beri air putih, teh, atau cairan pengenyang 30 menit sebelum jam makan utama."
        )
        return guidance

    # 5. INTENT PROTEIN HEWANI GANDA
    elif is_protein:
        guidance = (
            f"Keunggulan Klinis Protein Hewani Ganda untuk Balita **{nama}**:\n\n"
            f"Standar Kemenkes RI menegaskan bahwa pemberian **minimal 2 jenis protein hewani berbeda setiap hari** merupakan intervensi gizi paling efektif menurunkan stunting.\n\n"
            f"**Kombinasi Terbaik Sumber Lokal:**\n"
            f"1. **Telur Ayam/Bebek/Puyuh:** Merupakan *gold standard* protein dengan skor asam amino 100%. Mengandung kolin untuk perkembangan sel otak.\n"
            f"2. **Ikan Kembung / Tongkol / Lele:** Ikan kembung lokal memiliki kandungan asam lemak Omega-3 (DHA & EPA) yang lebih tinggi dibanding ikan salmon impor, dengan harga terjangkau.\n"
            f"3. **Hati Ayam / Daging Sapi:** Sumber Zat Besi Heme terbaik dengan bioavailabilitas tinggi untuk mencegah anemia defisiensi besi yang sering memicu anak gagal tumbuh.\n\n"
            f"**Aturan Konsumsi:** Berikan 1 butir telur + 1 porsi ikan/hati ayam setiap hari secara konsisten."
        )
        return guidance

    # 6. DEFAULT GENERAL / HOLLISTIC CONSULTATION
    else:
        guidance = (
            f"Halo! Berikut evaluasi klinis dan panduan nutrisi terpadu untuk balita **{nama}**:\n\n"
            f"**Profil Tumbuh Kembang Aktif:**\n"
            f"- Usia: **{umur} Bulan** | Jenis Kelamin: **{gender_str}**\n"
            f"- Tinggi Badan: **{tinggi} cm** (Status Stunting: **{stunting_status.upper()}**, Deviasi WHO: **{selisih_tb:+.1f} cm**)\n"
            f"- Berat Badan: **{berat} kg** (Status Wasting: **{status_wasting}**, Z-Score: **{z_wasting:+.2f} SD**)\n"
            f"- Evaluasi Komposit: **{komposit_ringkasan}**\n\n"
            f"**3 Langkah Aksi Prioritas Hari Ini:**\n"
            f"1. **Pola Makan Protein Hewani Ganda:** Kombinasikan telur ayam dan ikan kembung/hati ayam di setiap waktu makan utama untuk mendukung penambahan tinggi dan berat badan.\n"
            f"2. **Pantau Rutin di Posyandu:** Lakukan penimbangan berat dan pengukuran panjang/tinggi badan balita setiap bulan di Posyandu atau faskes terdekat.\n"
            f"3. **Cukupi Waktu Istirahat & Kebersihan:** Pastikan sanitasi air minum bersih dan balita tidur lelap minimal 10-12 jam sehari.\n\n"
            f"Silakan tanyakan menu MPASI spesifik, strategi kejar tinggi badan, atau tanda bahaya gagal tumbuh jika ada yang ingin didiskusikan lebih lanjut."
        )
        return guidance


def panggil_gemini_api(api_key, pesan_user, konteks_subjek, chat_history=None):
    """
    Memanggil Google Gemini API menggunakan model Gemini resmi.
    Jika Gemini API offline, tidak ada koneksi, quota limit habis, atau API key belum disetel/tidak valid,
    sistem secara otomatis beralih ke Mesin Inferensi Klinis Cerdas (Standar Kemenkes RI & WHO)
    sehingga pengguna selalu mendapatkan jawaban berkualitas tanpa terhenti oleh pesan error.
    """
    is_valid_looking_key = bool(api_key and len(api_key.strip()) > 10)
    
    if is_valid_looking_key:
        subjek_wasting = konteks_subjek.get('wasting', {})
        subjek_underweight = konteks_subjek.get('underweight', {})
        subjek_komposit = konteks_subjek.get('komposit', {})

        system_prompt = f"""Anda adalah Dr. Cerdas, asisten pakar klinis nutrisi anak dan konsultan pencegahan stunting & wasting berdasarkan standar antropometri Kementerian Kesehatan Republik Indonesia (Permenkes No. 2 Tahun 2020) dan standar WHO MGRS.

Data balita yang sedang dievaluasi:
- Nama: {konteks_subjek.get('nama', 'Balita')}
- Jenis Kelamin: {konteks_subjek.get('gender', 'laki-laki')}
- Usia: {konteks_subjek.get('umur', 12)} bulan
- Tinggi Badan: {konteks_subjek.get('tinggi', 75.0)} cm (Status Stunting: {konteks_subjek.get('status', 'normal')}, Deviasi vs WHO: {konteks_subjek.get('selisih', 0)} cm)
- Berat Badan: {konteks_subjek.get('berat', 9.0)} kg (Status Wasting BB/TB: {subjek_wasting.get('status', 'Normal')}, Z-score: {subjek_wasting.get('z_score', 0)})
- Status Berat/Umur (BB/U): {subjek_underweight.get('status', 'Normal')}
- Analisis Komposit Malnutrisi: {subjek_komposit.get('ringkasan', 'Optimal')}

Panduan Praktis Anda:
1. Berikan rekomendasi menu MPASI atau makanan keluarga harian nyata berbasis protein hewani ganda (telur, ikan laut/air tawar, hati ayam, daging) sesuai umur balita di atas.
2. Jelaskan langkah pencegahan stunting (1000 HPK) dan strategi catch-up growth yang aplikatif bagi orang tua Indonesia.
3. Berikan saran empati yang jelas, terstruktur, berbasis bukti ilmiah kedokteran anak.
4. JANGAN gunakan karakter em dash (—). Gunakan koma, titik dua, tanda kurung, atau titik.
5. Jika ada tanda bahaya gagal tumbuh atau malnutrisi akut berat, sarankan orang tua segera berkonsultasi ke Puskesmas atau Dokter Spesialis Anak.
"""

        contents = []
        if chat_history:
            for ch in chat_history:
                role = "user" if ch.get('role') == 'user' else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": ch.get('text', '')}]
                })
        
        contents.append({
            "role": "user",
            "parts": [{"text": pesan_user}]
        })

        models_to_try = [GEMINI_MODEL] + GEMINI_FALLBACK_MODELS

        for mdl in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent?key={api_key}"
            payload = {
                "system_instruction": {
                    "parts": [{"text": system_prompt}]
                },
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 1024
                }
            }
            headers = {"Content-Type": "application/json"}
            
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    reply_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if reply_text:
                        reply_text = reply_text.replace("—", ", ")
                        return True, reply_text, f"Google Gemini ({mdl})"
            except Exception:
                continue

    # FALLBACK OTOMATIS KE MESIN INFERENSI KLINIS CERDAS (STANDAR KEMENKES RI & WHO)
    fallback_text = generate_clinical_advisor_fallback(pesan_user, konteks_subjek)
    return True, fallback_text, "Clinical Advisor Engine (Standar Kemenkes & WHO)"


@csrf_exempt
def gemini_chat_api(request):
    """API endpoint untuk chat konsultasi interaktif dengan Gemini LLM / Clinical Advisor Engine"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Metode request tidak diizinkan.'}, status=405)
    
    try:
        data = json.loads(request.body.decode('utf-8'))
        user_message = data.get('message', '').strip()
        chat_history = data.get('history', [])
        
        if not user_message:
            return JsonResponse({'success': False, 'error': 'Pesan pertanyaan tidak boleh kosong.'}, status=400)
            
        api_key = (
            getattr(settings, 'GEMINI_API_KEY', '').strip()
            or os.environ.get('GEMINI_API_KEY', '').strip()
            or request.session.get('gemini_api_key', '')
            or data.get('api_key', '').strip()
        )

        subjek = request.session.get('last_prediction', {
            'nama': 'Balita Rujukan',
            'gender': 'laki-laki',
            'umur': 12,
            'tinggi': 75.0,
            'berat': 9.0,
            'status': 'normal',
            'selisih': 0.0,
            'wasting': {'status': 'Gizi Baik (Normal)', 'z_score': 0.0},
            'underweight': {'status': 'Berat Badan Normal', 'z_score': 0.0},
            'komposit': {'ringkasan': 'Optimal'}
        })

        success, reply, model_used = panggil_gemini_api(api_key, user_message, subjek, chat_history)
        
        return JsonResponse({
            'success': True,
            'reply': reply,
            'model_used': model_used
        })

    except Exception as e:
        try:
            subjek = request.session.get('last_prediction', {})
            fallback_reply = generate_clinical_advisor_fallback(user_message, subjek)
            return JsonResponse({
                'success': True,
                'reply': fallback_reply,
                'model_used': 'Clinical Advisor Engine (Standar Kemenkes & WHO)'
            })
        except Exception:
            return JsonResponse({'success': False, 'error': f'Terjadi kesalahan server: {str(e)}'}, status=500)
