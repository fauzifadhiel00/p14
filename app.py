"""
Aplikasi Web Deteksi Dini Stunting Balita & AI Nutritional Advisor
Pengembangan Proyek Sistem Cerdas (Pertemuan 10 & 14)
Politeknik Negeri Madiun (PNM)
Pengembang: Fadhiel Fauzi Firoos (NIM: 253307020) - Kelas 3A
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="AI Deteksi Stunting Balita",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik antarmuka
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-normal {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: bold;
        border-left: 6px solid #31C48D;
    }
    .status-tinggi {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: bold;
        border-left: 6px solid #3F83F8;
    }
    .status-stunted {
        background-color: #FEF08A;
        color: #713F12;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: bold;
        border-left: 6px solid #EAB308;
    }
    .status-severely {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: bold;
        border-left: 6px solid #F05252;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHING & LOAD MODEL & DATASET
# ==========================================
@st.cache_resource
def muat_model():
    model_path = "model_stunting.joblib"
    if os.path.exists(model_path):
        pkg = joblib.load(model_path)
        return pkg
    return None

@st.cache_data
def muat_dataset():
    data_path = "data_balita.csv"
    if os.path.exists(data_path):
        # Ambil sampel agar komputasi grafik eksplorasi responsif & cepat
        df = pd.read_csv(data_path)
        return df
    return None

model_package = muat_model()
dataset_balita = muat_dataset()

# Standar Median Tinggi Badan Ideal Balita (WHO MGRS & Kemenkes RI) untuk perbandingan
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

def hitung_estimasi_ideal(umur, gender):
    # Interpolasi linear sederhana dari tabel standar WHO
    umur_kunci = sorted(STANDAR_WHO.keys())
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

# ==========================================
# 3. SIDEBAR: IDENTITAS & PARAMETER
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/baby.png", width=120)
    st.markdown("### 👶 Si-Cerdas Balita")
    st.markdown("**Sistem Deteksi Stunting & Rekomendasi Gizi AI**")
    st.markdown("---")
    st.markdown("👤 **Pengembang**:")
    st.markdown("- **Nama**: Fadhiel Fauzi Firoos")
    st.markdown("- **NIM**: 253307020 / Kelas 3A")
    st.markdown("- **Institusi**: Politeknik Negeri Madiun")
    st.markdown("- **Mata Kuliah**: Sistem Cerdas (TI24317)")
    st.markdown("---")
    st.markdown("💡 **Basis Pengembangan**:")
    st.markdown("Pengembangan modul **P10 (Stunting Toddler Detection)** dan implementasi deployment web sesuai **Modul 14**.")
    
    st.markdown("---")
    st.markdown("⚡ **Spesifikasi Model ML**:")
    if model_package:
        st.success(f"Model: Random Forest\nAkurasi Uji: {model_package['test_acc']*100:.1f}%")
    else:
        st.error("Model belum dimuat.")

# ==========================================
# 4. KONTEN UTAMA: HEADER
# ==========================================
st.markdown('<p class="main-title">Aplikasi Deteksi Dini Stunting Balita & AI Nutritional Advisor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Integrasi Machine Learning untuk Klasifikasi Status Gizi Antropometri dan Kecerdasan Buatan untuk Rekomendasi Tumbuh Kembang 1000 HPK</p>', unsafe_allow_html=True)

# Tabs Navigasi
tab_prediksi, tab_ai, tab_grafik, tab_tentang = st.tabs([
    "🔍 Deteksi & Prediksi ML", 
    "🤖 Pemanfaatan AI (Advisor & Edukasi)", 
    "📊 Eksplorasi Grafik & Data", 
    "📖 Informasi Proyek & Riset"
])

# ==========================================
# TAB 1: PREDIKSI & DETEKSI CERDAS (ML)
# ==========================================
with tab_prediksi:
    st.markdown("### 📝 Input Data Antropometri Balita")
    st.info("Masukkan data balita (usia, jenis kelamin, dan tinggi badan) untuk mendapatkan klasifikasi status gizi instan berbasis model Machine Learning.")

    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        nama_balita = st.text_input("Nama Balita / Subjek", value="Balita Sehat", placeholder="Masukkan nama balita")
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"], index=0)
        gender_icon = "👦" if jenis_kelamin == "laki-laki" else "👧"

    with col_input2:
        umur_bulan = st.slider("Usia Balita (Bulan)", min_value=0, max_value=60, value=12, step=1, help="Rentang usia 0 sampai 60 bulan (5 tahun)")
        tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0, value=75.0, step=0.1, help="Tinggi badan diukur dalam posisi telentang (<2 tahun) atau berdiri (≥2 tahun)")

    # Hitung standar ideal WHO untuk usia dan gender terpilih
    tinggi_ideal, tinggi_min_normal = hitung_estimasi_ideal(umur_bulan, jenis_kelamin)
    selisih_ideal = tinggi_badan - tinggi_ideal

    # Tombol Prediksi
    st.markdown("<br>", unsafe_allow_html=True)
    btn_prediksi = st.button("🚀 Jalankan Prediksi Status Gizi ML", use_container_width=True)

    if btn_prediksi:
        if model_package is None:
            st.error("Model ML belum siap. Silakan periksa file model_stunting.joblib.")
        else:
            model = model_package['model']
            gender_num = 1 if jenis_kelamin == "laki-laki" else 0
            
            # Format input persis seperti fitur saat pelatihan (Umur, gender_num, Tinggi Badan)
            df_input = pd.DataFrame({
                'Umur (bulan)': [umur_bulan],
                'gender_num': [gender_num],
                'Tinggi Badan (cm)': [tinggi_badan]
            })

            # Prediksi kelas dan probabilitas
            prediksi = model.predict(df_input)[0]
            probabilitas = model.predict_proba(df_input)[0]
            daftar_kelas = model.classes_

            # Simpan riwayat di session state untuk tab AI
            st.session_state['hasil_prediksi'] = {
                'nama': nama_balita,
                'umur': umur_bulan,
                'gender': jenis_kelamin,
                'tinggi': tinggi_badan,
                'ideal': tinggi_ideal,
                'min_normal': tinggi_min_normal,
                'status': prediksi,
                'prob': dict(zip(daftar_kelas, probabilitas))
            }

            st.markdown("---")
            st.markdown(f"### 📋 Hasil Diagnosis: **{nama_balita}** ({gender_icon} {jenis_kelamin.capitalize()}, {umur_bulan} Bulan)")

            col_res1, col_res2, col_res3 = st.columns([1.5, 1, 1])

            with col_res1:
                if prediksi == "normal":
                    st.markdown(f'<div class="status-normal">🟢 <b>STATUS GIZI: NORMAL</b><br>Tinggi badan balita sesuai dengan rentang pertumbuhan standar usianya.</div>', unsafe_allow_html=True)
                elif prediksi == "tinggi":
                    st.markdown(f'<div class="status-tinggi">🔵 <b>STATUS GIZI: TINGGI</b><br>Tinggi badan balita berada di atas rata-rata populasi sebayanya.</div>', unsafe_allow_html=True)
                elif prediksi == "stunted":
                    st.markdown(f'<div class="status-stunted">🟡 <b>STATUS GIZI: STUNTED (PENDEK)</b><br>Tinggi badan balita berada di bawah batas normal standar WHO (-2 SD). Perlu evaluasi nutrisi.</div>', unsafe_allow_html=True)
                else: # severely stunted
                    st.markdown(f'<div class="status-severely">🔴 <b>STATUS GIZI: SEVERELY STUNTED (SANGAT PENDEK)</b><br>Kondisi defisit pertumbuhan kronis berat (-3 SD). Butuh intervensi klinis segera!</div>', unsafe_allow_html=True)

            with col_res2:
                st.metric("Tinggi Input", f"{tinggi_badan:.1f} cm", delta=f"{selisih_ideal:+.1f} cm vs Median WHO")

            with col_res3:
                st.metric("Tinggi Ideal WHO", f"{tinggi_ideal:.1f} cm", f"Batas Bawah: {tinggi_min_normal:.1f} cm")

            # Distribusi Probabilitas Prediksi Model
            st.markdown("<br><b>Tingkat Keyakinan Prediksi Model ML (Confidence Scores):</b>", unsafe_allow_html=True)
            df_prob = pd.DataFrame({
                'Status Gizi': [c.capitalize() for c in daftar_kelas],
                'Probabilitas (%)': [p * 100 for p in probabilitas]
            }).sort_values(by='Probabilitas (%)', ascending=False)
            
            st.bar_chart(df_prob.set_index('Status Gizi'), color="#3B82F6")

# ==========================================
# TAB 2: PEMANFAATAN AI (ADVISOR & EXPLAINABLE AI)
# ==========================================
with tab_ai:
    st.markdown("### 🤖 Kecerdasan Buatan: Nutritional Advisor & Clinical Decision Support")
    st.markdown("Modul ini mengombinasikan **Explainable AI (XAI)** untuk menganalisis penyebab diagnosis dan **Rule-based Clinical AI Engine** untuk menyusun panduan aksi gizi 1000 HPK.")

    riwayat = st.session_state.get('hasil_prediksi', None)

    if riwayat is None:
        st.warning("⚠️ Belum ada data prediksi aktif. Silakan lakukan prediksi terlebih dahulu di **Tab 1 (Deteksi & Prediksi ML)**.")
        # Data default untuk demo
        subjek_demo = {"nama": "Balita (Contoh)", "umur": 14, "gender": "laki-laki", "tinggi": 72.0, "status": "stunted", "ideal": 78.0, "min_normal": 74.0}
    else:
        subjek_demo = riwayat

    st.markdown(f"#### 🔍 Analisis Cerdas untuk: **{subjek_demo['nama']}** ({subjek_demo['umur']} Bulan, {subjek_demo['gender'].capitalize()}, TB: {subjek_demo['tinggi']} cm)")

    col_ai1, col_ai2 = st.columns([1, 1])

    with col_ai1:
        st.markdown("##### 🧠 1. Explainable AI: Mengapa Model Memutuskan Status Ini?")
        status = subjek_demo['status']
        umur = subjek_demo['umur']
        tb = subjek_demo['tinggi']
        ideal = subjek_demo.get('ideal', 80.0)
        min_norm = subjek_demo.get('min_normal', 75.0)

        if status == "normal":
            st.success(f"""
            - **Evaluasi Ambang Antropometri**: Tinggi badan {tb} cm berada di dalam rentang aman standar WHO (≥ {min_norm} cm).
            - **Pola Pertumbuhan**: Pertumbuhan linier balita berjalan optimal sesuai usia {umur} bulan.
            - **Tingkat Risiko**: Sangat rendah (Normal Z-Score). Pertahankan stimulasi dan pola asuh gizi seimbang.
            """)
        elif status == "tinggi":
            st.info(f"""
            - **Evaluasi Ambang Antropometri**: Tinggi badan {tb} cm melebihi median rata-rata ({ideal} cm).
            - **Pola Pertumbuhan**: Balita memiliki laju perkembangan tulang dan genetik di atas rata-rata.
            - **Tingkat Risiko**: Normal / Baik. Pastikan asupan nutrisi mikro tetap seimbang untuk menyokong pertumbuhan tulang.
            """)
        elif status == "stunted":
            st.warning(f"""
            - **Evaluasi Ambang Antropometri**: Tinggi badan {tb} cm berada {abs(ideal - tb):.1f} cm di bawah median WHO ({ideal} cm) dan di bawah batas aman normal ({min_norm} cm).
            - **Indikasi Klinis**: Mengindikasikan defisit gizi kronis dalam rentang waktu beberapa bulan terakhir (*Linear Growth Faltering*).
            - **Tingkat Risiko**: Sedang / Waspada. Potensi pemulihan tinggi (*catch-up growth*) jika diintervensi segera sebelum usia 2 tahun.
            """)
        else: # severely stunted
            st.error(f"""
            - **Evaluasi Ambang Antropometri**: Defisit tinggi badan signifikan ({tb} cm jauh di bawah batas normal {min_norm} cm).
            - **Indikasi Klinis**: Mengindikasikan malnutrisi kronis berat yang berisiko memengaruhi perkembangan kognitif dan imunitas anak.
            - **Tingkat Risiko**: Kritis (Merah). Membutuhkan rujukan segera ke Puskesmas / Dokter Spesialis Anak untuk Program Makanan Tambahan (PMT) Pemulihan.
            """)

    with col_ai2:
        st.markdown("##### 🥗 2. Rekomendasi Menu & Intervensi Nutrisi AI")
        if umur < 6:
            st.markdown("""
            * **Fase Usia**: 0 - 6 Bulan (ASI Eksklusif).
            * **Intervensi Utama**: 
              - Berikan ASI saja tanpa tambahan air putih atau makanan lain (*On Demand* min. 8-12 kali sehari).
              - Ibu menyusui wajib mengonsumsi protein hewani tinggi (ikan, telur, daging) dan suplemen zat besi/asam folat.
              - Lakukan IMD (Inisiasi Menyusu Dini) dan posisikan pelekatan menyusui dengan tepat.
            """)
        elif 6 <= umur <= 11:
            st.markdown("""
            * **Fase Usia**: 6 - 11 Bulan (Transisi MPASI Pertama).
            * **Intervensi Menu**:
              - Tekstur: Makanan lumat semi-kental (bubur saring), frekuensi 2-3 kali makan + 1-2 kali selingan.
              - **Protein Hewani Wajib**: 1 butir telur/hari, cincangan hati ayam, kaldu ceker, atau ikan air tawar.
              - Lemak tambahan: 1 sdt minyak kelapa / santan matang untuk meningkatkan densitas kalori.
            """)
        elif 12 <= umur <= 23:
            st.markdown("""
            * **Fase Usia**: 12 - 23 Bulan (Periode Kritis 1000 HPK).
            * **Intervensi Menu**:
              - Tekstur: Makanan cincang kasar / makanan keluarga, frekuensi 3-4 kali sehari + 2 kali cemilan sehat.
              - Target Protein: Minimal 2 variasi protein hewani berbeda per hari (contoh: telur rebus pagi, ikan kembung siang, daging cincang malam).
              - Suplementasi: Vitamin A kapsul merah pada bulan Februari & Agustus di Posyandu.
            """)
        else: # 24 - 59 bulan
            st.markdown("""
            * **Fase Usia**: 2 - 5 Tahun (Balita Aktif).
            * **Intervensi Menu**:
              - Menu keluarga seimbang (Isi Piringku: 1/3 karbohidrat, 1/3 sayur & buah, 1/3 lauk pauk protein hewani).
              - Hindari minuman manis kemasan (*ultra-processed food*) sebelum jam makan utama agar nafsu makan terjaga.
              - Stimulasi motorik kasar dan aktivitas luar ruangan untuk paparan sinar matahari (Vitamin D alami).
            """)

    st.markdown("---")
    st.markdown("##### 💬 3. Tanya AI Pencegahan Stunting (Konsultasi Interaktif)")
    pertanyaan_list = [
        "Bagaimana cara mengejar ketertinggalan tinggi badan balita (Catch-up Growth)?",
        "Mengapa protein hewani lebih efektif mencegah stunting dibanding protein nabati?",
        "Apa saja tanda bahaya (Red Flags) gagal tumbuh pada balita?",
        "Bagaimana hubungan sanitasi air bersih dengan kejadian stunting?"
    ]
    pilihan_tanya = st.selectbox("Pilih topik konsultasi atau ketik pertanyaan:", pertanyaan_list)

    if pilihan_tanya == pertanyaan_list[0]:
        st.info("""
        **Jawaban AI Advisor**:
        Untuk mengejar ketertinggalan tinggi badan (*catch-up growth*), prinsip utamanya adalah pemenuhan energi dan asam amino esensial:
        1. **Fokus Protein Hewani**: Berikan sumber protein hewani berkualitas tinggi seperti telur, susu pertumbuhan, ikan laut, atau daging ayam.
        2. **Cukupi Kalsium & Zink**: Kalsium bersama Zink dan Vitamin D adalah mikronutrien pembentuk matriks tulang panjang.
        3. **Atasi Penyakit Penyerta**: Pastikan anak bebas dari infeksi berulang seperti diare atau ISPA, serta rutin periksa riwayat alergi makanan.
        """)
    elif pilihan_tanya == pertanyaan_list[1]:
        st.info("""
        **Jawaban AI Advisor**:
        Protein hewani memiliki nilai biologis yang jauh lebih tinggi daripada protein nabati karena mengandung profil asam amino esensial lengkap dan mudah diserap tubuh anak. Selain itu, protein hewani kaya akan **Zink, Zat Besi Heme, dan Vitamin B12** yang berperan langsung dalam sekresi hormon pertumbuhan (*Insulin-like Growth Factor 1* / IGF-1).
        """)
    elif pilihan_tanya == pertanyaan_list[2]:
        st.info("""
        **Jawaban AI Advisor**:
        Tanda bahaya (*Red Flags*) yang harus segera dikonsultasikan ke dokter:
        - Berat badan anak tidak naik selama 2 kali penimbangan berturut-turut (garis pertumbuhan mendatar atau menurun).
        - Anak sering lemas, pasif, dan tidak mencapai tonggak perkembangan motorik (misal: belum bisa duduk di usia 9 bulan atau belum berjalan di usia 18 bulan).
        - Terjadi muntah atau diare berkepanjangan yang menyebabkan dehidrasi.
        """)
    else:
        st.info("""
        **Jawaban AI Advisor**:
        Sanitasi lingkungan yang buruk menyebabkan anak sering mengalami infeksi enterik (saluran cerna) yang disebut *Environmental Enteric Dysfunction* (EED). Kondisi ini menyebabkan dinding usus mengalami radang kronis sehingga nutrisi makanan yang dikonsumsi balita tidak terserap optimal ke dalam darah, memicu terjadinya stunting meski anak cukup makan.
        """)

# ==========================================
# TAB 3: EKSPLORASI GRAFIK & DATA
# ==========================================
with tab_grafik:
    st.markdown("### 📊 Eksplorasi Data & Visualisasi Grafik")
    st.markdown("Dataset yang digunakan adalah data antropometri balita terbuka (Kaggle: `stunting-balita-detection-121k-rows`).")

    if dataset_balita is not None:
        total_data = len(dataset_balita)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Data Sampel", f"{total_data:,} Baris")
        col_m2.metric("Proporsi Normal", f"{(dataset_balita['Status Gizi']=='normal').mean()*100:.1f}%")
        col_m3.metric("Proporsi Stunted", f"{(dataset_balita['Status Gizi']=='stunted').mean()*100:.1f}%")
        col_m4.metric("Severely Stunted", f"{(dataset_balita['Status Gizi']=='severely stunted').mean()*100:.1f}%")

        col_g1, col_g2 = st.columns([1, 1])

        with col_g1:
            st.markdown("#### 1. Distribusi Kategori Status Gizi")
            status_counts = dataset_balita['Status Gizi'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444']
            status_counts.plot(kind='bar', color=colors, ax=ax, edgecolor='black', alpha=0.85)
            ax.set_ylabel("Jumlah Balita")
            ax.set_xlabel("Status Gizi")
            ax.set_title("Distribusi Status Gizi Balita dalam Dataset", fontsize=11, fontweight='bold')
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)

        with col_g2:
            st.markdown("#### 2. Distribusi Status Gizi per Jenis Kelamin")
            gender_status = pd.crosstab(dataset_balita['Jenis Kelamin'], dataset_balita['Status Gizi'], normalize='index') * 100
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            gender_status.plot(kind='bar', stacked=True, ax=ax2, colormap='Spectral', edgecolor='black', alpha=0.85)
            ax2.set_ylabel("Persentase (%)")
            ax2.set_title("Proporsi Status Gizi Berdasarkan Jenis Kelamin", fontsize=11, fontweight='bold')
            plt.xticks(rotation=0)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("#### 3. Kurva Pertumbuhan: Sebaran Tinggi Badan vs Usia Balita")
        st.markdown("Grafik scatter plot di bawah memperlihatkan sebaran tinggi badan terhadap usia balita beserta garis rata-rata tinggi badan.")
        
        # Ambil subset acak 1.500 data untuk rendering grafik yang cepat dan bersih
        sample_viz = dataset_balita.sample(n=min(1500, len(dataset_balita)), random_state=42)
        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        
        palette = {'normal': '#10B981', 'tinggi': '#3B82F6', 'stunted': '#F59E0B', 'severely stunted': '#EF4444'}
        sns.scatterplot(
            data=sample_viz, 
            x='Umur (bulan)', 
            y='Tinggi Badan (cm)', 
            hue='Status Gizi', 
            palette=palette, 
            alpha=0.6, 
            s=35, 
            ax=ax3
        )
        
        # Jika ada input pengguna aktif di Tab 1, tandai posisinya di grafik!
        if riwayat is not None:
            ax3.scatter(
                riwayat['umur'], 
                riwayat['tinggi'], 
                color='black', 
                s=180, 
                marker='X', 
                edgecolors='white', 
                linewidth=2, 
                label=f"Posisi: {riwayat['nama']} ({riwayat['tinggi']} cm)"
            )
        
        ax3.set_title("Sebaran Tinggi Badan vs Usia (0-60 Bulan) Berdasarkan Status Gizi", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Usia (Bulan)")
        ax3.set_ylabel("Tinggi Badan (cm)")
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax3.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown("<br><b>Cuplikan Data Sampel:</b>", unsafe_allow_html=True)
        st.dataframe(dataset_balita.head(8), use_container_width=True)

    else:
        st.warning("File dataset `data_balita.csv` belum ditemukan di direktori kerja.")

# ==========================================
# TAB 4: TENTANG PROYEK & RISET
# ==========================================
with tab_tentang:
    st.markdown("### 📖 Dokumentasi Proyek Sistem Cerdas")
    st.markdown("""
    #### 🎯 Latar Belakang & Tujuan
    Stunting adalah kondisi gagal tumbuh kembang pada balita akibat kekurangan gizi kronis, terutama dalam **1000 Hari Pertama Kehidupan (HPK)**. Deteksi dini secara akurat sangat krusial agar intervensi gizi dapat segera dilakukan sebelum terjadi kerusakan permanen pada fungsi otak dan pertumbuhan fisik anak.
    
    Aplikasi ini dibangun untuk mewujudkan:
    1. **Otomasi Klasifikasi Antropometri**: Membantu kader Posyandu, Puskesmas, dan orang tua menentukan status gizi balita secara cepat dan objektif.
    2. **Explainable Clinical AI**: Memberikan penjelasan rasional di balik status diagnosis.
    3. **Actionable Recommendations**: Menyajikan rekomendasi menu gizi harian yang dapat langsung diterapkan oleh keluarga balita.

    #### 🔄 Hubungan Pengembangan dengan Portofolio Praktikum:
    - **P10 (Machine Learning Evaluation & Mini Project Proposal)**: Topik deteksi stunting pertama kali dirumuskan pada praktikum P10 menggunakan `data_balita.csv`.
    - **P09 & P11 (Ensemble Learning & Machine Learning Pipeline)**: Menggunakan algoritma *Random Forest Classifier* dengan 100 *decision trees* untuk mencapai generalisasi yang akurat.
    - **P13 (AI Assistant & NLP/RAG Concept)**: Pemanfaatan *Knowledge-based AI Advisor* untuk panduan klinis dan edukasi gizi.
    - **Modul 14 (Deployment Streamlit)**: Mengemas seluruh pipeline data science ke dalam aplikasi web interaktif menggunakan arsitektur `Streamlit + Joblib`.

    #### 👥 Identitas Mahasiswa & Pengembang
    - **Nama Mahasiswa**: Fadhiel Fauzi Firoos
    - **NIM**: 253307020
    - **Kelas**: 3A
    - **Program Studi**: D3/D4 Teknologi Informasi
    - **Institusi**: Politeknik Negeri Madiun (PNM)
    """)
    st.caption("© 2026 Si-Cerdas Balita · Tugas Sistem Cerdas Pertemuan 14")
