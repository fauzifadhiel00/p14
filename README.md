#  Si-Cerdas Balita: Aplikasi Deteksi Dini Stunting & AI Nutritional Advisor

> **Tugas Pengembangan Sistem Cerdas - Implementasi Web Framework Django**  
> Program Studi Teknologi Informasi · Politeknik Negeri Madiun (PNM)

---

##  1. Deskripsi Proyek

Aplikasi **Si-Cerdas Balita** adalah platform web cerdas berbasis **Django Framework** (didukung Machine Learning dan Artificial Intelligence) yang dirancang untuk mendeteksi risiko **stunting** pada anak usia 0–60 bulan secara dini, akurat, dan objektif berdasarkan standar antropometri World Health Organization (WHO) dan Kementerian Kesehatan Republik Indonesia.

Selain memprediksi status gizi balita, aplikasi ini memanfaatkan **Explainable AI (XAI)** untuk menjelaskan landasan klinis hasil klasifikasi serta menyediakan **AI Nutritional Advisor** yang memberikan rekomendasi menu MPASI berbasis protein hewani dan panduan aksi 1000 Hari Pertama Kehidupan (HPK).

---

##  2. Basis Pengembangan dari Modul Sebelumnya 

Proyek ini merupakan pengembangan langsung dari portofolio praktikum:
* **Modul P10 (Machine Learning Evaluation & Mini Project Proposal)**: Topik deteksi stunting pertama kali diusulkan dan diuji pada bagian *Pengembangan Praktikum P10* menggunakan dataset `data_balita.csv`. Catatan perbaikan P10 (menambahkan fitur tinggi badan) telah diimplementasikan penuh.
* **Modul P09 & P11 (Ensemble Machine Learning Pipeline)**: Menggunakan algoritma *Random Forest Classifier* (100 *decision trees*) dengan validasi silang (stratified split) sehingga mencapai akurasi uji 99.9%.
* **Modul P13 (AI Assistant & NLP/RAG Concept)**: Pemanfaatan asisten cerdas berbasis aturan klinis terstruktur untuk memberikan edukasi interaktif pencegahan stunting.
* **Modul 14 (Deployment Model ke Aplikasi Web Nyata)**: Menerapkan pemisahan model (pengetahuan) dan antarmuka web, di mana model dibekukan ke `model_stunting.joblib` dan disajikan via framework web Django.

---

##  3. Fitur Utama Aplikasi

Sesuai 4 pilar ketentuan penugasan:
1. ** Visualisasi Grafik (Data Exploration & Epidemiologi)**:
   * Kurva pertumbuhan antropometri WHO (sebaran Tinggi Badan vs Umur balita 0–60 bulan) dengan ambang batas stunting (-2 SD) dan median WHO MGRS (area line chart interaktif).
   * Grafik tren 7 tahun kasus stunting (TB/U) dan gizi kurang (BB/U) di Jawa Barat periode 2019–2025 (Chart.js grouped bar chart).
   * Grafik peringkat Top 10 Kabupaten/Kota kasus stunting tertinggi di Jawa Barat dilengkapi filter tahun real-time.
2. ** Machine Learning Model**:
   * Algoritma: *Random Forest Classifier* (100 estimators, max depth 15).
   * Input fitur: `Umur (bulan)`, `Jenis Kelamin`, `Tinggi Badan (cm)`.
   * Target klasifikasi: `normal`, `stunted`, `severely stunted`, dan `tinggi`.
   * Skor performa: Akurasi latih 100%, Akurasi uji 99.9%.
   * Serialisasi: Model dibekukan ke `model_stunting.joblib` dan dimuat secara efisien di memori.
3. ** Pemanfaatan AI (AI Nutritional Advisor, XAI & Google Gemini LLM)**:
   * *Diagnosis Ganda Komprehensif*: Menggabungkan deteksi **Stunting (TB/U - Kronis)** via Random Forest dengan **Wasting (BB/TB - Akut)** dan **Underweight (BB/U)** berdasarkan standar baku Permenkes No. 2 Tahun 2020 & WHO MGRS.
   * *Analisis Komposit Malnutrisi*: Deteksi risiko morbiditas tinggi akibat kondisi ganda (*Stunted & Wasted*).
   * *Explainable AI (XAI)*: Menghitung deviasi tinggi dan berat badan terhadap median baku WHO serta transparansi penalaran klinis.
   * *Konsultasi Interaktif Bebas Berbasis Google Gemini LLM (`gemini-3.5-flash-lite`)*: Pengguna dapat mengajukan pertanyaan bebas seputar menu MPASI, gizi protein hewani, dan tanda bahaya tumbuh kembang anak secara real-time.
   * *Action Plan Gizi 1000 HPK*: Panduan MPASI bertahap kaya protein hewani ganda (telur, ikan laut, hati ayam, susu) sesuai kelompok umur balita.
4. **Aplikasi Siap Pakai (Web App)**:
   * Dibangun dengan **Django Web Framework** (arsitektur MVC/MVT standar industri).
   * Desain UI responsif, elegan, modern (Bootstrap 5 + Bootstrap Icons + Chart.js).

---

## 4. Alur Kerja Sistem (Workflow Sederhana)

Secara umum, cara kerja sistem dari awal pengguna memasukkan data balita sampai mendapatkan saran nutrisi terbagi dalam 4 langkah mudah:

```

flowchart TD
    A["1. Input Data Balita<br/>(Nama, Umur, Gender, TB, BB)"] --> B["2. Pemrosesan Data di Django"]

    subgraph Proses ["Proses Pengolahan Data"]
        B --> C["Analisis Status Stunting<br/>(Random Forest)"]
        B --> D["Analisis Wasting & Underweight<br/>(Standar WHO dan Kemenkes)"]
    end

    C --> E["3. Simpan Hasil Pemeriksaan<br/>(Database SQLite)"]
    D --> E

    E --> F["4. Tampilkan Hasil Pemeriksaan"]
    F --> G["5. Dashboard dan Riwayat Pemeriksaan"]
    F --> H["6. Konsultasi AI<br/>(Google Gemini dan Rekomendasi MPASI)"]

```

### Penjelasan Tahapan Alur Kerja:

1. **Tahap 1: Pengguna Mengisi Formulir (Input)**
   * Pengguna (orang tua atau kader posyandu) membuka menu **Prediksi Status Gizi**.
   * Memasukkan data balita: Nama, Jenis Kelamin, Usia (bulan), Tinggi Badan (cm), Berat Badan (kg), dan wilayah Kabupaten/Kota.

2. **Tahap 2: Pemrosesan Cerdas (Machine Learning & Standar WHO)**
   * **Prediksi Stunting (TB/U):** Data umur, gender, dan tinggi badan diproses oleh model **Random Forest Classifier** yang sudah dilatih untuk menentukan status stunting (*Normal*, *Stunted*, *Severely Stunted*, atau *Tinggi*).
   * **Perhitungan Gizi Akut (BB/TB):** Sistem otomatis menghitung nilai Z-Score berat badan terhadap tinggi badan untuk mengetahui apakah anak mengalami *Wasting* (gizi kurang/kurus) sesuai standar baku WHO & Kemenkes RI.

3. **Tahap 3: Penyimpanan Otomatis ke Database**
   * Data balita dan hasil diagnosa langsung disimpan otomatis ke database lokal **SQLite** (`db.sqlite3`).
   * Hasil ini tersimpan secara permanen sehingga bisa dilihat kapan saja di menu **Riwayat Pemeriksaan**.

4. **Tahap 4: Output Hasil & Konsultasi AI (Google Gemini)**
   * Pengguna langsung melihat hasil status gizi balita dengan tampilan visual yang rapi (*Explainable AI*).
   * Data balita tadi otomatis dijadikan profil aktif di menu **Rekomendasi Nutrisi**. Di sini, pengguna bisa langsung berkonsultasi interaktif dengan asisten cerdas **Dr. Cerdas** yang ditenagai **Google Gemini AI** untuk mendapatkan contoh jadwal makan MPASI berprotein hewani ganda dan tips mengejar ketertinggalan pertumbuhan.

---

## 5. Struktur File Proyek Django

```text
D:\poltek\S3\Sistem Cerdas\Tugas\p14\
│
├── manage.py                  # Script entry point Django CLI
├── stunting_project/          # Pengaturan proyek Django (settings, urls, wsgi)
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── stunting_app/              # Aplikasi Django utama
│   ├── views.py               # Logika controller (Dashboard, Prediksi ML, AI Advisor)
│   ├── urls.py                # Routing URL web
│   └── templates/             # Template antarmuka HTML modern
│       ├── base.html          # Layout induk + Navbar + Footer
│       ├── dashboard.html     # Dashboard KPI & 4 Grafik Chart.js
│       ├── prediksi.html      # Form input & Hasil Klasifikasi ML
│       ├── riwayat.html       # Halaman Khusus Log & Surveilans SQLite Real-Time
│       ├── ai_advisor.html    # XAI & Rekomendasi Gizi 1000 HPK
│       └── tentang.html       # Dokumentasi riset & Profil pengembang
│
├── model_stunting.joblib      # File binary model ML hasil pelatihan (joblib)
├── data_balita.csv            # Dataset antropometri balita (120.999 baris)
├── app.py                     # Versi alternatif Streamlit
├── README.md                  # Dokumentasi proyek lengkap
└── *.ipynb                    # Portofolio notebook praktikum P01 s.d P13 & UTS
```

---

## 6. Cara Menjalankan Aplikasi Web Django

1. Buka **Terminal / PowerShell** atau **Anaconda Prompt**.
2. Masuk ke direktori tugas ini:
   ```powershell
   cd "D:\poltek\S3\Sistem Cerdas\Tugas\p14"
   ```
3. Jalankan server lokal Django:
   ```powershell
   python manage.py runserver
   ```
4. Buka browser dan akses alamat:
   ```text
   http://127.0.0.1:8000/
   ```

---

## 7. Sumber Data & Arsitektur Database (SQLite)

Proyek ini mengintegrasikan **3 lapisan data** yang saling melengkapi:

1. **Data Pelatihan Machine Learning (Benchmark WHO MGRS)**:
   * **Dataset**: `data_balita.csv` (120.999 baris data antropometri balita).
   * **Fungsi**: Melatih model Random Forest Classifier untuk memprediksi status stunting (TB/U) berdasarkan standar Permenkes No. 2 Tahun 2020.
   * **Sumber**: Rendi Putra (Kaggle Dataset Hub - CC BY-SA 4.0).

2. **Data Resmi Jawa Barat (7 Tahun: 2019 s.d. 2025)**:
   * **Data Utama**: `jumlah_balita_stunting_berdasarkan_kabupaten_kota_di.csv` (Dinkes Jabar - Kasus Stunting TB/U).
   * **Data Pendukung**: `od_17474_jml_balita_kurang_gizi_dengan_indeks_bbu_brdsrkn_k_v1.csv` (Dinkes Jabar - Kasus Gizi Kurang BB/U).
   * **Penyimpanan**: Disimpan ke database lokal `db.sqlite3` pada tabel `DataStuntingJabar` dan `DataGiziKurangJabar` (mencakup 27 Kabupaten/Kota di Jawa Barat).
   * **Skrip Impor Otomatis**: `python import_jabar_data.py`.

3. **Data Pemeriksaan Balita Real-Time (Live SQLite)**:
   * Setiap penimbangan baru yang dihitung melalui menu `/prediksi/` otomatis tersimpan ke tabel `PemeriksaanBalita` di `db.sqlite3`.
   * Log penimbangan terkini langsung ditampilkan pada tabel di menu **Dashboard**.

---

## 8. Identitas Pengembang & Kontribusi

* **Nama Mahasiswa**: Fadhiel Fauzi Firoos
* **NIM**: 253307020
* **Kelas**: 3A
* **Program Studi**: D3 Teknologi Informasi
