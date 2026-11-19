import os
import django
import pandas as pd
from pathlib import Path

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stunting_project.settings')
django.setup()

from stunting_app.models import DataStuntingJabar, DataGiziKurangJabar

BASE_DIR = Path(__file__).resolve().parent
STUNTING_CSV = BASE_DIR / 'data_stunting_jabar' / 'jumlah_balita_stunting_berdasarkan_kabupaten_kota_di.csv'
GIZI_CSV = BASE_DIR / 'data_gizi_kurang_jabar' / 'od_17474_jml_balita_kurang_gizi_dengan_indeks_bbu_brdsrkn_k_v1.csv'

LATEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

def import_stunting():
    print(f"Mengimpor Data Utama (Stunting Jawa Barat) untuk tahun {LATEST_YEARS}...")
    if not STUNTING_CSV.exists():
        print(f"File {STUNTING_CSV} tidak ditemukan!")
        return 0
        
    df = pd.read_csv(STUNTING_CSV)
    df_filtered = df[df['tahun'].isin(LATEST_YEARS)]
    
    # Hapus data tahun terkait terlebih dahulu agar idempotent (tidak duplikat jika dijalankan ulang)
    DataStuntingJabar.objects.filter(tahun__in=LATEST_YEARS).delete()
    
    objects = []
    for _, row in df_filtered.iterrows():
        objects.append(DataStuntingJabar(
            kode_provinsi=int(row.get('kode_provinsi', 32)),
            nama_provinsi=str(row.get('nama_provinsi', 'JAWA BARAT')),
            kode_kabupaten_kota=int(row.get('kode_kabupaten_kota', 0)),
            nama_kabupaten_kota=str(row.get('nama_kabupaten_kota', '')).strip().upper(),
            jumlah_balita_stunting=int(row.get('jumlah_balita_stunting', 0)),
            satuan=str(row.get('satuan', 'ORANG')),
            tahun=int(row.get('tahun', 0))
        ))
    DataStuntingJabar.objects.bulk_create(objects)
    print(f"Berhasil mengimpor {len(objects)} baris data stunting!")
    return len(objects)

def import_gizi_kurang():
    print(f"Mengimpor Data Pendukung (Gizi Kurang BB/U Jawa Barat) untuk tahun {LATEST_YEARS}...")
    if not GIZI_CSV.exists():
        print(f"File {GIZI_CSV} tidak ditemukan!")
        return 0
        
    df = pd.read_csv(GIZI_CSV)
    df_filtered = df[df['tahun'].isin(LATEST_YEARS)]
    
    # Hapus data tahun terkait terlebih dahulu agar idempotent
    DataGiziKurangJabar.objects.filter(tahun__in=LATEST_YEARS).delete()
    
    objects = []
    for _, row in df_filtered.iterrows():
        val = row.get('jumlah_balita', 0)
        objects.append(DataGiziKurangJabar(
            kode_provinsi=int(row.get('kode_provinsi', 32)),
            nama_provinsi=str(row.get('nama_provinsi', 'JAWA BARAT')),
            kode_kabupaten_kota=int(row.get('kode_kabupaten_kota', 0)),
            nama_kabupaten_kota=str(row.get('nama_kabupaten_kota', '')).strip().upper(),
            jumlah_balita=float(val) if pd.notna(val) else 0.0,
            satuan=str(row.get('satuan', 'ORANG')),
            tahun=int(row.get('tahun', 0))
        ))
    DataGiziKurangJabar.objects.bulk_create(objects)
    print(f"Berhasil mengimpor {len(objects)} baris data gizi kurang!")
    return len(objects)

if __name__ == '__main__':
    cnt_s = import_stunting()
    cnt_g = import_gizi_kurang()
    print("\n[SELESAI] Data berhasil tersimpan di SQLite (db.sqlite3):")
    print(f"- Total Data Stunting Jabar (2019-2025): {DataStuntingJabar.objects.count()} baris")
    print(f"- Total Data Gizi Kurang Jabar (2019-2025): {DataGiziKurangJabar.objects.count()} baris")
