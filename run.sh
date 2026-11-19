#!/bin/bash
echo "=========================================================="
echo "       SI-CERDAS BALITA (SISTEM CERDAS - PNM)"
echo "       Pengembang : Fadhiel Fauzi Firoos (253307020)"
echo "=========================================================="
echo ""

echo "[*] Menginstal dependensi..."
pip install -r requirements.txt --quiet

echo "[*] Melakukan migrasi database..."
python manage.py migrate --noinput

echo "[*] Memuat data resmi Jawa Barat ke SQLite (db.sqlite3)..."
python import_jabar_data.py

echo "[*] Menjalankan server Django..."
echo "Buka peramban di http://127.0.0.1:8000/"
python manage.py runserver
