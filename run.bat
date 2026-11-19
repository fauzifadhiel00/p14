
@echo off
title Si-Cerdas - Django Web Server
color 0b

echo ==========================================================
echo        SI-CERDAS (SISTEM CERDAS)
echo ==========================================================
echo.

echo [*] Memeriksa instalasi dependensi...
python -m pip install -r requirements.txt --quiet

echo [*] Melakukan inisialisasi database Django...
python manage.py migrate --noinput

echo [*] Memuat data resmi Jawa Barat ke SQLite (db.sqlite3)...
python import_jabar_data.py

echo [*] Membuka browser web (http://127.0.0.1:8000)...
start http://127.0.0.1:8000/

echo [*] Menjalankan Django Development Server...
echo [*] Tekan CTRL + C untuk menghentikan server.
echo.
python manage.py runserver
pause