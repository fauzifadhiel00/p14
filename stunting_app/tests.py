from django.test import TestCase, Client
import json
from stunting_app.models import DataStuntingJabar, DataGiziKurangJabar, PemeriksaanBalita

class StuntingAppTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Buat data uji stunting & gizi kurang jabar
        DataStuntingJabar.objects.create(
            kode_provinsi=32,
            nama_provinsi='JAWA BARAT',
            kode_kabupaten_kota=3201,
            nama_kabupaten_kota='KABUPATEN BOGOR',
            jumlah_balita_stunting=12500,
            satuan='ORANG',
            tahun=2025
        )
        DataGiziKurangJabar.objects.create(
            kode_provinsi=32,
            nama_provinsi='JAWA BARAT',
            kode_kabupaten_kota=3201,
            nama_kabupaten_kota='KABUPATEN BOGOR',
            jumlah_balita=11000.0,
            satuan='ORANG',
            tahun=2025
        )

    def test_dashboard_view(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Dashboard Eksplorasi")
        self.assertContains(resp, "Data Terkini Jawa Barat")
        self.assertContains(resp, "chartJabarTrend")

    def test_prediksi_get(self):
        resp = self.client.get('/prediksi/')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Wilayah Kabupaten/Kota")
        self.assertContains(resp, "Belum Ada Data yang Dihitung")

    def test_prediksi_post_with_weight_and_db_save(self):
        initial_count = PemeriksaanBalita.objects.count()
        data = {
            'nama': 'Adit Pratama',
            'gender': 'laki-laki',
            'umur': '14',
            'tinggi': '72.0',
            'berat': '7.2',
            'kabupaten_kota': 'KABUPATEN BOGOR'
        }
        resp = self.client.post('/prediksi/', data)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Adit Pratama")
        self.assertContains(resp, "Status Wasting")
        self.assertContains(resp, "Evaluasi Komposit Malnutrisi")
        self.assertContains(resp, "Tersimpan di SQLite")

        # Verifikasi data tersimpan di SQLite
        self.assertEqual(PemeriksaanBalita.objects.count(), initial_count + 1)
        record = PemeriksaanBalita.objects.first()
        self.assertEqual(record.nama, 'Adit Pratama')
        self.assertEqual(record.kabupaten_kota, 'KABUPATEN BOGOR')

        # Verifikasi setelah submit jika GET kembali dibuka, form bersih dan tidak menahan data lama
        resp_after = self.client.get('/prediksi/')
        self.assertEqual(resp_after.status_code, 200)
        self.assertContains(resp_after, "Belum Ada Data yang Dihitung")

    def test_riwayat_view(self):
        # Buat data balita riwayat
        PemeriksaanBalita.objects.create(
            nama='Rayyan Alfarizi',
            gender='laki-laki',
            umur_bulan=18,
            tinggi_badan=82.0,
            berat_badan=11.0,
            kabupaten_kota='KABUPATEN BANDUNG',
            status_stunting='normal',
            status_wasting='Gizi Baik (Normal)',
            status_underweight='Berat Badan Normal',
            evaluasi_komposit='Tumbuh Kembang Seimbang'
        )
        resp = self.client.get('/riwayat/')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Riwayat Pemeriksaan Balita")
        self.assertContains(resp, "Rayyan Alfarizi")
        self.assertContains(resp, "KABUPATEN BANDUNG")

        # Test search query
        resp_search = self.client.get('/riwayat/?q=Rayyan')
        self.assertEqual(resp_search.status_code, 200)
        self.assertContains(resp_search, "Rayyan Alfarizi")

        # Test hapus rekaman
        rec = PemeriksaanBalita.objects.filter(nama='Rayyan Alfarizi').first()
        resp_del = self.client.post(f'/riwayat/hapus/{rec.id}/')
        self.assertEqual(resp_del.status_code, 302)
        self.assertFalse(PemeriksaanBalita.objects.filter(id=rec.id).exists())

    def test_ai_advisor_view(self):
        resp = self.client.get('/ai-advisor/')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "gemini-3.5-flash-lite")
        self.assertContains(resp, "Konfigurasi Google Gemini API Key")

        # Buat data balita di DB
        balita = PemeriksaanBalita.objects.create(
            nama='Kenzo Alvino',
            gender='laki-laki',
            umur_bulan=20,
            tinggi_badan=84.0,
            berat_badan=11.5,
            kabupaten_kota='KOTA CIMAHI',
            status_stunting='normal',
            status_wasting='Gizi Baik (Normal)',
            status_underweight='Normal',
            evaluasi_komposit='Pertumbuhan Seimbang'
        )

        # Uji pemilihan subjek via query parameter ?id=
        resp_balita = self.client.get(f'/ai-advisor/?id={balita.id}')
        self.assertEqual(resp_balita.status_code, 200)
        self.assertContains(resp_balita, "Kenzo Alvino")
        self.assertContains(resp_balita, "Data Riwayat Terpilih")

        # Uji reset profil subjek via ?reset=1
        resp_reset = self.client.get('/ai-advisor/?reset=1')
        self.assertEqual(resp_reset.status_code, 302) # Redirect kembali ke /ai-advisor/
        resp_after_reset = self.client.get('/ai-advisor/')
        self.assertContains(resp_after_reset, "Profil Baku Rujukan")

    def test_save_gemini_key_api(self):
        payload = {'action': 'save', 'api_key': 'AIzaSyTestApiKey1234567890'}
        resp = self.client.post(
            '/api/save-gemini-key/', 
            json.dumps(payload), 
            content_type='application/json'
        )
        self.assertEqual(resp.status_code, 200)
        resp_data = resp.json()
        self.assertTrue(resp_data['success'])
        self.assertTrue(resp_data['has_key'])

    def test_gemini_chat_empty_key(self):
        payload = {'message': 'Rekomendasi MPASI'}
        resp = self.client.post(
            '/api/gemini-chat/', 
            json.dumps(payload), 
            content_type='application/json'
        )
        self.assertEqual(resp.status_code, 400)
        resp_data = resp.json()
        self.assertFalse(resp_data['success'])
        self.assertIn("Gemini API Key", resp_data['error'])
