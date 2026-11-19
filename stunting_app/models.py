from django.db import models

class DataStuntingJabar(models.Model):
    """
    Data Resmi Dinas Kesehatan Jawa Barat:
    Jumlah Balita Stunting Berdasarkan Kabupaten/Kota (TB/U)
    """
    kode_provinsi = models.IntegerField(default=32)
    nama_provinsi = models.CharField(max_length=50, default='JAWA BARAT')
    kode_kabupaten_kota = models.IntegerField()
    nama_kabupaten_kota = models.CharField(max_length=100)
    jumlah_balita_stunting = models.IntegerField()
    satuan = models.CharField(max_length=20, default='ORANG')
    tahun = models.IntegerField(db_index=True)

    class Meta:
        verbose_name = "Data Stunting Jabar"
        verbose_name_plural = "Data Stunting Jabar"
        ordering = ['tahun', 'nama_kabupaten_kota']

    def __str__(self):
        return f"{self.nama_kabupaten_kota} ({self.tahun}): {self.jumlah_balita_stunting} balita"


class DataGiziKurangJabar(models.Model):
    """
    Data Resmi Pendukung Dinas Kesehatan Jawa Barat:
    Jumlah Balita Kurang Gizi dengan Indeks BB/U Berdasarkan Kabupaten/Kota
    """
    kode_provinsi = models.IntegerField(default=32)
    nama_provinsi = models.CharField(max_length=50, default='JAWA BARAT')
    kode_kabupaten_kota = models.IntegerField()
    nama_kabupaten_kota = models.CharField(max_length=100)
    jumlah_balita = models.FloatField()
    satuan = models.CharField(max_length=20, default='ORANG')
    tahun = models.IntegerField(db_index=True)

    class Meta:
        verbose_name = "Data Gizi Kurang Jabar"
        verbose_name_plural = "Data Gizi Kurang Jabar"
        ordering = ['tahun', 'nama_kabupaten_kota']

    def __str__(self):
        return f"{self.nama_kabupaten_kota} ({self.tahun}): {self.jumlah_balita} balita"


class PemeriksaanBalita(models.Model):
    """
    Pencatatan Penimbangan Balita Realtime (Posyandu / Mandiri)
    Hasil inferensi Machine Learning & Z-Score klinis yang disimpan langsung ke database SQLite.
    """
    nama = models.CharField(max_length=100)
    gender = models.CharField(max_length=20)
    umur_bulan = models.IntegerField()
    tinggi_badan = models.FloatField()
    berat_badan = models.FloatField()
    kabupaten_kota = models.CharField(max_length=100, default='JAWA BARAT')
    status_stunting = models.CharField(max_length=50) # Prediksi Random Forest (TB/U)
    status_wasting = models.CharField(max_length=50)  # Klasifikasi Z-Score BB/TB
    status_underweight = models.CharField(max_length=50) # Klasifikasi Z-Score BB/U
    evaluasi_komposit = models.CharField(max_length=150)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Pemeriksaan Balita Realtime"
        verbose_name_plural = "Pemeriksaan Balita Realtime"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.nama} ({self.umur_bulan} bln) - {self.status_stunting}"
