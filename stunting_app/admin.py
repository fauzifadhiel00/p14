from django.contrib import admin
from .models import DataStuntingJabar, DataGiziKurangJabar, PemeriksaanBalita

@admin.register(DataStuntingJabar)
class DataStuntingJabarAdmin(admin.ModelAdmin):
    list_display = ('tahun', 'nama_kabupaten_kota', 'jumlah_balita_stunting', 'satuan')
    list_filter = ('tahun', 'nama_kabupaten_kota')
    search_fields = ('nama_kabupaten_kota',)

@admin.register(DataGiziKurangJabar)
class DataGiziKurangJabarAdmin(admin.ModelAdmin):
    list_display = ('tahun', 'nama_kabupaten_kota', 'jumlah_balita', 'satuan')
    list_filter = ('tahun', 'nama_kabupaten_kota')
    search_fields = ('nama_kabupaten_kota',)

@admin.register(PemeriksaanBalita)
class PemeriksaanBalitaAdmin(admin.ModelAdmin):
    list_display = ('nama', 'umur_bulan', 'gender', 'kabupaten_kota', 'status_stunting', 'status_wasting', 'created_at')
    list_filter = ('status_stunting', 'status_wasting', 'gender', 'kabupaten_kota')
    search_fields = ('nama', 'kabupaten_kota')
