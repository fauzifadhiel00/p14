from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('prediksi/', views.prediksi_view, name='prediksi'),
    path('riwayat/', views.riwayat_view, name='riwayat'),
    path('riwayat/hapus/<int:id>/', views.hapus_pemeriksaan_view, name='hapus_pemeriksaan'),
    path('ai-advisor/', views.ai_advisor_view, name='ai_advisor'),
    path('tentang/', views.tentang_view, name='tentang'),
    path('api/gemini-chat/', views.gemini_chat_api, name='gemini_chat_api'),
]
