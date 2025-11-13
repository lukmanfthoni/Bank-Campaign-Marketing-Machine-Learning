

## 📋 Ringkasan Proyek

Proyek ini bertujuan meningkatkan efektivitas kampanye pemasaran bank untuk produk deposito berjangka melalui machine learning. Dengan membangun model prediktif, bank dapat mengidentifikasi calon nasabah potensial secara lebih akurat, mengurangi biaya operasional, dan meningkatkan tingkat konversi.

### Permasalahan Bisnis

Bank menghadapi tantangan dalam efektivitas kampanye pemasaran:
- Biaya operasional tinggi (kontak telepon/cellular, tenaga marketing, waktu)
- Tingkat konversi rendah untuk produk deposito berjangka
- Pengalaman pelanggan buruk akibat kontak berulang yang tidak relevan
- Kehilangan peluang untuk fokus pada nasabah yang benar-benar berminat

### Tujuan

**Tujuan Utama:** Meningkatkan efektivitas kampanye pemasaran dengan menaikkan conversion rate dari kondisi saat ini menjadi 25-30% dalam 6 bulan melalui targeting yang lebih akurat.

**Tujuan Sekunder:**
- Mengurangi biaya kampanye pemasaran hingga 30-40%
- Mengoptimalkan frekuensi kontak kampanye per nasabah
- Meningkatkan efisiensi tim marketing
- Meningkatkan kepuasan nasabah
- Mengidentifikasi segmen nasabah dengan potensi tertinggi
- Menentukan timing optimal untuk kampanye
- Meningkatkan total nilai deposito yang berhasil dihimpun sebesar 40-50%

## 📊 Dataset

Dataset berisi **7.813 observasi** dengan fitur-fitur berikut:

### Profil Nasabah
- **age**: Usia nasabah
- **job**: Jenis pekerjaan (admin, wiraswasta, layanan, dll.)
- **balance**: Saldo rekening
- **housing**: Memiliki pinjaman rumah (yes/no)
- **loan**: Memiliki pinjaman pribadi (yes/no)

### Data Kampanye
- **contact**: Jalur komunikasi (cellular/telephone/unknown)
- **month**: Bulan kontak terakhir (1-12)
- **campaign**: Jumlah kontak selama kampanye
- **pdays**: Hari sejak kontak terakhir
- **deposit**: Variabel target - berlangganan deposito berjangka (yes/no)

**Distribusi Target:** 
- Tidak Deposit: 52,2%
- Deposit: 47,7%

## 🛠️ Teknologi yang Digunakan

```python
pandas==2.3.2
numpy==2.3.3
scikit-learn==1.7.1
xgboost==2.1.1
imbalanced-learn==0.14.0
matplotlib==3.10.6
seaborn
shap
lime==0.2.0.1
category-encoders==2.8.1
streamlit==1.51.0
joblib==1.5.2
dill==0.4.0
```

