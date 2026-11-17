# Dasbor Prediksi ISPU DKI Jakarta

Dasbor web untuk memprediksi enam parameter Indeks Standar Pencemar Udara (PM2,5, PM10, SO₂, CO, O₃, NO₂) di lima stasiun pemantau DKI Jakarta untuk 1 sampai 7 hari ke depan, memakai XGBoost dan LightGBM.

**Dasbor daring:** https://dasbor-ispu-dki.streamlit.app/

Dibuat sebagai bagian dari skripsi Program Studi Teknik Informatika, Fakultas Teknologi Informasi, Universitas Tarumanagara.

## Fitur

| Akses | Halaman |
|---|---|
| Semua pengunjung | Beranda, Data Historis, Prediksi Multi-Parameter, Perbandingan Metode, Evaluasi Model, Tanya Jawab |
| User (setelah mendaftar) | Unggah Data untuk Prediksi, Unduh Hasil, Profil |
| Admin | Kelola Data, Pelatihan Ulang Model, Kelola Pengguna, Profil |

## Data dan model

- **Data:** ISPU harian Satu Data Jakarta dan meteorologi harian NASA POWER, 1 Januari 2021 – 30 November 2025.
- **Stasiun:** DKI1 Bundaran HI, DKI2 Kelapa Gading, DKI3 Jagakarsa, DKI4 Lubang Buaya, DKI5 Kebon Jeruk.
- **Model:** satu model per stasiun × parameter × horizon (H+1 sampai H+7) × algoritma, sehingga totalnya 420 model. Setiap model belajar dari data stasiunnya sendiri (bobot 1) dan empat stasiun lain (bobot 0,3), menebak perubahan dari nilai terakhir, dan memakai 46 fitur.
- **Pembagian data:** latih 2021–2023, validasi Januari–Juni 2024, uji Juli 2024 – November 2025.

## Menjalankan di komputer sendiri

Butuh Python 3.10 atau lebih baru.

```bash
pip install -r requirements.txt
streamlit run app_dasbor_ispu.py
```

Di Windows, bila muncul `streamlit is not recognized`, pakai `py -m streamlit run app_dasbor_ispu.py`. Dasbor terbuka di http://localhost:8501.

Saat pertama dijalankan, sistem membuat `ispu.db`, mengimpor `dataset_final_2021_2025_A.csv`, dan mendaftarkan `models/v1` sebagai model versi 1.

**Akun Admin awal:** `admin@ispu.local` / `admin12345`. Kata sandi wajib diganti saat pertama masuk. Untuk memakai email dan kata sandi lain, atur variabel lingkungan `ISPU_ADMIN_EMAIL` dan `ISPU_ADMIN_PASSWORD` sebelum menjalankan dasbor untuk pertama kali.

Bila lupa kata sandi Admin:

```bash
python ispu_sistem.py reset-admin
```

## Mencoba pelatihan ulang model

Data asli berakhir 30 November 2025. Untuk mencoba pelatihan ulang, gunakan data buatan di `data_sintetis/SINTETIS_admin_2025-12-01_sd_2026-02-28_90hari.csv`:

1. Masuk sebagai Admin, buka **Kelola Data**, unggah berkas tersebut, lalu klik **Gabungkan ke data historis**.
2. Buka **Pelatihan Ulang Model**. Tombol latih aktif karena sudah ada 90 hari data baru per stasiun.
3. Jalankan pelatihan ulang (beberapa menit). Sistem membandingkan model lama dan model baru pada data yang sama, lalu mengaktifkan model baru bila galatnya tidak naik.

Data di berkas ini **bukan hasil pengukuran**. Nilainya diambil dari tanggal yang sama setahun sebelumnya lalu diberi derau acak, sehingga hanya untuk mencoba fitur.

Untuk kembali ke keadaan awal di komputer sendiri, hapus `ispu.db` dan folder `models/v2` (dan seterusnya), lalu jalankan ulang dasbor.

## Pengaturan di Streamlit Community Cloud

Isi **Settings → Secrets** dengan:

```toml
ISPU_ADMIN_EMAIL = "email.admin@contoh.com"
ISPU_ADMIN_PASSWORD = "kata-sandi-kuat"
```

Opsional, agar fitur "Kirim kode ke surel" aktif (memakai sandi aplikasi Gmail, bukan kata sandi akun):

```toml
ISPU_SMTP_HOST = "smtp.gmail.com"
ISPU_SMTP_USER = "alamat@gmail.com"
ISPU_SMTP_PASSWORD = "sandiaplikasi16huruf"
```

Jangan menulis kata sandi di kode atau berkas yang di-push.

## Batasan versi daring

Penyimpanan di Streamlit Community Cloud tidak permanen. Setiap kali aplikasi dimulai ulang (setelah pembaruan kode, reboot, atau tidur karena tidak dikunjungi), akun yang didaftarkan, data yang digabungkan, dan model hasil pelatihan ulang akan hilang, dan dasbor kembali ke data dan model versi 1.

## Struktur berkas

```
app_dasbor_ispu.py              antarmuka Streamlit (Pengunjung, User, Admin)
ispu_core.py                    data, fitur, model, evaluasi, validasi berkas unggahan
ispu_sistem.py                  basis data SQLite, akun, data historis, versi model, pelatihan ulang
dataset_final_2021_2025_A.csv   dataset hasil pengolahan data
models/hiperparameter.json      hiperparameter hasil penalaan
models/v1/                      420 model final (5 berkas, satu per stasiun) dan hasil evaluasinya
data_sintetis/                  data buatan untuk mencoba pelatihan ulang
.streamlit/config.toml          tema tampilan
```
