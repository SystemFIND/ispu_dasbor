"""Basis data SQLite, akun, data historis, versi model, dan pelatihan ulang.

python ispu_sistem.py reset-admin [--email alamat]
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

import ispu_core as core

BASE = Path(__file__).resolve().parent
DB = BASE / "ispu.db"
CSV_AWAL = BASE / "dataset_final_2021_2025_A.csv"
ADMIN_EMAIL = os.environ.get("ISPU_ADMIN_EMAIL", "admin@ispu.local")
ADMIN_SANDI = os.environ.get("ISPU_ADMIN_PASSWORD", "admin12345")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MENIT_TOKEN_SUREL, JEDA_KIRIM_DETIK = 15, 60
MAKS_GAGAL, MENIT_KUNCI = 5, 10
SANDI_UMUM = {
    "12345678",
    "123456789",
    "1234567890",
    "password",
    "password1",
    "password123",
    "qwerty123",
    "admin123",
    "admin1234",
    "admin12345",
    "abc12345",
    "iloveyou1",
    "11111111",
    "00000000",
}
KOL_TERSEDIA = [f"{p}_tersedia_sejak" for p in core.PARAM]

_kunci = threading.RLock()
_kunci_latih = threading.Lock()
_kunci_status = threading.Lock()
STATUS = {"jalan": False, "persen": 0.0, "pesan": "", "galat": None, "selesai": False}

STATUS_AKTIF, STATUS_MENUNGGU, STATUS_NONAKTIF = "Aktif", "Menunggu keputusan Admin", "Nonaktif"

SKEMA = """
CREATE TABLE IF NOT EXISTS pengguna(
    id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
    kata_sandi TEXT NOT NULL, peran TEXT NOT NULL CHECK(peran IN ('Admin','User')), dibuat TEXT NOT NULL,
    wajib_ganti INTEGER NOT NULL DEFAULT 0, aktif INTEGER NOT NULL DEFAULT 1, gagal_login INTEGER NOT NULL DEFAULT 0,
    terkunci_hingga TEXT, terakhir_masuk TEXT,
    kode_pulih TEXT, gagal_pulih INTEGER NOT NULL DEFAULT 0, kunci_pulih TEXT,
    token_surel TEXT, token_kedaluwarsa TEXT, token_dikirim TEXT, gagal_token INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS permintaan_reset(
    id INTEGER PRIMARY KEY AUTOINCREMENT, pengguna_id INTEGER NOT NULL, waktu TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Menunggu', selesai TEXT, oleh TEXT);
CREATE TABLE IF NOT EXISTS pembaruan_data(
    id INTEGER PRIMARY KEY AUTOINCREMENT, berkas TEXT NOT NULL, waktu TEXT NOT NULL, oleh TEXT NOT NULL,
    tanggal_awal TEXT NOT NULL, tanggal_akhir TEXT NOT NULL, baris INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS versi_model(
    id INTEGER PRIMARY KEY AUTOINCREMENT, nomor INTEGER NOT NULL, dibuat TEXT NOT NULL, batas_latih TEXT NOT NULL,
    data_hingga TEXT NOT NULL, lokasi TEXT NOT NULL, aktif INTEGER NOT NULL DEFAULT 0, keterangan TEXT,
    status TEXT NOT NULL DEFAULT 'Nonaktif');
CREATE TABLE IF NOT EXISTS evaluasi_model(
    id INTEGER PRIMARY KEY AUTOINCREMENT, versi_id INTEGER NOT NULL REFERENCES versi_model(id), sumber TEXT NOT NULL,
    stasiun TEXT NOT NULL, parameter TEXT NOT NULL, horizon INTEGER NOT NULL, algoritma TEXT NOT NULL,
    mae REAL, rmse REAL, r2 REAL, n INTEGER);
"""


@contextmanager
def koneksi():
    k = sqlite3.connect(DB, timeout=30)
    k.row_factory = sqlite3.Row
    k.execute("PRAGMA journal_mode=WAL")
    try:
        yield k
        k.commit()
    finally:
        k.close()


def _sekarang():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _tulis_data(d, k, mode):
    d = d.copy()
    for c in ["tanggal"] + KOL_TERSEDIA:
        d[c] = d[c].dt.strftime("%Y-%m-%d")
    d.to_sql("data_historis", k, if_exists=mode, index=False)


def ada_data():
    with koneksi() as k:
        return (
            k.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data_historis'").fetchone()
            is not None
        )


def impor_csv(path=None):
    d = core.normalkan_dataset(pd.read_csv(path or CSV_AWAL))
    with _kunci, koneksi() as k:
        _tulis_data(d, k, "replace")


def _migrasi(k):
    """Melengkapi basis data lama dengan kolom baru (wajib_ganti, status akun, kode pemulihan, dan status versi)."""
    kol = {r[1] for r in k.execute("PRAGMA table_info(pengguna)")}
    if "wajib_ganti" not in kol:
        k.execute("ALTER TABLE pengguna ADD COLUMN wajib_ganti INTEGER NOT NULL DEFAULT 0")
        for r in k.execute("SELECT id, kata_sandi FROM pengguna WHERE peran='Admin'").fetchall():
            if cek_sandi(ADMIN_SANDI, r["kata_sandi"]):  # Admin lama yang masih memakai kata sandi bawaan
                k.execute("UPDATE pengguna SET wajib_ganti=1 WHERE id=?", (r["id"],))
    for nama_kol, definisi in (
        ("aktif", "INTEGER NOT NULL DEFAULT 1"),
        ("gagal_login", "INTEGER NOT NULL DEFAULT 0"),
        ("terkunci_hingga", "TEXT"),
        ("terakhir_masuk", "TEXT"),
        ("kode_pulih", "TEXT"),
        ("gagal_pulih", "INTEGER NOT NULL DEFAULT 0"),
        ("kunci_pulih", "TEXT"),
        ("token_surel", "TEXT"),
        ("token_kedaluwarsa", "TEXT"),
        ("token_dikirim", "TEXT"),
        ("gagal_token", "INTEGER NOT NULL DEFAULT 0"),
    ):
        if nama_kol not in kol:
            k.execute(f"ALTER TABLE pengguna ADD COLUMN {nama_kol} {definisi}")
    kol = {r[1] for r in k.execute("PRAGMA table_info(versi_model)")}
    if "status" not in kol:
        k.execute("ALTER TABLE versi_model ADD COLUMN status TEXT NOT NULL DEFAULT 'Nonaktif'")
        k.execute("UPDATE versi_model SET status='Aktif' WHERE aktif=1")


def init_db():
    """Membuat tabel, akun Admin bawaan (wajib ganti kata sandi), dan mengimpor himpunan data awal."""
    with _kunci:
        with koneksi() as k:
            k.executescript(SKEMA)
            _migrasi(k)
            if k.execute("SELECT COUNT(*) FROM pengguna WHERE peran='Admin'").fetchone()[0] == 0:
                k.execute(
                    "INSERT INTO pengguna(nama,email,kata_sandi,peran,dibuat,wajib_ganti) VALUES(?,?,?,?,?,1)",
                    ("Administrator", ADMIN_EMAIL, hash_sandi(ADMIN_SANDI), "Admin", _sekarang()),
                )
        if not ada_data() and CSV_AWAL.exists():
            impor_csv()


def hash_sandi(sandi, garam=None, putaran=200_000):
    garam = garam or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", sandi.encode(), bytes.fromhex(garam), putaran).hex()
    return f"pbkdf2${putaran}${garam}${h}"


def cek_sandi(sandi, tersimpan):
    try:
        _, putaran, garam, h = tersimpan.split("$")
        return hmac.compare_digest(hash_sandi(sandi, garam, int(putaran)).split("$")[3], h)
    except (ValueError, IndexError):
        return False


def periksa_sandi(sandi, nama="", email=""):
    """Kebijakan kata sandi. Mengembalikan pesan galat, atau None bila kata sandi memenuhi syarat."""
    if len(sandi) < 8:
        return "Kata sandi minimal 8 karakter."
    if not re.search(r"[A-Za-z]", sandi) or not re.search(r"\d", sandi):
        return "Kata sandi harus memuat huruf dan angka."
    if sandi.lower() in SANDI_UMUM:
        return "Kata sandi terlalu umum. Pilih kata sandi yang lebih sulit ditebak."
    depan = email.strip().lower().split("@")[0] if email else ""
    if sandi.lower() in (email.strip().lower(), depan) or (nama.strip() and sandi.lower() == nama.strip().lower()):
        return "Kata sandi tidak boleh sama dengan nama atau alamat surel."
    return None


def sandi_sementara(panjang=10):
    """Kata sandi acak untuk reset oleh Admin; tanpa karakter yang mudah tertukar, selalu memuat huruf dan angka."""
    huruf, angka = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ", "23456789"
    while True:
        s = "".join(secrets.choice(huruf + angka) for _ in range(panjang))
        if any(c in huruf for c in s) and any(c in angka for c in s):
            return s


def buat_kode_pemulihan():
    """Kode pemulihan acak 16 karakter (sekitar 80 bit) berbentuk XXXX-XXXX-XXXX-XXXX."""
    abjad = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    k = "".join(secrets.choice(abjad) for _ in range(16))
    return "-".join(k[i : i + 4] for i in range(0, 16, 4))


def _normalkan_kode(kode):
    return re.sub(r"[^A-Z0-9]", "", (kode or "").upper())


def _pasang_kode(k, uid):
    """Membuat kode pemulihan baru, menyimpan hash-nya, dan mengembalikan kodenya; kode lama tidak berlaku."""
    kode = buat_kode_pemulihan()
    k.execute(
        "UPDATE pengguna SET kode_pulih=?, gagal_pulih=0, kunci_pulih=NULL WHERE id=?",
        (hash_sandi(_normalkan_kode(kode)), uid),
    )
    return kode


def daftar(nama, email, sandi, ulang):
    """Registrasi mandiri sebagai User; mengembalikan (ok, pesan, kode pemulihan)."""
    nama, email = nama.strip(), email.strip().lower()
    if not nama or not email or not sandi:
        return False, "Nama, alamat surel, dan kata sandi wajib diisi.", None
    if not EMAIL_RE.match(email):
        return False, "Format alamat surel tidak valid.", None
    galat = periksa_sandi(sandi, nama, email)
    if galat:
        return False, galat, None
    if sandi != ulang:
        return False, "Konfirmasi kata sandi tidak sama.", None
    with _kunci, koneksi() as k:
        if k.execute("SELECT 1 FROM pengguna WHERE email=?", (email,)).fetchone():
            return False, "Alamat surel sudah terdaftar.", None
        uid = k.execute(
            "INSERT INTO pengguna(nama,email,kata_sandi,peran,dibuat) VALUES(?,?,?,?,?)",
            (nama, email, hash_sandi(sandi), "User", _sekarang()),
        ).lastrowid
        kode = _pasang_kode(k, uid)
    return True, "Registrasi berhasil. Simpan kode pemulihan di bawah ini, lalu masuk dengan akun Anda.", kode


def _data_sesi(r):
    return {
        "id": r["id"],
        "nama": r["nama"],
        "email": r["email"],
        "peran": r["peran"],
        "wajib_ganti": bool(r["wajib_ganti"]),
    }


def masuk(email, sandi):
    """Mengembalikan (data pengguna, pesan). Data pengguna bernilai None bila masuk gagal."""
    umum = "Alamat surel atau kata sandi salah."
    sekarang = datetime.now()
    with _kunci, koneksi() as k:
        r = k.execute("SELECT * FROM pengguna WHERE email=?", (email.strip().lower(),)).fetchone()
        if r is None:
            hash_sandi(sandi)  # menyamakan waktu respons dengan akun yang ada
            return None, umum
        if r["terkunci_hingga"] and datetime.strptime(r["terkunci_hingga"], "%Y-%m-%d %H:%M:%S") > sekarang:
            sisa = (
                int((datetime.strptime(r["terkunci_hingga"], "%Y-%m-%d %H:%M:%S") - sekarang).total_seconds() // 60) + 1
            )
            return None, f"Terlalu banyak percobaan yang gagal. Coba lagi dalam {sisa} menit."
        if not cek_sandi(sandi, r["kata_sandi"]):
            gagal = r["gagal_login"] + 1
            if gagal >= MAKS_GAGAL:
                kunci = (sekarang + timedelta(minutes=MENIT_KUNCI)).strftime("%Y-%m-%d %H:%M:%S")
                k.execute("UPDATE pengguna SET gagal_login=0, terkunci_hingga=? WHERE id=?", (kunci, r["id"]))
                return None, f"Terlalu banyak percobaan yang gagal. Akun dikunci selama {MENIT_KUNCI} menit."
            k.execute("UPDATE pengguna SET gagal_login=? WHERE id=?", (gagal, r["id"]))
            return None, umum
        if not r["aktif"]:
            return None, "Akun ini dinonaktifkan. Hubungi Admin."
        k.execute(
            "UPDATE pengguna SET gagal_login=0, terkunci_hingga=NULL, terakhir_masuk=? WHERE id=?",
            (_sekarang(), r["id"]),
        )
        return _data_sesi(r), ""


def muat_pengguna(uid):
    """Data terbaru pengguna untuk menyegarkan sesi; None bila akun sudah tidak ada atau dinonaktifkan."""
    with koneksi() as k:
        r = k.execute("SELECT * FROM pengguna WHERE id=? AND aktif=1", (uid,)).fetchone()
    return _data_sesi(r) if r else None


def ganti_kata_sandi(uid, lama, baru, ulang):
    """Ganti kata sandi sendiri (wajib pada masuk pertama akun bawaan atau setelah reset oleh Admin)."""
    if not lama or not baru:
        return False, "Kata sandi saat ini dan kata sandi baru wajib diisi."
    with _kunci, koneksi() as k:
        r = k.execute("SELECT nama, email, kata_sandi FROM pengguna WHERE id=?", (uid,)).fetchone()
        if not r or not cek_sandi(lama, r["kata_sandi"]):
            return False, "Kata sandi saat ini salah."
        galat = periksa_sandi(baru, r["nama"], r["email"])
        if galat:
            return False, galat.replace("Kata sandi", "Kata sandi baru", 1)
        if baru != ulang:
            return False, "Konfirmasi kata sandi baru tidak sama."
        if baru == lama:
            return False, "Kata sandi baru harus berbeda dari kata sandi saat ini."
        k.execute("UPDATE pengguna SET kata_sandi=?, wajib_ganti=0 WHERE id=?", (hash_sandi(baru), uid))
    return True, "Kata sandi berhasil diganti."


def ubah_nama(uid, nama):
    nama = nama.strip()
    if not nama:
        return False, "Nama tidak boleh kosong."
    if len(nama) > 80:
        return False, "Nama terlalu panjang (maksimal 80 karakter)."
    with _kunci, koneksi() as k:
        k.execute("UPDATE pengguna SET nama=? WHERE id=?", (nama, uid))
    return True, "Nama berhasil diperbarui."


def profil(uid):
    with koneksi() as k:
        r = k.execute("SELECT nama, email, peran, dibuat, terakhir_masuk FROM pengguna WHERE id=?", (uid,)).fetchone()
    return dict(r) if r else None


PESAN_PULIH_GAGAL = "Alamat surel atau kode pemulihan salah."


def ada_kode_pemulihan(uid):
    with koneksi() as k:
        r = k.execute("SELECT kode_pulih FROM pengguna WHERE id=?", (uid,)).fetchone()
    return bool(r and r["kode_pulih"])


def buat_ulang_kode(uid, sandi):
    """Kode pemulihan baru setelah kata sandi saat ini diperiksa; mengembalikan (ok, pesan atau kode)."""
    with _kunci, koneksi() as k:
        r = k.execute("SELECT kata_sandi, peran FROM pengguna WHERE id=?", (uid,)).fetchone()
        if r and r["peran"] != "User":
            return False, "Akun Admin tidak memakai kode pemulihan."
        if not r or not cek_sandi(sandi or "", r["kata_sandi"]):
            return False, "Kata sandi saat ini salah."
        return True, _pasang_kode(k, uid)


def pulihkan_dengan_kode(email, kode, baru, ulang):
    """Mengatur kata sandi baru dengan kode pemulihan sekali pakai; kode baru dikembalikan bila berhasil."""
    email, norm = email.strip().lower(), _normalkan_kode(kode)
    sekarang = datetime.now()
    fmt = "%Y-%m-%d %H:%M:%S"
    with _kunci, koneksi() as k:
        r = k.execute("SELECT * FROM pengguna WHERE email=?", (email,)).fetchone()
        if r is None or not r["aktif"] or r["peran"] != "User" or not r["kode_pulih"]:
            hash_sandi(norm)  # menyamakan waktu respons
            return False, PESAN_PULIH_GAGAL, None
        if r["kunci_pulih"] and datetime.strptime(r["kunci_pulih"], fmt) > sekarang:
            sisa = int((datetime.strptime(r["kunci_pulih"], fmt) - sekarang).total_seconds() // 60) + 1
            return False, f"Terlalu banyak percobaan yang gagal. Coba lagi dalam {sisa} menit.", None
        if not cek_sandi(norm, r["kode_pulih"]):
            gagal = r["gagal_pulih"] + 1
            if gagal >= MAKS_GAGAL:
                kunci = (sekarang + timedelta(minutes=MENIT_KUNCI)).strftime(fmt)
                k.execute("UPDATE pengguna SET gagal_pulih=0, kunci_pulih=? WHERE id=?", (kunci, r["id"]))
                return (
                    False,
                    f"Terlalu banyak percobaan yang gagal. Pemulihan dikunci selama {MENIT_KUNCI} menit.",
                    None,
                )
            k.execute("UPDATE pengguna SET gagal_pulih=? WHERE id=?", (gagal, r["id"]))
            return False, PESAN_PULIH_GAGAL, None
        # Kode benar. Kebijakan kata sandi diperiksa setelahnya dan tidak menghabiskan kode.
        galat = periksa_sandi(baru, r["nama"], r["email"])
        if galat:
            return False, galat.replace("Kata sandi", "Kata sandi baru", 1), None
        if baru != ulang:
            return False, "Konfirmasi kata sandi baru tidak sama.", None
        k.execute(
            "UPDATE pengguna SET kata_sandi=?, wajib_ganti=0, gagal_login=0, terkunci_hingga=NULL WHERE id=?",
            (hash_sandi(baru), r["id"]),
        )
        k.execute(
            "UPDATE permintaan_reset SET status='Selesai', selesai=?, oleh=? WHERE pengguna_id=? AND status='Menunggu'",
            (_sekarang(), "kode pemulihan", r["id"]),
        )
        kode_baru = _pasang_kode(k, r["id"])
    return True, "Kata sandi berhasil diatur ulang. Silakan masuk dengan kata sandi baru.", kode_baru


PESAN_SUREL_TERKIRIM = (
    f"Jika alamat surel tersebut terdaftar, kode pemulihan sudah dikirim. Kode berlaku {MENIT_TOKEN_SUREL} menit "
    "dan hanya dapat dipakai sekali. Periksa juga folder spam."
)
PESAN_SUREL_GAGAL = "Alamat surel atau kode dari surel salah atau sudah kedaluwarsa."


def surel_siap():
    """True bila SMTP dikonfigurasi atau mode konsol (ISPU_SUREL_KONSOL=1) aktif."""
    e = os.environ
    return e.get("ISPU_SUREL_KONSOL") == "1" or bool(
        e.get("ISPU_SMTP_HOST") and e.get("ISPU_SMTP_USER") and e.get("ISPU_SMTP_PASSWORD")
    )


def kirim_surel(tujuan, subjek, isi):
    """Mengirim surel lewat SMTP. Pada mode konsol, isi surel dicetak ke terminal dan tidak dikirim ke mana pun."""
    e = os.environ
    if e.get("ISPU_SUREL_KONSOL") == "1":
        print(f"\n[SUREL UJI COBA] Kepada: {tujuan}\nSubjek: {subjek}\n{isi}\n", file=sys.stderr, flush=True)
        return
    pesan = EmailMessage()
    pesan["From"] = e.get("ISPU_SMTP_FROM") or e["ISPU_SMTP_USER"]
    pesan["To"], pesan["Subject"] = tujuan, subjek
    pesan.set_content(isi)
    port = int(e.get("ISPU_SMTP_PORT", "587"))
    if port == 465 or e.get("ISPU_SMTP_SSL") == "1":
        srv = smtplib.SMTP_SSL(e["ISPU_SMTP_HOST"], port, timeout=15)
    else:
        srv = smtplib.SMTP(e["ISPU_SMTP_HOST"], port, timeout=15)
    with srv:
        if not (port == 465 or e.get("ISPU_SMTP_SSL") == "1"):
            srv.starttls()
        srv.login(e["ISPU_SMTP_USER"], e["ISPU_SMTP_PASSWORD"])
        srv.send_message(pesan)


def _kirim_latar(tujuan, subjek, isi):
    """Pengiriman di thread terpisah agar waktu respons sama untuk surel yang terdaftar maupun tidak."""

    def kerja():
        try:
            kirim_surel(tujuan, subjek, isi)
        except Exception as ex:  # tidak boleh membocorkan keberadaan akun ke antarmuka
            print(f"[GALAT SUREL] gagal mengirim ke {tujuan}: {ex}", file=sys.stderr, flush=True)

    threading.Thread(target=kerja, daemon=True).start()


def kirim_kode_surel(email):
    """Mengirim kode sekali pakai ke surel akun User; pesannya sama untuk surel terdaftar maupun tidak."""
    if not surel_siap():
        return False, "Layanan surel belum dikonfigurasi di server ini. Hubungi pengelola aplikasi."
    email = email.strip().lower()
    fmt, kirim = "%Y-%m-%d %H:%M:%S", None
    sekarang = datetime.now()
    with _kunci, koneksi() as k:
        r = k.execute(
            "SELECT id, nama, token_dikirim FROM pengguna WHERE email=? AND aktif=1 AND peran='User'", (email,)
        ).fetchone()
        if r:
            terlalu_cepat = (
                r["token_dikirim"]
                and (sekarang - datetime.strptime(r["token_dikirim"], fmt)).total_seconds() < JEDA_KIRIM_DETIK
            )
            if not terlalu_cepat:
                abjad = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
                t = "".join(secrets.choice(abjad) for _ in range(8))
                kode = f"{t[:4]}-{t[4:]}"
                k.execute(
                    "UPDATE pengguna SET token_surel=?, token_kedaluwarsa=?, token_dikirim=?, gagal_token=0 WHERE id=?",
                    (
                        hash_sandi(t),
                        (sekarang + timedelta(minutes=MENIT_TOKEN_SUREL)).strftime(fmt),
                        sekarang.strftime(fmt),
                        r["id"],
                    ),
                )
                kirim = (email, r["nama"], kode)
    if kirim:
        tujuan, nama, kode = kirim
        _kirim_latar(
            tujuan,
            "Kode pemulihan kata sandi - Dasbor Prediksi ISPU",
            f"Halo {nama},\n\nKode untuk mengatur ulang kata sandi Anda:\n\n    {kode}\n\nKode berlaku"
            f" {MENIT_TOKEN_SUREL} menit dan hanya dapat dipakai sekali. Masukkan kode ini beserta kata sandi baru pada"
            " tab Lupa kata sandi di aplikasi.\n\nJika Anda tidak merasa meminta ini, abaikan surel ini; kata sandi"
            " Anda tidak berubah.\n",
        )
    return True, PESAN_SUREL_TERKIRIM


def atur_ulang_dengan_kode_surel(email, kode, baru, ulang):
    """Mengatur kata sandi baru dengan kode dari surel (sekali pakai, kedaluwarsa, dibatasi percobaannya)."""
    email, norm = email.strip().lower(), _normalkan_kode(kode)
    fmt, sekarang = "%Y-%m-%d %H:%M:%S", datetime.now()
    with _kunci, koneksi() as k:
        r = k.execute("SELECT * FROM pengguna WHERE email=?", (email,)).fetchone()
        valid = (
            r is not None
            and r["aktif"]
            and r["peran"] == "User"
            and r["token_surel"]
            and r["token_kedaluwarsa"]
            and datetime.strptime(r["token_kedaluwarsa"], fmt) > sekarang
        )
        if not valid:
            hash_sandi(norm)
            return False, PESAN_SUREL_GAGAL, None
        if not cek_sandi(norm, r["token_surel"]):
            gagal = r["gagal_token"] + 1
            if gagal >= MAKS_GAGAL:  # terlalu banyak tebakan: kode dihanguskan
                k.execute(
                    "UPDATE pengguna SET token_surel=NULL, token_kedaluwarsa=NULL, gagal_token=0 WHERE id=?", (r["id"],)
                )
            else:
                k.execute("UPDATE pengguna SET gagal_token=? WHERE id=?", (gagal, r["id"]))
            return False, PESAN_SUREL_GAGAL, None
        galat = periksa_sandi(baru, r["nama"], r["email"])
        if galat:
            return False, galat.replace("Kata sandi", "Kata sandi baru", 1), None
        if baru != ulang:
            return False, "Konfirmasi kata sandi baru tidak sama.", None
        k.execute(
            "UPDATE pengguna SET kata_sandi=?, wajib_ganti=0, gagal_login=0, terkunci_hingga=NULL, "
            "token_surel=NULL, token_kedaluwarsa=NULL, gagal_token=0 WHERE id=?",
            (hash_sandi(baru), r["id"]),
        )
        k.execute(
            "UPDATE permintaan_reset SET status='Selesai', selesai=?, oleh=? WHERE pengguna_id=? AND status='Menunggu'",
            (_sekarang(), "kode surel", r["id"]),
        )
        kode_baru = _pasang_kode(k, r["id"])
    return True, "Kata sandi berhasil diatur ulang. Silakan masuk dengan kata sandi baru.", kode_baru


def daftar_pengguna():
    with koneksi() as k:
        d = pd.read_sql_query(
            "SELECT p.id, p.nama, p.email, p.peran, p.dibuat, p.terakhir_masuk, p.aktif, p.terkunci_hingga, "
            "(SELECT COUNT(*) FROM permintaan_reset q WHERE q.pengguna_id=p.id AND q.status='Menunggu') AS minta_reset "
            "FROM pengguna p ORDER BY p.peran, p.nama",
            k,
        )
    sekarang = _sekarang()
    d["status"] = [
        ("Nonaktif" if not a else "Terkunci" if (isinstance(t, str) and t > sekarang) else "Aktif")
        for a, t in zip(d["aktif"], d["terkunci_hingga"])
    ]
    return d


def reset_sandi_oleh_admin(uid_target, oleh):
    """Kata sandi sementara yang wajib diganti saat masuk berikutnya; mengembalikan (ok, sandi atau pesan)."""
    sandi = sandi_sementara()
    with _kunci, koneksi() as k:
        r = k.execute("SELECT peran FROM pengguna WHERE id=?", (uid_target,)).fetchone()
        if r is None:
            return False, "Akun tidak ditemukan."
        if r["peran"] != "User":
            return False, "Reset kata sandi akun Admin dilakukan sendiri lewat halaman Profil."
        k.execute(
            "UPDATE pengguna SET kata_sandi=?, wajib_ganti=1, gagal_login=0, terkunci_hingga=NULL WHERE id=?",
            (hash_sandi(sandi), uid_target),
        )
        k.execute(
            "UPDATE permintaan_reset SET status='Selesai', selesai=?, oleh=? WHERE pengguna_id=? AND status='Menunggu'",
            (_sekarang(), oleh, uid_target),
        )
    return True, sandi


def atur_aktif(uid_target, aktif):
    with _kunci, koneksi() as k:
        r = k.execute("SELECT peran FROM pengguna WHERE id=?", (uid_target,)).fetchone()
        if r is None:
            return False, "Akun tidak ditemukan."
        if r["peran"] != "User":
            return False, "Akun Admin tidak dapat dinonaktifkan."
        k.execute("UPDATE pengguna SET aktif=? WHERE id=?", (1 if aktif else 0, uid_target))
    return True, (
        "Akun diaktifkan kembali."
        if aktif
        else "Akun dinonaktifkan. Sesi yang sedang berjalan berakhir pada interaksi berikutnya."
    )


def buka_kunci(uid_target):
    with _kunci, koneksi() as k:
        k.execute("UPDATE pengguna SET gagal_login=0, terkunci_hingga=NULL WHERE id=?", (uid_target,))
    return True, "Kunci akun dibuka."


def muat_data():
    with koneksi() as k:
        d = pd.read_sql_query(
            "SELECT * FROM data_historis ORDER BY stasiun, tanggal", k, parse_dates=["tanggal"] + KOL_TERSEDIA
        )
    return d.reset_index(drop=True)


def token_data():
    """Penanda perubahan data, dipakai sebagai kunci cache di antarmuka."""
    with koneksi() as k:
        r = k.execute("SELECT COUNT(*), MAX(tanggal) FROM data_historis").fetchone()
    return int(r[0]), str(r[1])


def gabung_data(d_baru, nama_berkas, oleh):
    """Menambahkan baris hasil pra-pemrosesan ke data historis dan mencatat riwayat pembaruan."""
    with _kunci, koneksi() as k:
        _tulis_data(d_baru[core.KOLOM_DATASET], k, "append")
        awal, akhir = d_baru["tanggal"].min(), d_baru["tanggal"].max()
        k.execute(
            "INSERT INTO pembaruan_data(berkas,waktu,oleh,tanggal_awal,tanggal_akhir,baris) VALUES(?,?,?,?,?,?)",
            (nama_berkas, _sekarang(), oleh, awal.strftime("%Y-%m-%d"), akhir.strftime("%Y-%m-%d"), len(d_baru)),
        )
    return dict(awal=awal, akhir=akhir, baris=len(d_baru), per_stasiun=d_baru.groupby("stasiun").size().to_dict())


def riwayat_pembaruan():
    with koneksi() as k:
        return pd.read_sql_query(
            "SELECT waktu, berkas, oleh, tanggal_awal, tanggal_akhir, baris FROM pembaruan_data ORDER BY id DESC", k
        )


def _versi(r):
    return dict(
        id=r["id"],
        nomor=r["nomor"],
        dibuat=r["dibuat"],
        batas_latih=pd.Timestamp(r["batas_latih"]),
        data_hingga=pd.Timestamp(r["data_hingga"]),
        folder=BASE / r["lokasi"],
        keterangan=r["keterangan"],
        status=r["status"],
    )


def versi_aktif():
    with koneksi() as k:
        r = k.execute("SELECT * FROM versi_model WHERE aktif=1 ORDER BY id DESC LIMIT 1").fetchone()
    return _versi(r) if r else None


def versi_terbaru():
    with koneksi() as k:
        r = k.execute("SELECT * FROM versi_model ORDER BY id DESC LIMIT 1").fetchone()
    return _versi(r) if r else None


def daftar_versi():
    with koneksi() as k:
        return pd.read_sql_query(
            "SELECT id, nomor, dibuat, batas_latih, data_hingga, status, keterangan FROM versi_model ORDER BY id DESC",
            k,
        )


def muat_evaluasi(versi_id, sumber="baru"):
    with koneksi() as k:
        return pd.read_sql_query(
            "SELECT stasiun, parameter, horizon, algoritma, mae, rmse, r2, n FROM evaluasi_model "
            "WHERE versi_id=? AND sumber=?",
            k,
            params=(versi_id, sumber),
        )


def _simpan_versi(nomor, batas, data_hingga, folder, ket, ev_baru, ev_lama=None, status=STATUS_AKTIF):
    """Menyimpan versi model; hanya status Aktif yang menggantikan versi aktif."""
    aktif = 1 if status == STATUS_AKTIF else 0
    with _kunci, koneksi() as k:
        if aktif:
            k.execute("UPDATE versi_model SET aktif=0, status=?", (STATUS_NONAKTIF,))
        else:
            k.execute("UPDATE versi_model SET status=? WHERE status=?", (STATUS_NONAKTIF, STATUS_MENUNGGU))
        vid = k.execute(
            "INSERT INTO versi_model(nomor,dibuat,batas_latih,data_hingga,lokasi,aktif,keterangan,status) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                nomor,
                _sekarang(),
                batas.strftime("%Y-%m-%d"),
                data_hingga.strftime("%Y-%m-%d"),
                str(Path(folder).relative_to(BASE)),
                aktif,
                ket,
                status,
            ),
        ).lastrowid
        for sumber, ev in (("baru", ev_baru), ("lama", ev_lama)):
            if ev is None or ev.empty:
                continue
            k.executemany(
                "INSERT INTO evaluasi_model(versi_id,sumber,stasiun,parameter,horizon,algoritma,mae,rmse,r2,n) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        vid,
                        sumber,
                        r.stasiun,
                        r.parameter,
                        int(r.horizon),
                        r.algoritma,
                        None if pd.isna(r.mae) else r.mae,
                        None if pd.isna(r.rmse) else r.rmse,
                        None if pd.isna(r.r2) else r.r2,
                        int(r.n),
                    )
                    for r in ev.itertuples()
                ],
            )
    return vid


def _daftarkan_dari_notebook(data):
    """Mendaftarkan models/v1 hasil pipeline sebagai versi 1 tanpa melatih ulang."""
    f = core.DIR_MODEL / "v1"
    if not ((f / "meta.json").exists() and (f / "evaluasi.csv").exists()):
        return False
    hilang = [s for s in core._stasiun_ada(data) if not (f / f"{s}.joblib").exists()]
    if hilang:
        raise ValueError("Folder models/v1 dari notebook tidak lengkap, model hilang untuk: " + ", ".join(hilang) + ".")
    m = json.loads((f / "meta.json").read_text(encoding="utf-8"))
    akhir = data["tanggal"].max()
    if pd.Timestamp(m["data_hingga"]) != akhir:
        raise ValueError(
            f"Model v1 dari notebook dilatih pada data hingga {m['data_hingga']}, sedangkan himpunan data di "
            f"dasbor berakhir {akhir:%Y-%m-%d}. Gunakan dataset_final yang sama dengan notebook, atau "
            "hapus folder models/v1 agar dasbor melatih sendiri."
        )
    ket = (
        "Pelatihan awal dari notebook ("
        + ("hiperparameter hasil penalaan" if m.get("hiperparameter_json") else "hiperparameter bawaan")
        + ")"
    )
    _simpan_versi(
        1, pd.Timestamp(m["batas_latih"]), pd.Timestamp(m["data_hingga"]), f, ket, pd.read_csv(f / "evaluasi.csv")
    )
    return True


def latih_awal(progres=None):
    """Versi 1 dari models/v1 bila ada; jika tidak ada, sistem melatihnya sendiri."""
    with _kunci_latih:
        if versi_aktif():
            return
        data = muat_data()
        if _daftarkan_dari_notebook(data):
            return
        v = data.loc[data["bagian"] == "validasi", "tanggal"]
        batas = v.max() if len(v) else core.BATAS_LATIH_AWAL
        folder = core.DIR_MODEL / "v1"
        ev = core.latih_semua(data, batas, folder, progres)
        _simpan_versi(
            1, batas, data["tanggal"].max(), folder, "Pelatihan awal (hiperparameter bawaan atau berkas penalaan)", ev
        )


def banding_versi(ev_baru, ev_lama):
    """MAE rata-rata per algoritma, versi lama vs baru, pada sel (stasiun, parameter, horizon, algoritma) yang sama."""
    kunci = ["stasiun", "parameter", "horizon", "algoritma"]
    b = ev_baru[ev_baru["algoritma"] != "Persistensi"]
    lama = ev_lama[ev_lama["algoritma"] != "Persistensi"]
    m = b.merge(lama, on=kunci, suffixes=("_baru", "_lama")).dropna(subset=["mae_baru", "mae_lama"])
    if m.empty:
        return pd.DataFrame(columns=["algoritma", "MAE lama", "MAE baru", "selisih", "sel", "tidak naik"])
    r = (
        m.groupby("algoritma")
        .agg(**{"MAE lama": ("mae_lama", "mean"), "MAE baru": ("mae_baru", "mean"), "sel": ("mae_baru", "size")})
        .reset_index()
    )
    r["selisih"] = r["MAE baru"] - r["MAE lama"]
    r["tidak naik"] = r["MAE baru"] <= r["MAE lama"]
    return r[["algoritma", "MAE lama", "MAE baru", "selisih", "sel", "tidak naik"]]


def aturan_aktivasi(ev_baru, ev_lama):
    """Versi baru aktif otomatis hanya bila MAE rata-rata kedua algoritma tidak naik."""
    r = banding_versi(ev_baru, ev_lama)
    if set(r["algoritma"]) != set(core.ALGORITMA):
        return (
            False,
            r,
            "Aturan aktivasi otomatis tidak dapat dinilai (sel pembanding tidak lengkap): menunggu keputusan Admin",
        )
    bagian = [
        f"{x['algoritma']} {x['MAE lama']:.3f} -> {x['MAE baru']:.3f} ({'tidak naik' if x['tidak naik'] else 'naik'})"
        for _, x in r.iterrows()
    ]
    lolos = bool(r["tidak naik"].all())
    return (
        lolos,
        r,
        ("MAE rata-rata " + "; ".join(bagian) + (": diaktifkan otomatis" if lolos else ": menunggu keputusan Admin")),
    )


def aktifkan_versi(vid, oleh):
    """Admin menyetujui versi yang menunggu keputusan, atau mengembalikan (rollback) ke versi lain yang sudah ada."""
    if STATUS["jalan"]:
        raise ValueError("Pelatihan ulang sedang berjalan; tunggu selesai sebelum mengganti versi aktif.")
    with _kunci, koneksi() as k:
        r = k.execute("SELECT * FROM versi_model WHERE id=?", (vid,)).fetchone()
        if r is None:
            raise ValueError("Versi model tidak ditemukan.")
        if not (BASE / r["lokasi"]).exists():
            raise ValueError("Folder berkas model versi ini tidak ditemukan; versi tidak dapat diaktifkan.")
        k.execute("UPDATE versi_model SET aktif=0, status=?", (STATUS_NONAKTIF,))
        k.execute(
            "UPDATE versi_model SET aktif=1, status=?, keterangan=COALESCE(keterangan,'') || ? WHERE id=?",
            (STATUS_AKTIF, f" | Diaktifkan oleh {oleh} pada {_sekarang()}", vid),
        )


def hari_data_baru(data, batas):
    """Jumlah hari data setelah batas latih versi aktif, per stasiun."""
    baru = data[data["tanggal"] > batas]
    return baru.groupby("stasiun").size().reindex(core.STASIUN, fill_value=0)


def prasyarat_latih_ulang():
    """Mengembalikan (boleh, pesan, tabel hari data baru per stasiun)."""
    akt = versi_aktif()
    hari = hari_data_baru(muat_data(), akt["data_hingga"])
    if hari.min() < core.MIN_HARI_BARU:
        return (
            False,
            (
                f"Data baru belum memadai: stasiun dengan data terpendek baru memiliki {int(hari.min())} hari, "
                f"batas minimum {core.MIN_HARI_BARU} hari. Pelatihan tidak dijalankan."
            ),
            hari,
        )
    return True, "Data baru memadai untuk pelatihan ulang.", hari


def latih_ulang(progres=None):
    """Melatih model baru dari data sampai data_hingga versi aktif; data sesudahnya menjadi segmen uji bersama."""
    with _kunci_latih:
        boleh, pesan, _ = prasyarat_latih_ulang()
        if not boleh:
            raise ValueError(pesan)
        akt, data = versi_aktif(), muat_data()
        batas = akt["data_hingga"]
        with koneksi() as k:
            nomor = k.execute("SELECT COALESCE(MAX(nomor),0)+1 FROM versi_model").fetchone()[0]
        folder = core.DIR_MODEL / f"v{nomor}"
        skala = (lambda f, t: progres(f * 0.9, t)) if progres else None
        ev_baru = core.latih_semua(data, batas, folder, skala)
        if progres:
            progres(0.92, "Mengevaluasi model lama pada segmen uji yang sama…")
        ev_lama = core.evaluasi_model_lama(data, akt["folder"], batas)
        lolos, _, ringkas = aturan_aktivasi(ev_baru, ev_lama)
        _simpan_versi(
            nomor,
            batas,
            data["tanggal"].max(),
            folder,
            f"Pelatihan ulang; segmen uji {batas + pd.Timedelta(days=1):%d-%m-%Y} sampai "
            f"{data['tanggal'].max():%d-%m-%Y}. {ringkas}",
            ev_baru,
            ev_lama,
            STATUS_AKTIF if lolos else STATUS_MENUNGGU,
        )
        if progres:
            progres(1.0, "Pelatihan ulang selesai.")


def mulai_latih_latar():
    """Menjalankan pelatihan ulang pada thread latar belakang. False bila sedang berjalan."""
    boleh, pesan, _ = prasyarat_latih_ulang()
    if not boleh:
        raise ValueError(pesan)
    with _kunci_status:
        if STATUS["jalan"]:
            return False
        STATUS.update(jalan=True, persen=0.0, pesan="Memulai pelatihan…", galat=None, selesai=False)

    def kerja():
        try:
            latih_ulang(lambda f, t: STATUS.update(persen=float(f), pesan=t))
        except Exception as e:
            STATUS["galat"] = str(e)
        finally:
            STATUS.update(jalan=False, selesai=True)

    threading.Thread(target=kerja, daemon=True).start()
    return True


def reset_sandi_admin_terminal(email=None):
    """Mengatur ulang kata sandi Admin dari terminal server: python ispu_sistem.py reset-admin [--email ...]."""
    import getpass

    init_db()
    with koneksi() as k:
        adm = k.execute("SELECT id, nama, email FROM pengguna WHERE peran='Admin' ORDER BY id").fetchall()
    if email:
        adm = [a for a in adm if a["email"] == email.strip().lower()]
    if not adm:
        print("Akun Admin tidak ditemukan.")
        return 1
    if len(adm) > 1:
        print("Ada beberapa akun Admin; tentukan dengan --email. Pilihan: " + ", ".join(a["email"] for a in adm))
        return 1
    a = adm[0]
    print(f"Mengatur ulang kata sandi Admin: {a['email']}")
    baru = getpass.getpass("Kata sandi baru: ")
    galat = periksa_sandi(baru, a["nama"], a["email"])
    if galat:
        print(galat)
        return 1
    if baru != getpass.getpass("Ulangi kata sandi baru: "):
        print("Konfirmasi tidak sama. Tidak ada yang diubah.")
        return 1
    with _kunci, koneksi() as k:
        k.execute(
            "UPDATE pengguna SET kata_sandi=?, wajib_ganti=0, gagal_login=0, terkunci_hingga=NULL WHERE id=?",
            (hash_sandi(baru), a["id"]),
        )
    print("Kata sandi Admin berhasil diatur ulang. Silakan masuk dengan kata sandi baru.")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Perintah pengelolaan untuk pemilik server Dasbor Prediksi ISPU.")
    sub = ap.add_subparsers(dest="perintah", required=True)
    p = sub.add_parser("reset-admin", help="atur ulang kata sandi akun Admin")
    p.add_argument("--email", help="alamat surel Admin (wajib bila ada lebih dari satu Admin)")
    arg = ap.parse_args()
    sys.exit(reset_sandi_admin_terminal(arg.email))
