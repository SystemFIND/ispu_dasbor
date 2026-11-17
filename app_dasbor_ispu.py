"""Dasbor Prediksi ISPU DKI Jakarta (Streamlit).

streamlit run app_dasbor_ispu.py
"""

import base64
import inspect
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

import ispu_core as core
import ispu_sistem as sis
from ispu_core import METEO, PARAM, STASIUN, WARNA, kategori, tgl_id

FOLDER_APP = Path(__file__).parent
FAVICON = FOLDER_APP / "favicon.png"

st.set_page_config(
    page_title="Dasbor Prediksi ISPU DKI Jakarta",
    page_icon=Image.open(FAVICON) if FAVICON.exists() else "🌫️",
    layout="wide",
)

# Warna algoritma sengaja berbeda dari warna kategori ISPU (hijau, biru, kuning, merah) agar tidak tertukar.
WARNA_ALG = {"Persistensi": "#9AA5B1", "XGBoost": "#C084FC", "LightGBM": "#FF8FB1"}
GARIS_DATA = "#E8ECEF"
WARNA_VARIABEL = ["#7CC4F5", "#F4A261", "#8BD3A0", "#E9C46A", "#C3A6F2", "#F28482", "#5FD3D0", "#F6BD60", "#B8C0FF",
                  "#E5989B"]

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
</style>
<style>
:root {
    --latar: #0E1621; --permukaan: #162231; --permukaan-2: #1C2B3D; --garis: rgba(255,255,255,.09);
    --teks: #E9EEF2; --teks-redup: #A3B1BF; --aksen: #6FA8B5; --aksen-hangat: #E0A83C;
}
html, body, .stApp, .stApp button, .stApp input, .stApp textarea, [data-testid="stMarkdownContainer"] {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}
.stApp {
    background:
        radial-gradient(1000px 460px at 100% -10%, rgba(111,168,181,.17), transparent 62%),
        radial-gradient(760px 420px at -8% 108%, rgba(224,168,60,.07), transparent 62%),
        var(--latar);
}
/* ---- warna teks eksplisit (tetap terbaca walau tema Streamlit terang) */
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"],
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-baseweb="tab"] { color: var(--teks); }
[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] { color: var(--teks-redup); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] * { color: var(--teks); }
footer, .stDeployButton { display: none !important; }

/* ---- tata letak dan tipografi */
.block-container { max-width: 1240px; padding: 2.4rem clamp(1rem, 4vw, 3rem) 4rem; }
h1, h2, h3, h4 { letter-spacing: -.02em; font-weight: 700; }
h2 { font-size: clamp(1.5rem, 2.6vw, 2rem); }
h4 { font-size: 1.05rem; color: var(--teks-redup); font-weight: 600; margin-top: 1.6rem; }

/* ---- sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(185deg, #1B2B3D 0%, #131E2C 60%, #0E1722 100%);
    border-right: 1px solid var(--garis);
}
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
/* Streamlit baru (stRadioOption) */
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    padding: 8px 12px; border-radius: 10px; width: 100%; transition: background .15s;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display: none; }
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background: rgba(255,255,255,.06); }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
    background: rgba(111,168,181,.20); box-shadow: inset 3px 0 0 var(--aksen-hangat);
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p { font-weight: 700; }
/* Streamlit lama (label baseweb) */
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    padding: 8px 12px; border-radius: 10px; width: 100%; transition: background .15s;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child { display: none; }
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover { background: rgba(255,255,255,.06); }
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
    background: rgba(111,168,181,.20); box-shadow: inset 3px 0 0 var(--aksen-hangat);
}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p { font-weight: 700; }

/* ---- kotak cara membaca di bawah grafik */
.baca { background: var(--permukaan); border: 1px solid var(--garis); border-left: 3px solid var(--aksen-hangat);
        border-radius: 10px; padding: 10px 14px 4px; margin: .2rem 0 1rem; font-size: .92rem; }
.baca .baca-judul { font-weight: 700; color: var(--aksen-hangat); margin-bottom: 4px; }
.baca ul { margin: 0 0 6px 1.1rem; padding: 0; }
.baca li { margin-bottom: 4px; color: var(--teks); }

/* ---- kartu parameter */
.kartu { background: var(--permukaan); border: 1px solid var(--garis); border-radius: 14px;
        padding: 14px 16px; margin-bottom: .6rem; }
.kartu .k-label { color: var(--teks-redup); font-size: .9rem; }
.kartu .k-nilai { font-size: clamp(1.5rem, 2.4vw, 2.1rem); font-weight: 700; margin: 2px 0 8px;
    font-variant-numeric: tabular-nums; }

/* ---- kartu metrik */
[data-testid="stMetric"] {
    background: var(--permukaan); border: 1px solid var(--garis); border-radius: 14px; padding: 14px 16px;
}
[data-testid="stMetricValue"] { font-size: clamp(1.5rem, 2.4vw, 2.1rem); font-weight: 700;
    font-variant-numeric: tabular-nums; }

/* ---- komponen input */
input, textarea, [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"] {
    background: var(--permukaan-2) !important; color: var(--teks) !important; border-color: var(--garis) !important;
}
[data-baseweb="select"] *, [data-baseweb="input"] * { color: var(--teks) !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: var(--permukaan-2) !important; }
[data-baseweb="popover"] li, [data-baseweb="menu"] li { color: var(--teks) !important; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    background: var(--permukaan-2); color: var(--teks); border: 1px solid rgba(111,168,181,.5);
    border-radius: 10px; font-weight: 600; padding: .45rem 1.1rem; transition: border-color .15s, color .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    border-color: var(--aksen-hangat); color: var(--aksen-hangat);
}
[data-testid="stExpander"] { border: 1px solid var(--garis); border-radius: 12px; background: rgba(255,255,255,.02); }
[data-baseweb="tab-list"] { gap: 6px; }
button:focus-visible, input:focus-visible { outline: 2px solid var(--aksen-hangat); outline-offset: 2px; }

/* ---- logo dan lencana */
.judul-logo { display:flex; align-items:center; gap:12px; margin: 4px 0 6px; }
.judul-logo img { width:46px; height:46px; border-radius:12px; box-shadow:0 4px 14px rgba(0,0,0,.35); }
.judul-logo span { font-size:1.35rem; font-weight:800; line-height:1.15; letter-spacing:-.02em; }
.lencana { display:inline-block; padding:2px 12px; border-radius:999px; font-weight:700;
           border:1px solid rgba(255,255,255,.35); }

/* ---- hero Beranda: warna latarnya mengikuti kategori udara */
.hero {
    display:grid; grid-template-columns: 1.5fr 1fr; gap: clamp(1rem, 3vw, 2.5rem); align-items:end;
    padding: clamp(1.4rem, 3.5vw, 2.6rem); border-radius: 22px; margin-bottom: 1.6rem;
    background: radial-gradient(520px 260px at 85% 0%, rgba(255,255,255,.14), transparent 70%),
                linear-gradient(135deg, var(--g1), var(--g2));
    border: 1px solid rgba(255,255,255,.14); box-shadow: 0 18px 50px rgba(0,0,0,.35);
}
.hero, .hero *:not(.lencana) { color: #fff !important; }
.hero-stasiun { font-size: clamp(1.1rem, 2.2vw, 1.5rem); font-weight: 700; }
.hero-tgl { opacity:.8; font-size:.95rem; margin-top:2px; }
.hero-angka { font-size: clamp(4rem, 11vw, 7.5rem); font-weight: 800; line-height: 1; letter-spacing:-.05em;
    margin: .6rem 0 .3rem; font-variant-numeric: tabular-nums; }
.hero-ket { opacity:.92; display:flex; flex-wrap:wrap; align-items:center; gap:.6rem; }
.hero .lencana { border: 2px solid rgba(255,255,255,.9); box-shadow: 0 2px 8px rgba(0,0,0,.3); }
.hero-sisi { display:flex; flex-direction:column; gap: 1rem; }
.fakta { background: rgba(0,0,0,.22); border: 1px solid rgba(255,255,255,.14); border-radius:14px; padding: 12px 16px; }
.fakta .f-label { display:block; opacity:.8; font-size:.85rem; }
.fakta .f-nilai { font-size: clamp(1.5rem, 3vw, 2.1rem); font-weight: 800; letter-spacing:-.02em;
    font-variant-numeric: tabular-nums; margin-right:.5rem; }

/* ---- responsif */
@media (max-width: 1000px) {
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: .75rem; }
    [data-testid="stColumn"] { flex: 1 1 calc(46% - .5rem) !important; min-width: calc(46% - .5rem) !important; }
}
@media (max-width: 760px) {
    .hero { grid-template-columns: 1fr; align-items:start; }
    .block-container { padding-top: 1.2rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _logo_b64():
    return base64.b64encode(FAVICON.read_bytes()).decode() if FAVICON.exists() else ""


def _judul_logo(teks):
    """Judul dengan ikon aplikasi di sampingnya (pengganti emoji)."""
    b = _logo_b64()
    gambar = f'<img src="data:image/png;base64,{b}" alt="">' if b else ""
    st.markdown(f'<div class="judul-logo">{gambar}<span>{teks}</span></div>', unsafe_allow_html=True)


def _lencana(kat):
    """Kategori ISPU sebagai lencana berwarna; teks gelap pada kuning agar tetap terbaca."""
    latar = WARNA.get(kat, "#555555")
    teks = "#1b1b1b" if kat == "Tidak Sehat" else "#ffffff"
    return f"<span class='lencana' style='background:{latar};color:{teks}'>{kat}</span>"


_GRADASI = {
    "Baik": ("#0C3B2E", "#1F7A58"),
    "Sedang": ("#0F2A4D", "#2A62B0"),
    "Tidak Sehat": ("#4A3708", "#8A6510"),
    "Sangat Tidak Sehat": ("#4A1616", "#A83A3A"),
    "Berbahaya": ("#0B0B10", "#34343F"),
}


def _hero(nama_stasiun, tanggal, ispu, kat, krit, besok, kat_besok):
    """Panel utama Beranda; latarnya berganti warna mengikuti kategori ISPU hari ini."""
    g1, g2 = _GRADASI.get(kat, ("#1C2B3D", "#2E4660"))
    st.markdown(
        f"<div class='hero' style='--g1:{g1};--g2:{g2}'>"
        f"<div><div class='hero-stasiun'>{nama_stasiun}</div>"
        f"<div class='hero-tgl'>Data terakhir {tanggal}</div>"
        f"<div class='hero-angka'>{ispu:.0f}</div>"
        f"<div class='hero-ket'>ISPU hari ini (nilai maksimum) {_lencana(kat)}</div></div>"
        "<div class='hero-sisi'>"
        f"<div class='fakta'><span class='f-label'>Parameter kritis</span><span class='f-nilai'>{krit}</span></div>"
        "<div class='fakta'><span class='f-label'>Prediksi besok (LightGBM)</span>"
        f"<span class='f-nilai'>{besok:.0f}</span>{_lencana(kat_besok)}</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _angka(nilai, desimal=1):
    """Angka dengan koma desimal dan tanda minus tipografis, sesuai penulisan bahasa Indonesia."""
    return f"{nilai:.{desimal}f}".replace(".", ",").replace("-", "−")


def _cara_baca(*poin, judul="Cara membaca"):
    """Kotak penjelasan cara membaca sebuah grafik atau tabel, ditulis untuk pembaca awam."""
    isi = "".join(f"<li>{x}</li>" for x in poin if x)
    st.markdown(f"<div class='baca'><div class='baca-judul'>{judul}</div><ul>{isi}</ul></div>", unsafe_allow_html=True)


def _lebar(fn):
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


def _kartu(kolom, label, nilai, kat=None):
    """Kartu ringkasan; kategori ISPU tampil sebagai lencana berwarna sesuai kategorinya."""
    lencana = _lencana(kat) if kat else ""
    kolom.markdown(
        f"<div class='kartu'><div class='k-label'>{label}</div><div class='k-nilai'>{nilai}</div>{lencana}</div>",
        unsafe_allow_html=True,
    )


def _tabel(df, **kw):
    st.dataframe(df, **_lebar(st.dataframe), **kw)


def _grafik(fig):
    st.plotly_chart(fig, **_lebar(st.plotly_chart))


def _csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# Pita kategori ISPU di latar grafik: batas kategori sama untuk keenam parameter karena ISPU sudah berupa indeks.
_PITA = [
    (0, 50, "Baik"),
    (50, 100, "Sedang"),
    (100, 200, "Tidak Sehat"),
    (200, 300, "Sangat Tidak Sehat"),
    (300, 500, "Berbahaya"),
]


def _pita_kategori(fig, ymax, row=None, col=None, label=True):
    """Menggambar pita warna kategori ISPU sampai ymax pada satu (sub)grafik."""
    # exclude_empty_subplots=False: tanpa ini Plotly melewati subgrafik yang belum berisi garis, sehingga pita
    # tidak tergambar bila dipanggil sebelum trace ditambahkan (seperti pada grafik Prediksi).
    kw = {} if row is None else dict(row=row, col=col, exclude_empty_subplots=False)
    for lo, hi, nama in _PITA:
        if lo >= ymax:
            break
        # Kepekatan 0,22 agar pita tetap terlihat di atas latar gelap; nama kategori ditulis di pojok kanan bawah pita.
        fig.add_hrect(
            y0=lo, y1=min(hi, ymax), fillcolor=WARNA[nama], opacity=0.22, line_width=0, layer="below",
            annotation_text=nama, annotation_position="inside bottom right",
            annotation_font=dict(size=10, color=WARNA[nama] if nama != "Berbahaya" else "#B0B0B0"), **kw,
        )
        if label and hi < 500 and hi <= ymax:
            fig.add_hline(
                y=hi,
                line=dict(color=WARNA[nama], width=1, dash="dot"),
                opacity=0.45,
                annotation_text=f"batas {nama} ({hi})",
                annotation_position="top left",
                annotation_font_size=10,
                **kw,
            )


def _rentang_kosong(tanggal, penanda):
    """Rentang tanggal berurutan tanpa data (penanda 9) -> daftar (awal, akhir)."""
    out, awal, sebelum = [], None, None
    for t, f in zip(tanggal, penanda):
        if f == 9 and awal is None:
            awal = t
        if f != 9 and awal is not None:
            out.append((awal, sebelum))
            awal = None
        sebelum = t
    if awal is not None:
        out.append((awal, sebelum))
    return out


def _gambar_aktual(fig, lp, nama_param, row=None, col=None, legenda=True):
    """Garis aktual: putus pada hari kosong, titik penuh = observasi asli, lingkaran kosong = interpolasi."""
    kw = {} if row is None else dict(row=row, col=col)
    y = lp["aktual"].where(lp["penanda"] != 9)
    fig.add_trace(
        go.Scatter(
            x=lp["tanggal_target"],
            y=y,
            mode="lines",
            name="Aktual (observasi)",
            connectgaps=False,
            line=dict(color=GARIS_DATA, width=2),
            showlegend=legenda,
            legendgroup="aktual",
        ),
        **kw,
    )
    asli, interp = lp[lp["penanda"] == 0], lp[lp["penanda"] == 1]
    fig.add_trace(
        go.Scatter(
            x=asli["tanggal_target"],
            y=asli["aktual"],
            mode="markers",
            name="Observasi asli",
            marker=dict(color=GARIS_DATA, size=5),
            showlegend=False,
            legendgroup="aktual",
            hovertemplate=f"%{{x|%d-%m-%Y}}<br>Aktual {nama_param}: %{{y:.0f}}<extra></extra>",
        ),
        **kw,
    )
    if len(interp):
        fig.add_trace(
            go.Scatter(
                x=interp["tanggal_target"],
                y=interp["aktual"],
                mode="markers",
                name="Hasil interpolasi (celah ≤ 3 hari)",
                showlegend=legenda,
                legendgroup="interp",
                marker=dict(color="rgba(0,0,0,0)", size=8, line=dict(color=GARIS_DATA, width=1.5)),
                hovertemplate="%{x|%d-%m-%Y}<br>Interpolasi: %{y:.0f}<extra></extra>",
            ),
            **kw,
        )
    for a, b in _rentang_kosong(lp["tanggal_target"], lp["penanda"]):
        fig.add_vrect(
            x0=a - pd.Timedelta(hours=12),
            x1=b + pd.Timedelta(hours=12),
            fillcolor="#9AA5B1",
            opacity=0.22,
            line_width=0,
            layer="below",
            **kw,
        )


@st.cache_resource(show_spinner=False)
def _init():
    sis.init_db()
    return True


@st.cache_data(show_spinner=False)
def data_historis(token):
    """token berubah setiap data bertambah sehingga cache otomatis diperbarui."""
    return sis.muat_data()


@st.cache_resource(show_spinner=False, max_entries=10)
def model_stasiun(folder, kode):
    return core.muat_model(folder, kode)


@st.cache_data(show_spinner=False)
def prediksi_cache(folder, kode, token):
    g = data_historis(token)
    return core.prediksi_terbaru(g[g["stasiun"] == kode], model_stasiun(folder, kode))


@st.cache_data(show_spinner=False)
def lampau_cache(folder, kode, p, h, mulai, akhir, token):
    g = data_historis(token)
    return core.prediksi_lampau(g[g["stasiun"] == kode], model_stasiun(folder, kode), p, h, mulai, akhir)


@st.cache_data(show_spinner=False)
def uji_cache(folder, kode, p, h, batas, token):
    g = data_historis(token)
    return core.prediksi_uji(g[g["stasiun"] == kode], model_stasiun(folder, kode), p, h, batas)


_init()
if not sis.ada_data():
    st.error(
        "Himpunan data belum tersedia. Letakkan **dataset_final_2021_2025_A.csv** di folder yang sama dengan "
        "app_dasbor_ispu.py, lalu muat ulang halaman ini."
    )
    st.stop()
if sis.versi_aktif() is None:
    _judul_logo("Menyiapkan model awal")
    st.info(
        "Pertama kali dijalankan, sistem mendaftarkan model awal dari notebook (folder models/v1). Bila folder itu "
        "tidak ada, sistem melatih 420 model sendiri (5 stasiun × 6 parameter × 7 horizon × 2 algoritma), yang "
        "memerlukan beberapa menit dan hanya terjadi sekali."
    )
    bar = st.progress(0.0, text="Memulai…")
    try:
        sis.latih_awal(lambda f, t: bar.progress(min(max(float(f), 0.0), 1.0), text=t))
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.rerun()

pengguna = st.session_state.get("pengguna")
pesan_sesi = None
if pengguna:
    segar = sis.muat_pengguna(pengguna["id"])  # akun bisa dinonaktifkan atau direset Admin saat sesi berjalan
    if segar is None:
        for k in ("pengguna", "hasil_unggahan", "tampil_historis", "tampil_prediksi"):
            st.session_state.pop(k, None)
        pengguna, pesan_sesi = None, "Sesi berakhir karena akun Anda sudah tidak aktif. Hubungi Admin."
    else:
        pengguna = st.session_state["pengguna"] = segar
if pengguna and pengguna.get("wajib_ganti"):
    st.title("🔐 Ganti kata sandi")
    st.warning(
        "Kata sandi akun ini bersifat sementara (bawaan atau hasil reset Admin). Ganti kata sandi sebelum melanjutkan."
    )
    with st.form("form_ganti_sandi"):
        sandi_lama = st.text_input("Kata sandi saat ini", type="password")
        sandi_baru = st.text_input("Kata sandi baru (minimal 8 karakter, memuat huruf dan angka)", type="password")
        sandi_ulang = st.text_input("Ulangi kata sandi baru", type="password")
        kirim = st.form_submit_button("Simpan kata sandi baru")
    if kirim:
        ok, pesan = sis.ganti_kata_sandi(pengguna["id"], sandi_lama, sandi_baru, sandi_ulang)
        if ok:
            st.session_state["pengguna"] = {**pengguna, "wajib_ganti": False}
            st.rerun()
        st.error(pesan)
    if st.button("Keluar"):
        st.session_state.pop("pengguna", None)
        st.rerun()
    st.stop()
peran = pengguna["peran"] if pengguna else None
token = sis.token_data()
data = data_historis(token)
akt = sis.versi_aktif()

MENU_PUBLIK = [
    "Beranda",
    "Data Historis",
    "Prediksi Multi-Parameter",
    "Perbandingan Metode",
    "Evaluasi Model",
    "Tanya Jawab",
]
MENU_PERAN = {
    None: ["Masuk / Daftar"],
    "User": ["Unggah Data untuk Prediksi", "Unduh Hasil", "Profil"],
    "Admin": ["Kelola Data", "Pelatihan Ulang Model", "Kelola Pengguna", "Profil"],
}

with st.sidebar:
    _judul_logo("Dasbor Prediksi ISPU")
    st.caption("DKI Jakarta · XGBoost dan LightGBM")
    if pengguna:
        st.success(f"{pengguna['nama']} · {peran}")
        if st.button("Keluar"):
            for k in ("pengguna", "hasil_unggahan", "tampil_historis", "tampil_prediksi"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        if pesan_sesi:
            st.warning(pesan_sesi)
        st.caption("Anda mengakses sebagai Pengunjung.")
    halaman = st.radio("Menu", MENU_PUBLIK + MENU_PERAN[peran])
    stasiun = st.selectbox("Stasiun pemantauan", list(STASIUN), format_func=STASIUN.get)
    st.divider()
    st.caption(
        f"Model versi {akt['nomor']} · data hingga {tgl_id(akt['data_hingga'])}. "
        "Data: Satu Data Jakarta dan NASA POWER."
    )

g = data[data["stasiun"] == stasiun]
FOLDER = str(akt["folder"])


def _wajib(peran_wajib):
    """Pembatasan hak akses (Tabel 3.5), berlaku walaupun halaman dipanggil langsung."""
    if peran != peran_wajib:
        st.warning(f"Halaman ini hanya dapat diakses oleh {peran_wajib}. Silakan masuk dengan akun yang sesuai.")
        st.stop()


def _galat(daftar_galat, judul="Berkas ditolak:"):
    st.error(judul)
    for e in daftar_galat:
        st.markdown(f"- {e}")


def _baca_csv(berkas):
    try:
        return pd.read_csv(berkas)
    except Exception as e:
        st.error(f"Berkas tidak dapat dibaca sebagai CSV: {e}")
        return None


def beranda():
    pr, x, tgl = prediksi_cache(FOLDER, stasiun, token)
    terakhir = g[g["tanggal"] == tgl].iloc[0]
    nilai = {p: terakhir[p] for p in PARAM}
    ispu = np.nanmax(list(nilai.values()))
    krit = max(nilai, key=lambda k: -1 if pd.isna(nilai[k]) else nilai[k])
    besok = pr[(pr.algoritma == "LightGBM") & (pr.horizon == 1)]
    maks_besok = besok.nilai.max()
    _hero(STASIUN[stasiun], tgl_id(tgl), ispu, kategori(ispu), PARAM[krit], maks_besok, kategori(maks_besok))
    st.markdown("#### Nilai enam parameter pada hari terakhir")
    for k, (p, nm) in zip(st.columns(6), PARAM.items()):
        _kartu(k, nm, "-" if pd.isna(nilai[p]) else f"{nilai[p]:.0f}", kategori(nilai[p]))
    _cara_baca(
        "Angka pada kartu adalah indeks ISPU setiap polutan pada hari terakhir. <b>Semakin kecil semakin baik</b>: "
        "0–50 Baik, 51–100 Sedang, 101–200 Tidak Sehat, 201–300 Sangat Tidak Sehat, di atas 300 Berbahaya.",
        f"ISPU hari itu (angka besar di panel atas) diambil dari polutan dengan angka tertinggi, yaitu "
        f"<b>{PARAM[krit]}</b>. Polutan inilah yang paling menentukan kualitas udara hari itu.",
    )
    st.markdown("#### Nilai aktual dan nilai prediksi")
    c = st.columns([1.3, 1, 1])
    mode = c[0].radio("Tampilan", ["Satu parameter", "Keenam parameter"], horizontal=True, key="b_mode")
    n_hari = c[1].selectbox("Rentang", [30, 60, 90], format_func=lambda v: f"{v} hari terakhir", key="b_hari")
    alg_b = c[2].selectbox("Algoritma prediksi", list(core.ALGORITMA) + ["Keduanya"], key="b_alg")
    algs = list(core.ALGORITMA) if alg_b == "Keduanya" else [alg_b]
    mulai = tgl - pd.Timedelta(days=n_hari - 1)
    tgl_pred = [tgl + pd.Timedelta(days=h) for h in core.HORIZON]

    def _satu(fig, p, row=None, col=None, legenda=True):
        kw = {} if row is None else dict(row=row, col=col)
        lp = lampau_cache(FOLDER, stasiun, p, 1, mulai, tgl, token)
        _gambar_aktual(fig, lp, PARAM[p], row, col, legenda)
        for a in algs:
            fig.add_trace(
                go.Scatter(
                    x=lp["tanggal_target"],
                    y=lp[a],
                    mode="lines",
                    name=f"Prediksi H+1 {a} (lampau)",
                    line=dict(color=WARNA_ALG[a], dash="dash", width=1.6),
                    showlegend=legenda,
                    legendgroup=f"lampau{a}",
                    hovertemplate=f"%{{x|%d-%m-%Y}}<br>Prediksi {a}: %{{y:.0f}}<extra></extra>",
                ),
                **kw,
            )
            s_ = pr[(pr.algoritma == a) & (pr.kode == p)].sort_values("horizon")
            akhir = lp["aktual"].where(lp["penanda"] != 9).iloc[-1] if len(lp) else np.nan
            xs = ([tgl] if pd.notna(akhir) else []) + tgl_pred
            ys = ([akhir] if pd.notna(akhir) else []) + list(s_.nilai)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=f"Prediksi H+1–H+7 {a} (ke depan)",
                    line=dict(color=WARNA_ALG[a], dash="dot", width=2),
                    marker=dict(size=7, symbol="diamond"),
                    showlegend=legenda,
                    legendgroup=f"depan{a}",
                    hovertemplate=f"%{{x|%d-%m-%Y}}<br>Prediksi {a}: %{{y:.0f}}<extra></extra>",
                ),
                **kw,
            )
        nilai_maks = np.nanmax(
            np.r_[
                lp["aktual"].to_numpy(float),
                lp[algs].to_numpy(float).ravel(),
                pr[pr.kode == p].nilai.to_numpy(float),
                50.0,
            ]
        )
        _pita_kategori(fig, max(60.0, nilai_maks * 1.15), row, col, label=row is None)
        fig.add_vline(x=tgl, line=dict(color="#E0A83C", width=1, dash="dash"), **kw)
        return lp

    if mode == "Satu parameter":
        parameter = st.selectbox("Parameter", list(PARAM), format_func=PARAM.get, key="b_param")
        fig = go.Figure()
        lp = _satu(fig, parameter)
        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.18),
            xaxis_title="Tanggal",
            yaxis_title=f"Indeks ISPU {PARAM[parameter]}",
        )
        _grafik(fig)
        ok = lp["penanda"] == 0
        if ok.any():
            selisih = {a: np.abs(lp.loc[ok, a] - lp.loc[ok, "aktual"]).mean() for a in algs}
            teks = "; ".join(f"{a} rata-rata selisih {_angka(v)} poin" for a, v in selisih.items())
            st.caption(f"Pada {int(ok.sum())} hari berobservasi asli dalam rentang ini, prediksi H+1: {teks}.")
    else:
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[f"ISPU {n}" for n in PARAM.values()],
            vertical_spacing=0.09,
            horizontal_spacing=0.07,
        )
        for i, p in enumerate(PARAM):
            _satu(fig, p, i // 2 + 1, i % 2 + 1, legenda=(i == 0))
        fig.update_layout(height=860, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.06))
        fig.update_yaxes(title_text="Indeks ISPU")
        _grafik(fig)
    sampel = (
        "termasuk periode data latih"
        if mulai <= pd.Timestamp(akt["batas_latih"])
        else "di luar data latih model (tidak dipakai saat melatih)"
    )
    st.caption(
        "Garis putih: nilai aktual. Titik penuh: observasi asli; lingkaran kosong: hasil interpolasi celah ≤ 3 hari."
        " **Area abu-abu: tidak ada data** pada hari itu (alat tidak mengirim data atau data dibuang saat penjaminan"
        " mutu), sehingga garis aktual sengaja dibiarkan terputus dan tidak disambung. Garis putus-putus: prediksi H+1"
        f" yang dibuat sehari sebelumnya untuk setiap tanggal ({sampel}). Garis titik-titik dengan wajik: prediksi"
        " tujuh hari ke depan, disambungkan dari nilai aktual terakhir. Pita warna latar: kategori ISPU."
    )
    _cara_baca(
        "<b>Garis putih naik berarti udara memburuk</b>, turun berarti membaik. Bila garis masuk pita kuning atau "
        "merah, udara pada hari itu tidak sehat.",
        "<b>Prediksi yang baik</b> adalah garis putus-putus yang berimpit dengan garis putih. Jarak tegak di antara "
        "keduanya adalah besar kesalahan prediksi pada hari itu; angka \"rata-rata selisih\" di atas grafik "
        "merangkumnya (semakin kecil semakin tepat).",
        "Garis tegak jingga menandai tanggal data terakhir. Di sebelah kanannya adalah prediksi tujuh hari ke depan; "
        "semakin jauh harinya, semakin kurang pasti prediksinya.",
    )


def _arti_korelasi(r):
    if pd.isna(r):
        return "tidak dapat dihitung"
    kuat = "kuat" if abs(r) >= 0.7 else "sedang" if abs(r) >= 0.4 else "lemah" if abs(r) >= 0.2 else "hampir tidak ada"
    if kuat == "hampir tidak ada":
        return "hampir tidak ada hubungan"
    arah = "searah (naik bersama, turun bersama)" if r > 0 else "berlawanan (satu naik, yang lain turun)"
    return f"hubungan {kuat} {arah}"


def _korelasi(d, pilih, semua):
    """Matriks korelasi Pearson antarvariabel terpilih, disertai penjelasan cara membacanya."""
    kor = d[pilih].corr()
    nama = [semua[v] for v in pilih]
    hm = go.Figure(
        go.Heatmap(
            z=kor.values, x=nama, y=nama, zmin=-1, zmax=1, zmid=0, colorscale="RdBu",
            text=[[_angka(v, 2) if pd.notna(v) else "" for v in baris] for baris in kor.values], texttemplate="%{text}",
            hovertemplate="%{y} dan %{x}: %{z:.2f}<extra></extra>",
        )
    )
    hm.update_layout(height=max(320, 90 + 60 * len(pilih)), margin=dict(l=10, r=10, t=10, b=10))
    _grafik(hm)
    pasangan = sorted(
        ((kor.iloc[i, j], nama[j], nama[i]) for i in range(len(pilih)) for j in range(i)),
        key=lambda t: -abs(t[0]) if pd.notna(t[0]) else 0,
    )
    _cara_baca(
        "Setiap kotak berisi angka korelasi (r) antara variabel di bawahnya dan variabel di sebelah kirinya, dari −1 "
        "sampai 1.",
        "Mendekati <b>1</b> (biru tua): kedua variabel naik dan turun bersama. Mendekati <b>−1</b> (merah tua): saat "
        "satu naik, yang lain turun. Sekitar <b>0</b> (putih): tidak ada pola bersama. Patokan kasar: di atas 0,7 "
        "kuat, 0,4–0,7 sedang, 0,2–0,4 lemah.",
        "Kotak diagonal selalu bernilai 1 karena membandingkan variabel dengan dirinya sendiri, dan kotak di atas "
        "diagonal sama dengan kotak di bawahnya (cerminan). Jadi setiap pasangan cukup dibaca sekali.",
        "Korelasi hanya menunjukkan pola yang bergerak bersama, <b>bukan sebab-akibat</b>.",
    )
    _cara_baca(
        *[f"{a} dan {b}: r = {_angka(r, 2)}, {_arti_korelasi(r)}." for r, a, b in pasangan[:3]],
        judul="Pada rentang terpilih",
    )
    st.caption("Dihitung dari hari yang memiliki nilai untuk kedua variabel.")


def halaman_data_historis():
    st.header("Data Historis")
    st.subheader(STASIUN[stasiun])
    tmin, tmax = g["tanggal"].min().date(), g["tanggal"].max().date()
    semua = {**{p: f"ISPU {n}" for p, n in PARAM.items()}, **METEO}
    c = st.columns([2, 3])
    rentang = c[0].date_input(
        "Rentang tanggal", value=(max(tmin, tmax - timedelta(days=364)), tmax), min_value=tmin, max_value=tmax
    )
    pilih = c[1].multiselect(
        "Variabel (satu atau beberapa)", list(semua), default=["pm25", "pm10"], format_func=semua.get
    )
    if not isinstance(rentang, (tuple, list)) or len(rentang) < 2:
        st.info("Pilih tanggal awal dan tanggal akhir pada kalender.")
        return
    if not pilih:
        st.info("Pilih minimal satu variabel.")
        return
    d = g[(g["tanggal"] >= pd.Timestamp(rentang[0])) & (g["tanggal"] <= pd.Timestamp(rentang[1]))]
    # Satu grafik per variabel dengan sumbu waktu yang sama: setiap variabel memiliki satuan dan rentang nilai sendiri,
    # sehingga ditumpuk dalam satu grafik justru menyulitkan pembacaan.
    n = len(pilih)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True, vertical_spacing=min(0.06, 0.25 / n),
        subplot_titles=[semua[v] for v in pilih],
    )
    for i, v in enumerate(pilih, start=1):
        fig.add_trace(
            go.Scatter(
                x=d["tanggal"], y=d[v], mode="lines", name=semua[v], showlegend=False,
                line=dict(width=1.3, color=WARNA_VARIABEL[(i - 1) % len(WARNA_VARIABEL)]),
                hovertemplate="%{x|%d-%m-%Y}: %{y:.1f}<extra>" + semua[v] + "</extra>",
            ),
            row=i, col=1,
        )
        fig.update_yaxes(title_text="Indeks" if v in PARAM else semua[v].split("(")[-1].rstrip(")"), row=i, col=1,
                         title_font=dict(size=11))
    for a in fig.layout.annotations:
        a.update(x=0, xanchor="left", font=dict(size=13))
    fig.update_layout(height=150 + 190 * n, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
    _grafik(fig)
    st.caption(
        "Setiap variabel digambar pada grafiknya sendiri karena satuan dan rentang nilainya berbeda; sumbu tanggal "
        "dibuat sama sehingga lonjakan pada tanggal yang sama dapat dibandingkan dari atas ke bawah. Memperbesar satu "
        "grafik akan ikut memperbesar grafik lainnya. Garis yang terputus menandakan nilai kosong, termasuk periode "
        "Januari sampai November 2022 yang secara struktural hampir tidak memiliki observasi."
    )
    _cara_baca(
        "Garis naik berarti nilainya meningkat. Untuk keenam polutan ISPU, <b>semakin tinggi semakin buruk</b> "
        "kualitas udaranya (di atas 100 berarti Tidak Sehat).",
        "Untuk suhu, kelembapan, angin, dan hujan tidak ada nilai baik atau buruk; variabel ini dipakai untuk melihat "
        "kondisi cuaca. Hujan dan angin yang kuat umumnya membantu menurunkan polutan, sehingga coba bandingkan "
        "puncak hujan dengan turunnya PM2,5 pada tanggal yang sama.",
    )
    if n >= 2:
        with st.expander("Hubungan antarvariabel (korelasi)"):
            _korelasi(d, pilih, semua)
    tabel = d[["tanggal", "stasiun"] + pilih].copy()
    st.session_state["tampil_historis"] = tabel.assign(tanggal=tabel["tanggal"].dt.strftime("%Y-%m-%d"))
    with st.expander("Tabel data pada rentang terpilih"):
        _tabel(tabel.assign(tanggal=tabel["tanggal"].dt.strftime("%d-%m-%Y")).set_index("tanggal"))


def _tampil_prediksi(pr, tgl, x, kunci, state, meta):
    """Tabel, grafik, dan rincian fitur; dipakai oleh modul Prediksi dan modul Unggah Data."""
    alg = st.radio("Algoritma", core.ALGORITMA, horizontal=True, key=f"alg_{kunci}")
    tabel = (
        pr[pr.algoritma == alg]
        .pivot(index="horizon", columns="parameter", values="nilai")[list(PARAM.values())]
        .round(1)
    )
    tabel.insert(0, "Tanggal", [f"{tgl + pd.Timedelta(days=h):%d-%m-%Y}" for h in tabel.index])
    tabel["ISPU maks."] = tabel[list(PARAM.values())].max(axis=1)
    tabel["Kategori"] = tabel["ISPU maks."].map(kategori)
    tabel.index = [f"H+{h}" for h in tabel.index]
    st.markdown("#### Tabel prediksi")
    _tabel(tabel)
    _cara_baca(
        "Setiap baris adalah satu hari ke depan: H+1 besok (dihitung dari data terakhir), H+7 tujuh hari lagi. Kolom "
        "polutan berisi perkiraan indeks ISPU; <b>semakin kecil semakin baik</b>.",
        "Kolom <b>ISPU maks.</b> adalah angka tertinggi dari keenam polutan pada hari itu, dan kolom Kategori "
        "diturunkan dari angka itu. Kategori inilah yang dipakai untuk menyebut kualitas udara hari tersebut.",
    )
    st.session_state[state] = tabel.reset_index(names="Horizon").assign(Algoritma=alg, **meta)
    st.markdown("#### Grafik prediksi per polutan")
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[f"ISPU {n}" for n in PARAM.values()],
        vertical_spacing=0.2,
        horizontal_spacing=0.06,
    )
    label_x = [f"H+{h}<br>{tgl + pd.Timedelta(days=h):%d-%m}" for h in core.HORIZON]
    for i, (p, nm) in enumerate(PARAM.items()):
        r, c = i // 3 + 1, i % 3 + 1
        s = pr[(pr.algoritma == alg) & (pr.kode == p)].sort_values("horizon")
        ymax = max(60.0, float(np.nanmax(np.r_[s.nilai.to_numpy(float), 0.0])) * 1.25)
        _pita_kategori(fig, ymax, r, c, label=False)
        fig.add_trace(
            go.Scatter(
                x=label_x,
                y=s.nilai,
                mode="lines+markers+text",
                name=nm,
                showlegend=False,
                text=[f"{v:.0f}" for v in s.nilai],
                textposition="top center",
                textfont=dict(size=10),
                line=dict(color=WARNA_ALG[alg], width=2),
                hovertemplate=f"{nm} %{{x}}: %{{y:.1f}}<extra></extra>",
            ),
            row=r,
            col=c,
        )
        fig.update_yaxes(range=[0, ymax], row=r, col=c)
    fig.update_xaxes(title_text="Horizon (hari ke depan)", title_font=dict(size=11), tickfont=dict(size=10))
    fig.update_yaxes(title_text="Indeks ISPU", title_font=dict(size=11))
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=70, b=10))
    _grafik(fig)
    st.caption(
        "Setiap polutan ditampilkan pada grafiknya sendiri. Sumbu mendatar adalah horizon prediksi: H+1 berarti satu "
        "hari setelah data terakhir, H+7 tujuh hari setelahnya (tanggal di bawahnya). Warna latar menunjukkan kategori "
        "ISPU (hijau Baik 0–50, biru Sedang 51–100, kuning Tidak Sehat 101–200, dan seterusnya); batas kategori sama "
        "untuk semua polutan karena ISPU sudah merupakan indeks hasil konversi konsentrasi masing-masing polutan."
    )
    sa = pr[pr.algoritma == alg]
    puncak = sa.loc[sa.nilai.idxmax()] if sa.nilai.notna().any() else None
    if puncak is None:
        kalimat_puncak = ""
    elif puncak.nilai > 100:
        nama_puncak = PARAM.get(puncak.kode, puncak.kode)
        kalimat_puncak = (
            f"Pada prediksi {alg} ini <b>{nama_puncak} masuk pita kuning (Tidak Sehat)</b> dengan nilai tertinggi "
            f"{puncak.nilai:.0f} pada H+{int(puncak.horizon)}; polutan inilah yang perlu diwaspadai."
        )
    else:
        nama_puncak = PARAM.get(puncak.kode, puncak.kode)
        kalimat_puncak = (
            f"Pada prediksi {alg} ini nilai tertinggi adalah {nama_puncak} {puncak.nilai:.0f} "
            f"(H+{int(puncak.horizon)}), masih di bawah 100, sehingga <b>tidak ada polutan yang diperkirakan Tidak "
            "Sehat</b> dan pita kuning tidak tampil. Pita kuning hanya muncul bila ada prediksi yang cukup tinggi, "
            "karena tinggi sumbu menyesuaikan nilai prediksi."
        )
    _cara_baca(
        "Angka di atas setiap titik adalah perkiraan ISPU hari itu. Garis yang <b>naik ke kanan berarti udara "
        "diperkirakan memburuk</b>, turun berarti membaik.",
        "Warna latar grafik adalah pita kategori, dan namanya tertulis di pojok kanan bawah setiap pita: <b>hijau Baik "
        "(0–50)</b>, biru Sedang (51–100), kuning Tidak Sehat (101–200). Titik yang berada di pita hijau berarti "
        "udara diperkirakan Baik.",
        kalimat_puncak,
        "H+1 adalah prediksi yang paling dapat dipercaya; semakin ke kanan (H+7) semakin kurang pasti.",
    )
    with st.expander("Rincian nilai fitur masukan model (hari t dan sebelumnya)"):
        _tabel(x.T.rename(columns={x.index[0]: "nilai"}).round(2))


def halaman_prediksi():
    st.header("Prediksi Multi-Parameter")
    pr, x, tgl = prediksi_cache(FOLDER, stasiun, token)
    st.subheader(
        f"{STASIUN[stasiun]} · prediksi untuk {tgl_id(tgl + pd.Timedelta(days=1))} "
        f"sampai {tgl_id(tgl + pd.Timedelta(days=7))}"
    )
    _tampil_prediksi(
        pr,
        tgl,
        x,
        "prediksi",
        "tampil_prediksi",
        dict(Stasiun=stasiun, Data_terakhir=tgl.strftime("%Y-%m-%d"), Versi_model=akt["nomor"]),
    )
    st.caption(
        "Kategori diturunkan dari nilai indeks maksimum pada setiap hari prediksi. Prediksi bersifat perkiraan "
        "dan tidak menggantikan pemantauan resmi."
    )


def halaman_perbandingan():
    st.header("Perbandingan Metode")
    st.subheader(f"{STASIUN[stasiun]} · periode uji setelah {tgl_id(akt['batas_latih'])}")
    st.caption(
        "Nilai aktual dibandingkan dengan prediksi XGBoost dan LightGBM pada periode uji, yaitu periode yang tidak "
        "dipakai untuk melatih model. Sumbu mendatar adalah tanggal yang diprediksi (tanggal dasar + horizon)."
    )
    t1, t2 = st.tabs(["Satu parameter", "Per polutan (keenam parameter)"])
    with t1:
        c = st.columns(2)
        p = c[0].selectbox("Parameter", list(PARAM), format_func=PARAM.get, key="pb_param")
        h = c[1].selectbox(
            "Horizon", list(core.HORIZON), format_func=lambda v: f"H+{v} ({v} hari ke depan)", key="pb_h"
        )
        pu = uji_cache(FOLDER, stasiun, p, h, str(akt["batas_latih"]), token)
        if pu.empty:
            st.info("Belum ada data uji untuk kombinasi ini.")
        else:
            fig = go.Figure()
            fig.add_scatter(
                x=pu.tanggal_target,
                y=pu.aktual,
                mode="lines+markers",
                name="Aktual",
                line=dict(color=GARIS_DATA),
                marker=dict(size=4),
            )
            for a in core.ALGORITMA:
                fig.add_scatter(
                    x=pu.tanggal_target,
                    y=pu[a],
                    mode="lines",
                    name=f"Prediksi {a}",
                    line=dict(color=WARNA_ALG[a], dash="dash"),
                )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title=f"Indeks ISPU {PARAM[p]}",
                xaxis_title=f"Tanggal yang diprediksi (H+{h})",
                legend=dict(orientation="h"),
                xaxis=dict(rangeslider=dict(visible=True)),
            )
            _grafik(fig)
            # Sama dengan Evaluasi Model: hanya hari dengan observasi asli dan prediksi tersedia (nilai hari t ada).
            ok = pu.aktual.notna() & pu[list(core.ALGORITMA)].notna().all(axis=1)
            baris = []
            for a in core.ALGORITMA:
                m = core.metrik(pu.aktual[ok], pu[a][ok]) if ok.sum() else dict(mae=np.nan, rmse=np.nan, r2=np.nan)
                baris.append(
                    {"Metode": a, "Jumlah observasi": int(ok.sum()), "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"]}
                )
            _tabel(pd.DataFrame(baris).round(3).set_index("Metode"))
            st.caption(
                "Garis aktual terputus pada hari tanpa observasi asli. Ukuran kesalahan hanya dihitung pada observasi"
                " asli."
            )
            bm = pd.DataFrame(baris).set_index("Metode")
            hasil = ""
            if bm["MAE"].notna().all():
                unggul = bm["MAE"].idxmin()
                kalah = bm["MAE"].idxmax()
                ma, mb = (_angka(bm.loc[x, "MAE"]) for x in (unggul, kalah))
                # Selisih < 0,5 poin hilang saat ISPU dibulatkan ke bilangan bulat, sehingga kedua model disebut setara.
                if bm.loc[kalah, "MAE"] - bm.loc[unggul, "MAE"] < 0.5:
                    rata = _angka(bm["MAE"].mean())
                    hasil = (
                        "Pada pilihan ini, XGBoost dan LightGBM <b>hampir sama tepatnya</b>: keduanya rata-rata "
                        f"meleset sekitar {rata} poin ISPU."
                    )
                else:
                    hasil = (
                        f"Pada pilihan ini, <b>{unggul} lebih tepat</b>: rata-rata meleset {ma} poin, "
                        f"sedangkan {kalah} {mb} poin."
                    )
            _cara_baca(
                "Garis putih adalah nilai yang benar-benar terjadi. <b>Prediksi yang baik garisnya menempel</b> dan "
                "mengikuti naik-turun garis putih. Geser bilah kecil di bawah grafik untuk memilih periode tertentu.",
                "<b>MAE</b> adalah rata-rata selisih prediksi dengan nilai aktual, dalam poin ISPU, dan <b>RMSE</b> "
                "mirip tetapi lebih menghukum selisih yang besar. Untuk keduanya, <b>semakin kecil semakin baik</b>.",
                "<b>R²</b> menunjukkan seberapa baik prediksi mengikuti naik-turun data: 1 sempurna, 0 sama saja "
                "dengan menebak rata-rata, negatif lebih buruk daripada menebak rata-rata. <b>Semakin besar semakin "
                "baik</b>.",
                hasil,
            )
    with t2:
        h2 = st.selectbox(
            "Horizon", list(core.HORIZON), format_func=lambda v: f"H+{v} ({v} hari ke depan)", key="pb_h2"
        )
        fig = make_subplots(
            rows=3,
            cols=2,
            vertical_spacing=0.12,
            horizontal_spacing=0.07,
            subplot_titles=[f"ISPU {n}" for n in PARAM.values()],
        )
        judul = []
        for i, (p, nm) in enumerate(PARAM.items()):
            r, c = i // 2 + 1, i % 2 + 1
            pu = uji_cache(FOLDER, stasiun, p, h2, str(akt["batas_latih"]), token)
            if pu.empty:
                judul.append(f"ISPU {nm}")
                continue
            fig.add_trace(
                go.Scatter(
                    x=pu.tanggal_target,
                    y=pu.aktual,
                    mode="lines",
                    name="Aktual",
                    showlegend=(i == 0),
                    line=dict(color=GARIS_DATA, width=1.4),
                    legendgroup="a",
                ),
                row=r,
                col=c,
            )
            for a in core.ALGORITMA:
                fig.add_trace(
                    go.Scatter(
                        x=pu.tanggal_target,
                        y=pu[a],
                        mode="lines",
                        name=f"Prediksi {a}",
                        showlegend=(i == 0),
                        legendgroup=a,
                        line=dict(color=WARNA_ALG[a], dash="dash", width=1.2),
                    ),
                    row=r,
                    col=c,
                )
            # Sama dengan Evaluasi Model: hanya hari dengan observasi asli dan prediksi tersedia (nilai hari t ada).
            ok = pu.aktual.notna() & pu[list(core.ALGORITMA)].notna().all(axis=1)
            m = {a: core.metrik(pu.aktual[ok], pu[a][ok]) for a in core.ALGORITMA} if ok.sum() else {}
            judul.append(
                f"ISPU {nm} · R² XGB {_angka(m['XGBoost']['r2'], 2)}, LGBM {_angka(m['LightGBM']['r2'], 2)}"
                if m
                else f"ISPU {nm}"
            )
        for k, a in enumerate(fig.layout.annotations[: len(judul)]):
            a.text = judul[k]
        fig.update_yaxes(title_text="Indeks ISPU", title_font=dict(size=11))
        # Judul sumbu mendatar tidak diulang di tiap subgrafik agar tidak menabrak judul baris di bawahnya;
        # margin atas memberi ruang bagi bilah ikon grafik.
        fig.update_layout(height=1100, margin=dict(l=10, r=10, t=80, b=40), legend=dict(orientation="h", y=-0.04))
        _grafik(fig)
        st.caption(
            f"Sumbu mendatar setiap grafik adalah tanggal yang diprediksi (tanggal dasar + {h2} hari). Setiap polutan "
            "ditampilkan terpisah karena pola dan rentang nilainya berbeda. R² pada judul tiap grafik dihitung dari "
            "observasi asli pada periode uji."
        )
        _cara_baca(
            "Seperti tab sebelumnya: garis putus-putus yang menempel pada garis putih berarti prediksinya baik.",
            "Angka R² pada judul setiap grafik: <b>semakin mendekati 1 semakin baik</b>. Polutan dengan R² rendah atau "
            "negatif adalah polutan yang sulit diprediksi pada horizon ini.",
            "Coba ganti horizon dari H+1 ke H+7: umumnya R² menurun karena hari yang lebih jauh lebih sulit "
            "diperkirakan.",
        )


def _batang(ev, kol_x, kol_y, urutan=None, label_x=None, judul_x=None, judul_y=None):
    agg = ev.groupby([kol_x, "algoritma"])[kol_y].mean().reset_index()
    fig = go.Figure()
    daftar = ("XGBoost", "LightGBM") if kol_y == "r2" else ("Persistensi", "XGBoost", "LightGBM")
    for a in daftar:
        s = agg[agg.algoritma == a]
        fig.add_bar(
            x=s[kol_x] if label_x is None else s[kol_x].map(label_x),
            y=s[kol_y],
            name=a,
            marker_color=WARNA_ALG[a],
            text=s[kol_y].round(2),
            textposition="outside",
        )
    fig.update_layout(
        barmode="group",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h"),
        xaxis_title=judul_x,
        yaxis_title=judul_y,
    )
    if urutan:
        fig.update_xaxes(categoryorder="array", categoryarray=urutan)
    _grafik(fig)
    model = agg[agg.algoritma.isin(["XGBoost", "LightGBM"])].groupby(kol_x)[kol_y].mean()
    nama_grup = (lambda v: label_x(v)) if label_x else str
    lebih_baik = "semakin tinggi" if kol_y == "r2" else "semakin pendek"
    ringkas = ""
    if len(model) > 1:
        baik, buruk = (model.idxmax(), model.idxmin()) if kol_y == "r2" else (model.idxmin(), model.idxmax())
        ringkas = (
            f"Rata-rata XGBoost dan LightGBM: paling baik pada <b>{nama_grup(baik)}</b>, paling buruk pada "
            f"<b>{nama_grup(buruk)}</b>."
        )
    _cara_baca(
        f"Untuk {judul_y}, <b>batang {lebih_baik} berarti lebih baik</b>."
        + (" R² = 1 sempurna; R² di bawah 0 berarti lebih buruk daripada menebak rata-rata." if kol_y == "r2" else
           " Angkanya adalah besar kesalahan prediksi dalam poin ISPU."),
        "Bandingkan warna dalam satu kelompok: ungu XGBoost, merah muda LightGBM"
        + (", abu-abu persistensi (tebakan paling sederhana: besok sama dengan hari ini). Model berguna bila "
           "batangnya lebih pendek daripada batang abu-abu." if kol_y != "r2" else "."),
        ringkas,
    )


def _warnai_peringkat(df, kolom, n=3):
    """Baris peringkat teratas hijau, terbawah merah (untuk tabel model terbaik)."""

    def gaya(r):
        if r["Peringkat"] <= n:
            return ["background-color: rgba(46,158,79,.28)"] * len(r)
        if r["Peringkat"] > len(df) - n:
            return ["background-color: rgba(217,83,79,.28)"] * len(r)
        return [""] * len(r)

    return df.style.apply(gaya, axis=1).format({k: "{:.3f}" for k in kolom})


_NAMA_KOLOM = {
    "stasiun": "Stasiun",
    "parameter": "Parameter",
    "horizon": "Horizon",
    "algoritma": "Algoritma",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "n": "Jumlah observasi uji",
    "peringkat": "Peringkat",
}


def _kalimat_model(r):
    bag = []
    if "stasiun" in r:
        bag.append(str(r["stasiun"]))
    if "parameter" in r:
        bag.append(PARAM.get(r["parameter"], r["parameter"]))
    if "horizon" in r:
        bag.append(f"H+{int(r['horizon'])}")
    return " · ".join(bag + [str(r["algoritma"])])


def halaman_evaluasi():
    st.header("Evaluasi Model")
    ev = sis.muat_evaluasi(akt["id"], "baru")
    if ev.empty:
        st.info("Hasil evaluasi belum tersedia.")
        return
    ev = ev.assign(Parameter=ev["parameter"].map(PARAM))
    st.caption(
        f"Model versi {akt['nomor']}, dievaluasi satu kali pada data uji setelah {tgl_id(akt['batas_latih'])}, "
        "hanya pada observasi asli."
    )
    pilihan = {"R²": "r2", "MAE": "mae", "RMSE": "rmse"}
    nm = st.radio("Ukuran", list(pilihan), horizontal=True)
    kol = pilihan[nm]
    st.caption(
        "R²: 1 berarti sempurna, 0 sama dengan menebak rata-rata, negatif berarti lebih buruk daripada menebak "
        "rata-rata. MAE dan RMSE dalam poin ISPU: semakin kecil semakin baik."
    )
    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Ringkasan", "Model terbaik", "Per stasiun", "Per parameter", "Per horizon", "Tabel rinci"]
    )
    with t1:
        rata = ev.groupby("algoritma")[["mae", "rmse", "r2"]].mean()
        model = [a for a in core.ALGORITMA if a in rata.index]
        pv = ev.pivot_table(index=["stasiun", "parameter", "horizon"], columns="algoritma", values="mae")

        st.markdown("##### Akurasi dibanding persistensi")
        akurasi = rata.reindex(model + ["Persistensi"]).dropna(how="all")[["mae", "rmse"]].round(3)
        akurasi.columns = ["MAE", "RMSE"]
        if "Persistensi" in pv:
            akurasi["Lebih akurat dari persistensi (MAE)"] = [
                f"{(pv[a] < pv['Persistensi']).sum()} dari {len(pv)} kombinasi" if a in model else "–"
                for a in akurasi.index
            ]
        _tabel(akurasi)

        st.markdown("##### Kebaikan model (R²)")
        _tabel(rata.reindex(model)[["r2"]].round(3).rename(columns={"r2": "R²"}))
        st.caption(
            "Rata-rata seluruh stasiun, parameter, dan horizon. Jumlah model: 5 stasiun × 6 parameter × 7 horizon × "
            "2 algoritma = 420."
        )
        _cara_baca(
            "<b>Persistensi</b> adalah tebakan paling sederhana: nilai besok dianggap sama dengan nilai hari ini. "
            "Model dianggap berguna bila kesalahannya lebih kecil daripada persistensi.",
            "<b>MAE dan RMSE</b>: rata-rata selisih prediksi, dalam poin ISPU; <b>semakin kecil semakin baik</b>. "
            "Kolom \"Lebih akurat dari persistensi\" menghitung berapa banyak kombinasi stasiun, parameter, dan "
            "horizon yang dimenangkan model.",
            "<b>R²</b>: 1 sempurna, 0 sama dengan menebak rata-rata; <b>semakin besar semakin baik</b>.",
        )
    with t2:
        st.markdown(
            "Peringkat model **tanpa persistensi**, diurutkan dari yang terbaik. Pilih level analisis: misalnya "
            "*Parameter × horizon* untuk melihat polutan dan horizon mana yang paling dapat diprediksi."
        )
        c = st.columns([2, 1])
        lv = c[0].selectbox("Level analisis", ["Seluruh 420 model"] + list(core.LEVEL), key="mt_level")
        mt = c[1].selectbox("Diurutkan menurut", ["R²", "MAE", "RMSE"], key="mt_metrik")
        km = pilihan[mt]
        if lv == "Seluruh 420 model":
            tb = core.peringkat_semua(ev, km)[
                ["peringkat", "stasiun", "parameter", "horizon", "algoritma", "mae", "rmse", "r2", "n"]
            ]
        else:
            tb = core.model_terbaik(ev, core.LEVEL[lv], km)
            st.caption("Pada level ini metrik dirata-ratakan untuk setiap kelompok, lalu dipilih algoritma terbaiknya.")
        best, worst = tb.iloc[0].to_dict(), tb.iloc[-1].to_dict()
        k1, k2 = st.columns(2)
        k1.success(f"**Terbaik:** {_kalimat_model(best)} — R² {_angka(best['r2'], 3)}, MAE {_angka(best['mae'], 2)}")
        k2.error(f"**Terburuk:** {_kalimat_model(worst)} — R² {_angka(worst['r2'], 3)}, MAE {_angka(worst['mae'], 2)}")
        tb = tb.assign(parameter=tb["parameter"].map(PARAM)) if "parameter" in tb else tb
        tb = tb.assign(horizon=tb["horizon"].map(lambda v: f"H+{int(v)}")) if "horizon" in tb else tb
        tampil = tb.rename(columns=_NAMA_KOLOM)
        st.dataframe(
            _warnai_peringkat(tampil, ["MAE", "RMSE", "R²"]), hide_index=True, height=420, **_lebar(st.dataframe)
        )
        _cara_baca(
            "Peringkat 1 adalah model terbaik menurut ukuran yang dipilih. <b>Baris hijau</b> tiga terbaik, <b>baris "
            "merah</b> tiga terburuk.",
            "Kotak hijau dan merah di atas tabel merangkum yang terbaik dan terburuk dalam satu kalimat. Untuk R², "
            "angka lebih besar lebih baik; untuk MAE dan RMSE, angka lebih kecil lebih baik.",
        )
        st.download_button(
            "Unduh peringkat (CSV)",
            _csv(tampil),
            file_name=f"peringkat_model_{lv.replace(' ', '_')}.csv",
            mime="text/csv",
        )
        st.markdown("##### Sampai horizon berapa prediksi masih baik?")
        ambang = st.slider("Ambang R² yang dianggap baik", 0.10, 0.90, 0.50, 0.05, key="mt_ambang")
        per = st.radio("Dikelompokkan per", ["Parameter", "Stasiun × parameter"], horizontal=True, key="mt_per")
        hj = core.horizon_terjauh(ev, ambang, ("parameter",) if per == "Parameter" else ("stasiun", "parameter"))
        hj["parameter"] = hj["parameter"].map(PARAM)
        hj["horizon_terjauh"] = hj["horizon_terjauh"].map(
            lambda v: "belum mencapai ambang" if v == 0 else f"sampai H+{v}"
        )
        _tabel(hj.rename(columns=_NAMA_KOLOM | {"horizon_terjauh": "Prediksi masih baik"}).round(3), hide_index=True)
        st.caption(
            "R² per horizon adalah rata-rata kelima stasiun (atau satu stasiun) untuk algoritma terbaik. "
            '"Sampai H+k" berarti R² ≥ ambang berturut-turut dari H+1 sampai H+k.'
        )
        _cara_baca(
            "Tabel ini menjawab: <b>sampai berapa hari ke depan prediksi masih bisa diandalkan?</b> Semakin jauh "
            "(mendekati H+7), semakin baik polutan itu dapat diprediksi.",
            "\"Belum mencapai ambang\" berarti bahkan prediksi besok (H+1) belum cukup baik menurut ambang yang "
            "dipilih. Geser ambang ke kanan untuk standar yang lebih ketat, ke kiri untuk yang lebih longgar.",
        )
    with t3:
        _batang(ev, "stasiun", kol, urutan=list(STASIUN), judul_x="Stasiun", judul_y=nm)
    with t4:
        _batang(ev, "Parameter", kol, urutan=list(PARAM.values()), judul_x="Parameter", judul_y=nm)
    with t5:
        _batang(
            ev,
            "horizon",
            kol,
            urutan=[f"H+{h}" for h in core.HORIZON],
            label_x=lambda v: f"H+{v}",
            judul_x="Horizon (hari ke depan)",
            judul_y=nm,
        )
    with t6:
        c = st.columns(3)
        fs = c[0].multiselect("Stasiun", list(STASIUN), default=list(STASIUN))
        fp = c[1].multiselect("Parameter", list(PARAM.values()), default=list(PARAM.values()))
        fh = c[2].multiselect("Horizon", list(core.HORIZON), default=list(core.HORIZON), format_func=lambda v: f"H+{v}")
        d = ev[ev.stasiun.isin(fs) & ev.Parameter.isin(fp) & ev.horizon.isin(fh)]
        d = d[["stasiun", "Parameter", "horizon", "algoritma", "mae", "rmse", "r2", "n"]].sort_values(
            ["stasiun", "Parameter", "horizon", "algoritma"]
        )
        d.columns = ["Stasiun", "Parameter", "Horizon", "Metode", "MAE", "RMSE", "R²", "Jumlah observasi uji"]
        _tabel(d.round(3), hide_index=True)
        _cara_baca(
            "Setiap baris adalah satu model: satu stasiun, satu polutan, satu horizon, dan satu metode. MAE dan RMSE "
            "<b>lebih kecil lebih baik</b>; R² <b>lebih besar lebih baik</b>. Jumlah observasi uji adalah banyaknya "
            "hari yang dipakai untuk menilai model itu.",
        )


# Pertanyaan tentang cara membaca grafik dan hasil (dikelompokkan terpisah di halaman Tanya Jawab).
FAQ_GRAFIK = [
    (
        "Secara ringkas, angka mana yang berarti lebih baik dan mana yang lebih buruk?",
        (
            "- **Nilai ISPU**: semakin **kecil** semakin baik (udara lebih bersih).\n"
            "- **Selisih prediksi dengan nilai aktual**: semakin **kecil** semakin tepat.\n"
            "- **MAE** dan **RMSE** (besar kesalahan, dalam poin ISPU): semakin **kecil** semakin baik.\n"
            "- **R²** (seberapa baik prediksi mengikuti naik-turun data): semakin **besar** semakin baik; 1 sempurna,"
            " 0 sama dengan menebak rata-rata, negatif lebih buruk daripada menebak rata-rata.\n"
            "- **Korelasi r**: tidak ada baik atau buruk; angka yang mendekati 1 atau −1 berarti hubungannya kuat.\n"
            "- **Prediksi masih baik sampai H+k**: semakin **jauh** (mendekati H+7) semakin baik.\n"
            "- **Peringkat model**: peringkat 1 terbaik; baris hijau tiga terbaik, baris merah tiga terburuk.\n\n"
            "Setiap grafik di dasbor juga memiliki kotak **Cara membaca** tepat di bawahnya."
        ),
    ),
    (
        "Bagaimana cara memperbesar, menggeser, atau menyimpan grafik?",
        (
            "Arahkan kursor ke grafik, lalu ikon kecil muncul di pojok kanan atas. Dari kiri ke kanan: **kamera** "
            "untuk menyimpan grafik sebagai gambar PNG; **kaca pembesar** lalu tarik kotak pada grafik untuk "
            "memperbesar bagian itu; **panah empat arah** untuk menggeser grafik; **plus** dan **minus** untuk "
            "memperbesar dan memperkecil; **kotak dengan panah** untuk menyesuaikan skala otomatis; **rumah** untuk "
            "kembali ke tampilan semula; dan **layar penuh**. Cara lain: klik dua kali pada grafik untuk kembali ke "
            "tampilan semula, arahkan kursor ke garis untuk melihat tanggal dan nilainya, dan klik nama pada "
            "keterangan (legend) untuk menyembunyikan atau menampilkan garis itu."
        ),
    ),
    (
        "Bagaimana membaca panel atas dan kartu di Beranda?",
        (
            "Panel besar di atas menampilkan ISPU hari itu, yaitu nilai tertinggi dari keenam polutan pada tanggal "
            "data terakhir, beserta kategorinya. Di sisi kanan tertulis parameter kritis (polutan penyumbang nilai "
            "tertinggi) dan prediksi ISPU maksimum untuk besok. Warna panel ikut berganti mengikuti kategori hari itu. "
            "Enam kartu di bawahnya adalah nilai ISPU setiap polutan pada tanggal yang sama, masing-masing dengan "
            "label kategorinya, sehingga terlihat polutan mana yang paling tinggi."
        ),
    ),
    (
        "Bagaimana membaca grafik nilai aktual dan prediksi di Beranda?",
        (
            "Garis putih adalah nilai yang benar-benar terukur. Titik penuh berarti observasi asli, lingkaran kosong"
            " berarti nilai isian untuk celah pendek (paling lama tiga hari). Area abu-abu berarti tidak ada data pada"
            " hari itu. Garis putus-putus adalah prediksi H+1 yang dibuat sehari sebelumnya: semakin rapat garis ini"
            " dengan garis putih, semakin baik prediksinya. Garis titik-titik dengan wajik di ujung kanan adalah"
            " prediksi tujuh hari ke depan. Pita warna di latar menunjukkan kategori ISPU: bila garis masuk pita"
            " kuning, udara berkategori Tidak Sehat pada hari itu."
        ),
    ),
    (
        "Bagaimana membaca grafik di halaman Data Historis?",
        (
            "Setiap variabel yang dipilih digambar pada grafiknya sendiri, karena satuannya berbeda (ISPU berupa "
            "indeks, suhu dalam °C, kelembapan dalam %, angin dalam m/s, hujan dalam mm). Sumbu tanggal semua grafik "
            "sama, jadi baca dari atas ke bawah pada tanggal yang sama untuk membandingkan, misalnya apakah PM2,5 "
            "turun saat hujan tinggi. Memperbesar satu grafik ikut memperbesar grafik lainnya. Garis yang naik berarti "
            "nilai meningkat; garis datar yang sangat panjang dengan nilai persis sama patut dicurigai sebagai "
            "gangguan alat."
        ),
    ),
    (
        "Bagaimana membaca grafik korelasi (hubungan antarvariabel)?",
        (
            "Grafik ini berupa kotak-kotak. Setiap kotak berisi angka korelasi r antara variabel di bawahnya dan "
            "variabel di sebelah kirinya, dari −1 sampai 1. Biru tua dan mendekati 1: kedua variabel cenderung naik "
            "dan turun bersama, misalnya PM2,5 dan PM10 (r sekitar 0,9) yang sama-sama berasal dari debu dan asap. "
            "Merah tua dan mendekati −1: satu naik saat yang lain turun. Putih atau sekitar 0: tidak ada pola bersama. "
            "Patokan kasar: di atas 0,7 kuat, 0,4–0,7 sedang, 0,2–0,4 lemah. Kotak diagonal selalu 1 karena "
            "membandingkan variabel dengan dirinya sendiri, dan kotak di atas diagonal adalah cerminan kotak di "
            "bawahnya. Di bawah grafik tertulis tiga pasangan dengan hubungan terkuat beserta artinya. Korelasi tinggi "
            "tidak berarti satu variabel menyebabkan variabel lainnya."
        ),
    ),
    (
        "Bagaimana membaca tabel dan grafik di halaman Prediksi?",
        (
            "Tabel berisi prediksi ISPU setiap polutan untuk H+1 (besok, dihitung dari data terakhir) sampai H+7, "
            "beserta ISPU maksimum dan kategorinya per hari. Enam grafik di bawahnya menggambarkan isi tabel yang sama "
            "per polutan: angka di atas titik adalah nilai prediksinya, dan warna latar adalah kategori ISPU. Ganti "
            "algoritma untuk melihat hasil XGBoost atau LightGBM. Prediksi untuk hari yang lebih jauh umumnya kurang "
            "tepat dibanding hari yang dekat."
        ),
    ),
    (
        "Bagaimana membaca grafik di halaman Perbandingan Metode?",
        (
            "Grafik ini memakai periode uji, yaitu data yang tidak dipakai melatih model, sehingga menunjukkan "
            "kemampuan model pada data yang belum pernah dilihatnya. Garis putih adalah nilai aktual, garis "
            "putus-putus ungu prediksi XGBoost, dan garis putus-putus merah muda prediksi LightGBM. Model yang baik "
            "garisnya mengikuti naik-turun garis putih. Tabel di bawahnya meringkas selisihnya: MAE dan RMSE lebih "
            "kecil berarti lebih baik, R² lebih besar berarti lebih baik. Pada tab per polutan, nilai R² tertulis pada "
            "judul setiap grafik."
        ),
    ),
    (
        "Bagaimana membaca grafik batang dan tabel di halaman Evaluasi Model?",
        (
            "Grafik batang membandingkan rata-rata MAE, RMSE, atau R² per stasiun, per parameter, atau per horizon "
            "untuk persistensi (abu-abu), XGBoost (ungu), dan LightGBM (merah muda). Untuk MAE dan RMSE, batang yang "
            "lebih pendek berarti lebih akurat; untuk R², batang yang lebih tinggi berarti lebih baik. Pada tab Model "
            "terbaik, baris hijau adalah tiga peringkat teratas dan baris merah tiga peringkat terbawah. Tabel "
            "\"Prediksi masih baik\" menunjukkan sampai berapa hari ke depan R² masih di atas ambang yang Anda pilih."
        ),
    ),
]

FAQ_UMUM = [
    (
        "Apa isi dasbor ini dan dari mana saya sebaiknya mulai?",
        (
            "Dasbor ini menampilkan kualitas udara di lima stasiun pemantauan Jakarta beserta perkiraannya untuk tujuh"
            " hari ke depan. Untuk sekadar mengetahui kondisi udara, buka **Beranda** dan pilih stasiun terdekat di"
            " sidebar. **Data Historis** untuk melihat data masa lalu, **Prediksi Multi-Parameter** untuk angka"
            " perkiraan tujuh hari, sedangkan **Perbandingan Metode** dan **Evaluasi Model** untuk Anda yang ingin tahu"
            " seberapa tepat modelnya."
        ),
    ),
    (
        "Di mana lokasi kelima stasiun?",
        (
            "DKI1 Bundaran HI (Jakarta Pusat), DKI2 Kelapa Gading (Jakarta Utara), DKI3 Jagakarsa (Jakarta Selatan),"
            " DKI4 Lubang Buaya (Jakarta Timur), dan DKI5 Kebon Jeruk (Jakarta Barat). Pilih stasiun yang paling dekat"
            " dengan lokasi Anda; pilihan ini berlaku untuk semua halaman."
        ),
    ),
    (
        "Apa arti setiap kategori ISPU bagi kegiatan sehari-hari?",
        (
            "**Baik (0–50)**: udara bersih, kegiatan di luar ruangan aman. **Sedang (51–100)**: masih dapat diterima,"
            " tetapi orang yang sangat sensitif sebaiknya memperhatikan kondisinya. **Tidak Sehat (101–200)**: kelompok"
            " sensitif, seperti anak-anak, lansia, dan penderita gangguan pernapasan atau jantung, sebaiknya mengurangi"
            " kegiatan berat di luar ruangan. **Sangat Tidak Sehat (201–300)**: semua orang sebaiknya mengurangi"
            " kegiatan di luar ruangan. **Berbahaya (di atas 300)**: hindari kegiatan di luar ruangan. Ikuti selalu"
            " imbauan resmi pemerintah setempat."
        ),
    ),
    (
        "Mengapa tanggal data terakhir bukan hari ini?",
        (
            "Dasbor memakai data harian resmi yang ditambahkan Admin secara berkala, bukan data langsung dari alat. "
            "Tanggal data terakhir tertulis di panel Beranda dan di bagian bawah sidebar. Prediksi tujuh hari dihitung "
            "mulai dari tanggal itu, bukan dari tanggal hari ini; tanggal persisnya tertulis di judul halaman Prediksi."
        ),
    ),
    (
        "Apa arti H+1 sampai H+7?",
        (
            "H adalah tanggal data terakhir. H+1 berarti satu hari setelahnya (\"besok\" menurut data), H+2 dua hari"
            " setelahnya, dan seterusnya sampai H+7. Semakin besar angkanya, semakin jauh harinya dan semakin kurang"
            " pasti prediksinya."
        ),
    ),
    (
        "Apa beda nilai aktual dan nilai prediksi?",
        (
            "Nilai aktual adalah nilai yang benar-benar terukur oleh stasiun. Nilai prediksi adalah perkiraan yang "
            "dihitung model dari data hari-hari sebelumnya. Untuk tanggal yang sudah lewat, keduanya dapat "
            "dibandingkan untuk melihat ketepatan model; untuk tanggal ke depan, hanya prediksi yang tersedia."
        ),
    ),
]

FAQ_MODEL_BARU = [
    (
        "Apa itu periode latih dan periode uji?",
        (
            "Model belajar dari data lama (periode latih), lalu dinilai pada data sesudahnya yang tidak pernah "
            "dilihatnya saat belajar (periode uji). Penilaian pada periode uji menggambarkan kemampuan model "
            "menghadapi data baru. Semua angka di halaman Perbandingan Metode dan Evaluasi Model berasal dari periode "
            "uji; tanggal batasnya tertulis di judul halaman Perbandingan Metode."
        ),
    ),
    (
        "Mengapa prediksi XGBoost dan LightGBM bisa berbeda, dan mana yang dipakai di Beranda?",
        (
            "Keduanya dilatih terpisah dengan cara membangun pohon keputusan yang berbeda, sehingga angkanya bisa "
            "sedikit berbeda. Panel atas Beranda menampilkan prediksi besok dari LightGBM; pada grafik Beranda dan "
            "halaman Prediksi Anda dapat memilih salah satu atau keduanya. Mana yang lebih tepat untuk polutan dan "
            "horizon tertentu dapat dilihat di halaman Evaluasi Model."
        ),
    ),
]

FAQ = [
    (
        "Apa itu ISPU?",
        (
            "ISPU (Indeks Standar Pencemar Udara) adalah angka tanpa satuan yang menggambarkan tingkat pencemaran"
            " udara. Semakin tinggi angkanya, semakin buruk kualitas udara. Angka ini bukan konsentrasi polutan dalam"
            " µg/m³, melainkan hasil konversi konsentrasi ke skala indeks."
        ),
    ),
    (
        "Apa saja enam parameter yang ditampilkan?",
        (
            "PM2,5 dan PM10 adalah partikel halus di udara, SO₂ adalah sulfur dioksida, CO adalah karbon monoksida, "
            "O₃ adalah ozon permukaan, dan NO₂ adalah nitrogen dioksida. Setiap parameter memiliki nilai ISPU sendiri."
        ),
    ),
    (
        "Bagaimana nilai ISPU maksimum dan kategorinya ditentukan?",
        (
            "ISPU hari itu adalah nilai tertinggi dari keenam parameter, dan parameter yang memberi nilai tertinggi"
            " disebut parameter kritis. Kategorinya: Baik (0–50), Sedang (51–100), Tidak Sehat (101–200), Sangat Tidak"
            " Sehat (201–300), dan Berbahaya (di atas 300)."
        ),
    ),
    (
        "Bagaimana sistem memprediksi tujuh hari ke depan?",
        (
            "Sistem memakai strategi langsung (multi-horizon): untuk setiap stasiun, parameter, dan hari ke depan (H+1"
            " sampai H+7) dilatih satu model tersendiri untuk XGBoost dan satu untuk LightGBM, sehingga ada 5 × 6 × 7 ×"
            " 2 = 420 model. Masukannya adalah nilai keenam parameter pada hari ini dan dua hari sebelumnya, rata-rata"
            " 7 dan 30 hari terakhir, suhu, kelembapan, angin, dan hujan pada hari ini dan kemarin, posisi hari dalam"
            " tahun (musim), dan hari dalam minggu. Model menebak perubahan dari nilai terakhir, lalu prediksi = nilai"
            " terakhir + perubahan. Prediksi H+7 tidak dihitung dari prediksi H+1, sehingga kesalahan tidak menumpuk"
            " dari hari ke hari."
        ),
    ),
    (
        'Apa maksud model "berbobot"?',
        (
            "Setiap model tetap milik satu stasiun. Saat dilatih, model DKI1 membaca data DKI1 dengan bobot penuh (1)"
            " dan data empat stasiun lain dengan bobot lebih kecil (0,3), ditambah penanda stasiun. Cara ini menambah"
            " contoh belajar karena data tiap stasiun terbatas, tetapi karakter stasiun itu sendiri tetap paling"
            " berpengaruh."
        ),
    ),
    (
        "Apa beda XGBoost dan LightGBM?",
        (
            "Keduanya termasuk keluarga gradient boosting, yaitu menggabungkan banyak pohon keputusan kecil yang saling"
            " memperbaiki kesalahan. XGBoost membentuk pohon tingkat demi tingkat, sedangkan LightGBM membentuk pohon"
            " dengan memperluas cabang yang paling mengurangi kesalahan sehingga umumnya lebih cepat dilatih. Dasbor"
            " menampilkan keduanya agar dapat dibandingkan."
        ),
    ),
    (
        "Apa itu persistensi dan mengapa dipakai sebagai pembanding?",
        (
            "Persistensi adalah tebakan paling sederhana: nilai besok dianggap sama dengan nilai hari ini. Model yang"
            " dibuat baru layak dipakai bila lebih baik daripada tebakan sederhana ini."
        ),
    ),
    (
        "Apa arti MAE, RMSE, dan R²?",
        (
            "MAE adalah rata-rata selisih mutlak antara prediksi dan kenyataan (dalam poin ISPU). RMSE serupa tetapi"
            " lebih menghukum kesalahan besar. Keduanya dibandingkan dengan persistensi. R² menunjukkan seberapa baik"
            " model menjelaskan naik turunnya nilai: 1 berarti sempurna, 0 sama dengan menebak rata-rata, dan negatif"
            " berarti lebih buruk daripada menebak rata-rata."
        ),
    ),
    (
        "Bagaimana memilih model terbaik?",
        (
            "Buka Evaluasi Model, tab Model terbaik. Pilih level analisis (misalnya parameter, horizon, atau stasiun ×"
            " parameter × horizon) dan ukuran pengurutan (R², MAE, atau RMSE). Tabel menampilkan peringkat dari terbaik"
            " ke terburuk tanpa persistensi, beserta horizon terjauh yang prediksinya masih baik untuk setiap polutan."
        ),
    ),
    (
        "Mengapa ada garis yang terputus atau nilai kosong pada data?",
        (
            "Garis terputus berarti pada hari itu tidak ada data, bukan kesalahan program. Nilai kosong muncul, antara"
            " lain, karena alat tidak mengirim data atau data tidak normal akibat pemeliharaan alat dan gangguan sensor"
            " dibuang pada penjaminan mutu. Celah paling lama tiga hari diisi dengan interpolasi linear (lingkaran"
            " kosong pada grafik), sedangkan celah yang lebih panjang dibiarkan kosong (area abu-abu) agar tidak"
            " menampilkan nilai karangan. Januari sampai November 2022 hampir tidak memiliki observasi sehingga tidak"
            " direkonstruksi."
        ),
    ),
    (
        "Mengapa grafik prediksi dipisahkan per polutan?",
        (
            "Pola dan rentang nilai tiap polutan berbeda, sehingga satu grafik bertumpuk menyulitkan pembacaan. Setiap"
            " polutan ditampilkan pada grafiknya sendiri dengan warna latar kategori ISPU. Batas kategori (50, 100,"
            " 200, 300) sama untuk semua polutan karena ISPU adalah indeks: konsentrasi setiap polutan sudah dikonversi"
            " ke skala yang sama, sehingga nilai 120 pada NO₂ dan 120 pada O₃ sama-sama berkategori Tidak Sehat."
        ),
    ),
    (
        "Apakah data meteorologi berasal dari stasiun yang sama?",
        (
            "Tidak. Data ISPU berasal dari lima stasiun pemantauan DLH DKI Jakarta, sedangkan suhu, kelembapan, angin,"
            " dan hujan berasal dari NASA POWER (data reanalisis beresolusi sekitar 0,5° × 0,625°). Akibatnya DKI1,"
            " DKI2, dan DKI5 memperoleh nilai meteorologi yang sama, demikian pula DKI3 dan DKI4."
        ),
    ),
    (
        "Seberapa jauh prediksi ini dapat dipercaya?",
        (
            "Prediksi adalah perkiraan statistik dari pola historis, bukan pengukuran. Akurasi cenderung menurun untuk"
            " hari yang lebih jauh dan untuk kejadian pencemaran mendadak. Hasil berlaku di sekitar stasiun"
            " masing-masing, bukan seluruh wilayah administrasi, dan tidak menggantikan pemantauan resmi. Lihat modul"
            " Evaluasi Model untuk besaran kesalahannya."
        ),
    ),
    (
        "Apa yang bisa saya lakukan setelah mendaftar sebagai User?",
        (
            "User dapat mengunggah data miliknya (ISPU dan meteorologi beberapa hari terakhir) untuk mendapatkan"
            " prediksi tujuh hari, serta mengunduh data historis dan hasil prediksi dalam format CSV. Berkas yang"
            " diunggah hanya dipakai untuk prediksi, tidak disimpan, dan tidak memengaruhi data maupun model yang"
            " dipakai pengguna lain."
        ),
    ),
    (
        "Saya lupa kata sandi, bagaimana?",
        (
            "Buka menu Masuk / Daftar, pilih tab Lupa kata sandi, lalu isi alamat surel, kode pemulihan yang Anda"
            " terima saat mendaftar, dan kata sandi baru. Kode hanya berlaku sekali; setelah berhasil, Anda menerima"
            ' kode pemulihan baru yang perlu disimpan lagi. Bila kode pemulihan hilang, pilih "Kirim kode ke surel":'
            " kode sekali pakai (berlaku 15 menit) dikirim ke alamat surel akun, lalu dipakai untuk membuat kata sandi"
            " baru. Setelah 5 kali salah memasukkan kata sandi atau kode, akun dikunci sementara."
        ),
    ),
    (
        "Apa itu kode pemulihan?",
        (
            "Kode acak 16 karakter (contoh bentuk: ABCD-EFGH-JKLM-NPQR) yang ditampilkan satu kali setelah registrasi."
            " Aplikasi hanya menyimpan hash-nya, sehingga kode tidak dapat dilihat lagi, bahkan oleh Admin. Simpan di"
            " tempat aman. Kode baru dapat dibuat di halaman Profil; kode lama langsung tidak berlaku. Akun Admin tidak"
            " memakai kode pemulihan."
        ),
    ),
    (
        "Mengapa perlu login untuk mengunggah dan mengunduh?",
        (
            "Beranda, data historis, prediksi, perbandingan metode, evaluasi, dan tanya jawab terbuka untuk umum karena"
            " datanya bersumber dari data pemerintah yang terbuka. Fitur unggah dan unduh dibatasi untuk pengguna"
            " terdaftar."
        ),
    ),
    (
        "Bagaimana format berkas yang harus saya unggah?",
        (
            "Berkas CSV dengan kolom: tanggal, pm25, pm10, so2, co, o3, no2, suhu, rh, angin, hujan. Minimal tiga hari"
            " terakhir yang berurutan, dengan keenam nilai ISPU lengkap pada tiga hari terakhir dan meteorologi lengkap"
            " pada dua hari terakhir. Contoh berkas dapat diunduh pada halaman Unggah Data untuk Prediksi."
        ),
    ),
]


PERTANYAAN_UMUM = (
    "Apa itu ISPU?",
    "Apa saja enam parameter yang ditampilkan?",
    "Bagaimana nilai ISPU maksimum dan kategorinya ditentukan?",
)
PERTANYAAN_PENUTUP_UMUM = "Seberapa jauh prediksi ini dapat dipercaya?"
PERTANYAAN_AKUN = (
    "Apa yang bisa saya lakukan setelah mendaftar sebagai User?",
    "Saya lupa kata sandi, bagaimana?",
    "Apa itu kode pemulihan?",
    "Mengapa perlu login untuk mengunggah dan mengunduh?",
    "Bagaimana format berkas yang harus saya unggah?",
)


def _kelompok_faq():
    """Mengelompokkan pertanyaan: pengantar umum, membaca grafik, model dan metode, akun dan berkas."""
    lama = dict(FAQ)
    umum = [(q, lama[q]) for q in PERTANYAAN_UMUM] + FAQ_UMUM
    umum.append((PERTANYAAN_PENUTUP_UMUM, lama[PERTANYAAN_PENUTUP_UMUM]))
    lain = set(PERTANYAAN_UMUM) | set(PERTANYAAN_AKUN) | {PERTANYAAN_PENUTUP_UMUM}
    model = [(q, a) for q, a in FAQ if q not in lain] + FAQ_MODEL_BARU
    return [
        ("Mulai dari sini", umum),
        ("Membaca grafik dan hasil", FAQ_GRAFIK),
        ("Model dan metode", model),
        ("Akun dan berkas", [(q, lama[q]) for q in PERTANYAAN_AKUN]),
    ]


def halaman_faq():
    st.header("Tanya Jawab (FAQ)")
    st.write("Penjelasan istilah dan cara membaca dasbor dengan bahasa yang mudah dipahami.")
    for judul, daftar in _kelompok_faq():
        st.subheader(judul)
        for q, a in daftar:
            with st.expander(q):
                st.markdown(a)


def _tampil_kode(kunci, judul):
    """Menampilkan kode pemulihan baru sampai pengguna menyatakan sudah menyimpannya."""
    t = st.session_state.get(kunci)
    if not t:
        return False
    st.success(t["pesan"])
    st.markdown(f"**{judul}**")
    st.code(t["kode"], language=None)
    st.warning(
        "Kode ini hanya tampil sekali dan tidak dapat dilihat lagi. Salin dan simpan di tempat aman. "
        "Bila hilang, Anda masih dapat meminta kode baru lewat surel."
    )
    if st.button("Saya sudah menyimpan kode ini", key=kunci + "_ok"):
        st.session_state.pop(kunci, None)
        st.rerun()
    return True


def halaman_masuk():
    st.header("Masuk / Daftar")
    if _tampil_kode("kode_masuk", "Kode pemulihan Anda"):
        return
    t1, t2, t3 = st.tabs(["Masuk", "Daftar sebagai User", "Lupa kata sandi"])
    with t1:
        with st.form("form_masuk"):
            email = st.text_input("Alamat surel")
            sandi = st.text_input("Kata sandi", type="password")
            if st.form_submit_button("Masuk"):
                u, pesan = sis.masuk(email, sandi)
                if u:
                    st.session_state["pengguna"] = u
                    st.rerun()
                st.error(pesan)
    with t2:
        with st.form("form_daftar"):
            nama = st.text_input("Nama")
            email = st.text_input("Alamat surel", key="d_email")
            sandi = st.text_input(
                "Kata sandi (minimal 8 karakter, memuat huruf dan angka)", type="password", key="d_sandi"
            )
            ulang = st.text_input("Ulangi kata sandi", type="password")
            if st.form_submit_button("Daftar"):
                ok, pesan, kode = sis.daftar(nama, email, sandi, ulang)
                if ok:
                    st.session_state["kode_masuk"] = {"pesan": pesan, "kode": kode}
                    st.rerun()
                st.error(pesan)
        st.caption(
            "Akun Admin tidak dapat didaftarkan melalui formulir ini. Setelah mendaftar, Anda menerima kode pemulihan "
            "untuk mengatur ulang kata sandi bila lupa."
        )
    with t3:
        st.write("Masukkan alamat surel akun, kode pemulihan yang Anda terima saat mendaftar, dan kata sandi baru.")
        with st.form("form_lupa"):
            email = st.text_input("Alamat surel", key="l_email")
            kode = st.text_input("Kode pemulihan (contoh: ABCD-EFGH-JKLM-NPQR)", key="l_kode")
            baru = st.text_input(
                "Kata sandi baru (minimal 8 karakter, memuat huruf dan angka)", type="password", key="l_baru"
            )
            ulang = st.text_input("Ulangi kata sandi baru", type="password", key="l_ulang")
            if st.form_submit_button("Atur ulang kata sandi"):
                if not sis.EMAIL_RE.match(email.strip()):
                    st.error("Format alamat surel tidak valid.")
                else:
                    ok, pesan, kode_baru = sis.pulihkan_dengan_kode(email, kode, baru, ulang)
                    if ok:
                        st.session_state["kode_masuk"] = {
                            "pesan": pesan + " Kode pemulihan lama sudah tidak berlaku; simpan kode baru di bawah ini.",
                            "kode": kode_baru,
                        }
                        st.rerun()
                    st.error(pesan)
        with st.expander("Kode pemulihan hilang? Kirim kode ke surel"):
            st.write(
                "Kami kirim kode sekali pakai ke alamat surel akun Anda (berlaku 15 menit). Dengan kode itu Anda dapat "
                "membuat kata sandi baru dan menerima kode pemulihan baru."
            )
            with st.form("form_kirim_surel"):
                email2 = st.text_input("Alamat surel", key="s_email")
                if st.form_submit_button("Kirim kode ke surel"):
                    if not sis.EMAIL_RE.match(email2.strip()):
                        st.error("Format alamat surel tidak valid.")
                    else:
                        ok, pesan = sis.kirim_kode_surel(email2)
                        (st.info if ok else st.error)(pesan)
            st.markdown("**Sudah menerima kode dari surel?**")
            with st.form("form_kode_surel"):
                email3 = st.text_input("Alamat surel", key="s2_email")
                kode3 = st.text_input("Kode dari surel (contoh: ABCD-EFGH)", key="s2_kode")
                baru3 = st.text_input(
                    "Kata sandi baru (minimal 8 karakter, memuat huruf dan angka)", type="password", key="s2_baru"
                )
                ulang3 = st.text_input("Ulangi kata sandi baru", type="password", key="s2_ulang")
                if st.form_submit_button("Atur ulang dengan kode surel"):
                    if not sis.EMAIL_RE.match(email3.strip()):
                        st.error("Format alamat surel tidak valid.")
                    else:
                        ok, pesan, kode_baru = sis.atur_ulang_dengan_kode_surel(email3, kode3, baru3, ulang3)
                        if ok:
                            st.session_state["kode_masuk"] = {
                                "pesan": (
                                    pesan + " Kode pemulihan lama sudah tidak berlaku; simpan kode baru di bawah ini."
                                ),
                                "kode": kode_baru,
                            }
                            st.rerun()
                        st.error(pesan)
        st.caption("Akun Admin tidak memakai jalur ini; Admin mengatur ulang kata sandi lewat terminal di server.")


def halaman_profil():
    if pengguna is None:
        st.warning("Silakan masuk untuk membuka halaman ini.")
        st.stop()
    st.header("Profil")
    p = sis.profil(pengguna["id"])
    if "info_profil" in st.session_state:
        st.success(st.session_state.pop("info_profil"))
    _tabel(
        pd.DataFrame(
            {
                "Keterangan": ["Nama", "Alamat surel", "Peran", "Terdaftar sejak", "Masuk terakhir"],
                "Isi": [p["nama"], p["email"], p["peran"], p["dibuat"], p["terakhir_masuk"] or "-"],
            }
        ),
        hide_index=True,
    )
    st.markdown("#### Ubah nama")
    with st.form("form_nama"):
        nama = st.text_input("Nama", value=p["nama"])
        if st.form_submit_button("Simpan nama"):
            ok, pesan = sis.ubah_nama(pengguna["id"], nama)
            if ok:
                st.session_state["pengguna"] = {**pengguna, "nama": nama.strip()}
                st.session_state["info_profil"] = pesan
                st.rerun()
            st.error(pesan)
    st.markdown("#### Ganti kata sandi")
    with st.form("form_sandi_profil", clear_on_submit=True):
        lama = st.text_input("Kata sandi saat ini", type="password")
        baru = st.text_input("Kata sandi baru (minimal 8 karakter, memuat huruf dan angka)", type="password")
        ulang = st.text_input("Ulangi kata sandi baru", type="password")
        if st.form_submit_button("Simpan kata sandi"):
            ok, pesan = sis.ganti_kata_sandi(pengguna["id"], lama, baru, ulang)
            (st.success if ok else st.error)(pesan)
    if pengguna["peran"] != "User":
        return
    st.markdown("#### Kode pemulihan")
    if _tampil_kode("kode_profil", "Kode pemulihan baru Anda"):
        return
    if sis.ada_kode_pemulihan(pengguna["id"]):
        st.write(
            "Akun ini memiliki kode pemulihan. Kode asli tidak dapat dilihat lagi; buat kode baru bila kode lama hilang"
            " atau pernah bocor (kode lama langsung tidak berlaku)."
        )
    else:
        st.warning(
            "Akun ini belum memiliki kode pemulihan. Buat sekarang agar kata sandi dapat diatur ulang bila lupa."
        )
    with st.form("form_kode_profil", clear_on_submit=True):
        sandi_kode = st.text_input("Kata sandi saat ini", type="password", key="kp_sandi")
        if st.form_submit_button("Buat kode pemulihan baru"):
            ok, hasil = sis.buat_ulang_kode(pengguna["id"], sandi_kode)
            if ok:
                st.session_state["kode_profil"] = {"pesan": "Kode pemulihan baru berhasil dibuat.", "kode": hasil}
                st.rerun()
            st.error(hasil)


def halaman_pengguna():
    _wajib("Admin")
    st.header("Kelola Pengguna")
    d = sis.daftar_pengguna()
    antre = d[d["minta_reset"] > 0]
    if len(antre):
        st.warning("Permintaan reset kata sandi menunggu: " + ", ".join(antre["email"]) + ".")
    st.markdown("#### Daftar akun")
    _tabel(
        d[["nama", "email", "peran", "status", "dibuat", "terakhir_masuk"]]
        .rename(
            columns={
                "nama": "Nama",
                "email": "Alamat surel",
                "peran": "Peran",
                "status": "Status",
                "dibuat": "Terdaftar",
                "terakhir_masuk": "Masuk terakhir",
            }
        )
        .fillna("-"),
        hide_index=True,
    )
    st.markdown("#### Tindakan pada akun User")
    u = d[d["peran"] == "User"].reset_index(drop=True)
    if u.empty:
        st.info("Belum ada akun User.")
        return
    if "info_pengguna" in st.session_state:
        st.success(st.session_state.pop("info_pengguna"))
    if "sandi_sementara" in st.session_state:
        t = st.session_state.pop("sandi_sementara")
        st.success(f"Kata sandi sementara untuk {t['email']}:")
        st.code(t["sandi"], language=None)
        st.caption(
            "Kata sandi ini hanya tampil sekali. Sampaikan kepada pemilik akun melalui jalur di luar aplikasi; "
            "ia wajib menggantinya saat masuk."
        )
    idx = st.selectbox(
        "Akun",
        range(len(u)),
        format_func=lambda i: f"{u.loc[i, 'nama']} ({u.loc[i, 'email']})"
        + (" - minta reset" if u.loc[i, "minta_reset"] else ""),
    )
    akun = u.loc[idx]
    c1, c2, c3 = st.columns(3)
    if c1.button("Reset kata sandi"):
        ok, hasil = sis.reset_sandi_oleh_admin(int(akun["id"]), pengguna["nama"])
        if ok:
            st.session_state["sandi_sementara"] = {"email": akun["email"], "sandi": hasil}
        else:
            st.session_state["info_pengguna"] = hasil
        st.rerun()
    if akun["status"] == "Nonaktif":
        if c2.button("Aktifkan akun"):
            st.session_state["info_pengguna"] = sis.atur_aktif(int(akun["id"]), True)[1]
            st.rerun()
    elif c2.button("Nonaktifkan akun"):
        st.session_state["info_pengguna"] = sis.atur_aktif(int(akun["id"]), False)[1]
        st.rerun()
    if akun["status"] == "Terkunci" and c3.button("Buka kunci"):
        st.session_state["info_pengguna"] = sis.buka_kunci(int(akun["id"]))[1]
        st.rerun()


def halaman_unggah():
    _wajib("User")
    st.header("Unggah Data untuk Prediksi")
    st.write(
        f"Prediksi memakai model stasiun **{STASIUN[stasiun]}** (ubah pada sidebar). Unggah berkas CSV berisi data "
        "harian terbaru Anda; minimal tiga hari terakhir yang berurutan. Berkas ini hanya dipakai untuk prediksi, "
        "tidak disimpan, dan tidak mengubah data historis maupun model."
    )
    st.code(", ".join(core.KOLOM_UNGGAH), language=None)
    contoh = g.tail(5)[core.KOLOM_UNGGAH].assign(tanggal=lambda d: d["tanggal"].dt.strftime("%Y-%m-%d")).round(1)
    st.download_button("Unduh contoh berkas (CSV)", _csv(contoh), file_name="contoh_unggahan.csv", mime="text/csv")
    berkas = st.file_uploader("Berkas CSV", type=["csv"], key="up_user")
    if berkas is None:
        return
    df = _baca_csv(berkas)
    if df is None:
        return
    d, galat = core.validasi_unggahan(df)
    if galat:
        _galat(galat)
        return
    st.success(f"Berkas sah: {len(d)} hari, {tgl_id(d['tanggal'].min())} sampai {tgl_id(d['tanggal'].max())}.")
    pr, x, tgl = core.prediksi_unggahan(d, model_stasiun(FOLDER, stasiun))
    st.subheader(f"Prediksi untuk {tgl_id(tgl + pd.Timedelta(days=1))} sampai {tgl_id(tgl + pd.Timedelta(days=7))}")
    _tampil_prediksi(
        pr,
        tgl,
        x,
        "unggah",
        "hasil_unggahan",
        dict(Stasiun_model=stasiun, Data_terakhir=tgl.strftime("%Y-%m-%d"), Versi_model=akt["nomor"]),
    )
    st.info("Hasil ini dapat diunduh pada halaman Unduh Hasil.")


def halaman_unduh():
    _wajib("User")
    st.header("Unduh Hasil")
    st.write("Unduh data atau hasil prediksi yang sedang ditampilkan pada sesi Anda dalam format CSV.")
    sumber = [
        ("Data historis", "tampil_historis", "Halaman Data Historis", "data_historis"),
        ("Hasil prediksi stasiun", "tampil_prediksi", "Halaman Prediksi Multi-Parameter", "prediksi"),
        (
            "Hasil prediksi dari data unggahan",
            "hasil_unggahan",
            "Halaman Unggah Data untuk Prediksi",
            "prediksi_unggahan",
        ),
    ]
    for judul, kunci, asal, nama in sumber:
        st.subheader(judul)
        df = st.session_state.get(kunci)
        if df is None:
            st.caption(f"Belum ada. Buka dan tampilkan data pada {asal} terlebih dahulu.")
            continue
        st.caption(f"{len(df)} baris siap diunduh.")
        _tabel(df.head(8), hide_index=True)
        st.download_button("Unduh CSV", _csv(df), file_name=f"{nama}_{stasiun}.csv", mime="text/csv", key=f"dl_{kunci}")


def halaman_kelola():
    _wajib("Admin")
    st.header("Kelola Data")
    st.markdown("#### Kondisi data historis saat ini")
    ring = data.groupby("stasiun").agg(
        tanggal_awal=("tanggal", "min"), tanggal_akhir=("tanggal", "max"), jumlah_baris=("tanggal", "size")
    )
    ring["tanggal_awal"], ring["tanggal_akhir"] = ring["tanggal_awal"].dt.date, ring["tanggal_akhir"].dt.date
    _tabel(ring)
    if "ringkasan_gabung" in st.session_state:
        r = st.session_state.pop("ringkasan_gabung")
        st.success(
            f"Data berhasil digabungkan: {r['baris']} baris, {tgl_id(r['awal'])} sampai {tgl_id(r['akhir'])}. "
            + ", ".join(f"{s}: {n}" for s, n in r["per_stasiun"].items())
        )
    st.markdown("#### Unggah data periode terbaru")
    st.write("Berkas CSV dengan kolom berikut. Hanya data setelah tanggal terakhir pada tabel di atas yang diterima.")
    st.code(", ".join(core.KOLOM_BARU), language=None)
    akhir = data["tanggal"].max()
    contoh = pd.DataFrame(
        {
            "stasiun": list(STASIUN),
            "tanggal": (akhir + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            **{c: 0.0 for c in list(PARAM) + list(METEO)},
        }
    )
    st.download_button("Unduh contoh format (CSV)", _csv(contoh), file_name="contoh_data_baru.csv", mime="text/csv")
    berkas = st.file_uploader(
        "Berkas CSV data terbaru", type=["csv"], key=f"up_admin_{st.session_state.get('kunci_up', 0)}"
    )
    if berkas is None:
        _riwayat()
        return
    df = _baca_csv(berkas)
    if df is None:
        return
    d, galat = core.validasi_data_baru(df, data)
    if galat:
        _galat(galat)
        return
    proc = core.praproses_data_baru(d, data)
    st.success(
        "Format dan kelengkapan kolom sesuai. Pra-pemrosesan: penandaan pencilan (Hampel), interpolasi celah "
        "paling lama 3 hari, dan penanda asal nilai."
    )
    ring_baru = proc.groupby("stasiun").agg(
        tanggal_awal=("tanggal", "min"),
        tanggal_akhir=("tanggal", "max"),
        baris=("tanggal", "size"),
        pm25_diinterpolasi=("pm25_flag", lambda s: int((s == 1).sum())),
        pm25_kosong=("pm25_flag", lambda s: int((s == 9).sum())),
    )
    ring_baru["tanggal_awal"], ring_baru["tanggal_akhir"] = (
        ring_baru["tanggal_awal"].dt.date,
        ring_baru["tanggal_akhir"].dt.date,
    )
    _tabel(ring_baru)
    if st.button("Gabungkan ke data historis", type="primary"):
        r = sis.gabung_data(proc, berkas.name, pengguna["email"])
        data_historis.clear()
        st.session_state["ringkasan_gabung"] = r
        st.session_state["kunci_up"] = st.session_state.get("kunci_up", 0) + 1
        st.rerun()
    _riwayat()


def _riwayat():
    st.markdown("#### Riwayat pembaruan data")
    rw = sis.riwayat_pembaruan()
    if rw.empty:
        st.caption("Belum ada pembaruan data.")
    else:
        _tabel(rw, hide_index=True)


def halaman_latih_ulang():
    _wajib("Admin")
    st.header("Pelatihan Ulang Model")
    if "pesan_versi" in st.session_state:
        st.success(st.session_state.pop("pesan_versi"))
    st.write(
        f"Model aktif: **versi {akt['nomor']}** (dilatih dengan data hingga {tgl_id(akt['batas_latih'])}; "
        f"data tersedia hingga {tgl_id(akt['data_hingga'])}). Pelatihan ulang memisahkan data setelah "
        f"{tgl_id(akt['data_hingga'])} sebagai segmen uji yang tidak ikut dilatih, dan memakai hiperparameter "
        "hasil penalaan tanpa pencarian ulang."
    )
    boleh, pesan, hari = sis.prasyarat_latih_ulang()
    _tabel(hari.rename("Hari data baru").to_frame())
    (st.success if boleh else st.warning)(pesan)
    berjalan = sis.STATUS["jalan"]
    if st.button("Latih ulang model XGBoost dan LightGBM", type="primary", disabled=(not boleh) or berjalan):
        try:
            sis.mulai_latih_latar()
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    terbaru = sis.versi_terbaru()
    if berjalan:
        st.progress(min(max(sis.STATUS["persen"], 0.0), 1.0), text=sis.STATUS["pesan"])
        st.caption("Pelatihan berjalan pada latar belakang sistem. Halaman ini diperbarui otomatis.")
        time.sleep(2)
        st.rerun()
    elif sis.STATUS["galat"]:
        st.error(f"Pelatihan gagal: {sis.STATUS['galat']}")
    elif sis.STATUS["selesai"]:
        if terbaru["status"] == sis.STATUS_AKTIF:
            st.success(
                f"Pelatihan ulang berhasil. Model versi {terbaru['nomor']} memenuhi aturan aktivasi dan kini aktif."
            )
        else:
            st.warning(
                f"Pelatihan ulang selesai, tetapi versi {terbaru['nomor']} belum aktif karena MAE rata-rata salah satu"
                f" algoritma naik. Tinjau perbandingan di bawah, lalu setujui atau biarkan versi {akt['nomor']} tetap"
                " aktif."
            )
        sis.STATUS["selesai"] = False
    lama, baru = sis.muat_evaluasi(terbaru["id"], "lama"), sis.muat_evaluasi(terbaru["id"], "baru")
    if not lama.empty:
        st.markdown(
            f"#### Perbandingan evaluasi: versi aktif sebelum pelatihan (lama) dan versi {terbaru['nomor']} (baru)"
        )
        st.caption(
            "Kedua model dievaluasi pada segmen uji yang sama (observasi asli saja), sehingga perbandingan adil. "
            "Versi baru aktif otomatis hanya jika MAE rata-rata XGBoost dan LightGBM masing-masing tidak naik."
        )
        a = lama.groupby("algoritma")[["mae", "rmse", "r2"]].mean()
        b = baru[baru.algoritma != "Persistensi"].groupby("algoritma")[["mae", "rmse", "r2"]].mean()
        tb = pd.concat({"Lama": a, "Baru": b}, axis=1).swaplevel(axis=1).sort_index(axis=1)
        tb.columns = [f"{m.upper() if m != 'r2' else 'R²'} {s.lower()}" for m, s in tb.columns]
        _tabel(tb.round(3))
        aturan = sis.banding_versi(baru, lama)
        aturan["tidak naik"] = aturan["tidak naik"].map({True: "Ya", False: "Tidak"})
        _tabel(
            aturan.round(3).rename(columns={"sel": "Jumlah sel dibandingkan", "tidak naik": "MAE tidak naik"}),
            hide_index=True,
        )
        fig = go.Figure()
        fig.add_bar(x=a.index, y=a["mae"], name="Lama", marker_color="#999999")
        fig.add_bar(x=b.index, y=b["mae"], name="Baru", marker_color="#2e9e4f")
        fig.update_layout(
            barmode="group",
            height=300,
            yaxis_title="MAE rata-rata",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h"),
        )
        _grafik(fig)
        _cara_baca(
            "MAE adalah rata-rata selisih prediksi dalam poin ISPU. <b>Batang hijau (baru) yang lebih pendek daripada "
            "batang abu-abu (lama)</b> berarti model baru lebih akurat pada data uji yang sama.",
        )
    st.markdown("#### Keputusan versi aktif")
    daftar = sis.daftar_versi()
    lain = daftar[daftar["status"] != sis.STATUS_AKTIF]
    if lain.empty:
        st.caption("Belum ada versi lain yang dapat diaktifkan.")
    else:
        pilihan = {f"Versi {r['nomor']} · {r['status']} · dibuat {r['dibuat']}": r for _, r in lain.iterrows()}
        r = pilihan[st.selectbox("Versi yang akan diaktifkan", list(pilihan))]
        tombol = (
            f"Setujui dan aktifkan versi {r['nomor']}"
            if r["status"] == sis.STATUS_MENUNGGU
            else f"Kembalikan ke versi {r['nomor']} (rollback)"
        )
        if st.button(tombol, disabled=berjalan):
            try:
                sis.aktifkan_versi(int(r["id"]), pengguna["nama"])
                st.session_state["pesan_versi"] = f"Model versi {r['nomor']} sekarang aktif."
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    st.markdown("#### Riwayat versi model")
    _tabel(daftar.drop(columns=["id"]), hide_index=True)


HALAMAN = {
    "Beranda": beranda,
    "Data Historis": halaman_data_historis,
    "Prediksi Multi-Parameter": halaman_prediksi,
    "Perbandingan Metode": halaman_perbandingan,
    "Evaluasi Model": halaman_evaluasi,
    "Tanya Jawab": halaman_faq,
    "Masuk / Daftar": halaman_masuk,
    "Unggah Data untuk Prediksi": halaman_unggah,
    "Unduh Hasil": halaman_unduh,
    "Kelola Data": halaman_kelola,
    "Pelatihan Ulang Model": halaman_latih_ulang,
    "Kelola Pengguna": halaman_pengguna,
    "Profil": halaman_profil,
}
HALAMAN[halaman]()
