"""Inti perhitungan dasbor: skema data, pra-pemrosesan, fitur, 420 model berbobot, penalaan, evaluasi, dan validasi
berkas. Tidak bergantung pada Streamlit, sehingga pipeline_final.py dan dasbor memakai kode yang sama.

Desain model: satu model per (stasiun, parameter, horizon, algoritma); baris stasiun sendiri berbobot 1 dan empat
stasiun lain BOBOT_STASIUN_LAIN; target selisih terhadap basis; fungsi kerugian galat mutlak (L1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DIR_MODEL = BASE / "models"

PARAM = {"pm25": "PM2,5", "pm10": "PM10", "so2": "SO₂", "co": "CO", "o3": "O₃", "no2": "NO₂"}
METEO = {"suhu": "Suhu (°C)", "rh": "Kelembapan (%)", "angin": "Angin (m/s)", "hujan": "Hujan (mm)"}
BULAN = [
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
]
STASIUN = {
    "DKI1": "DKI1 – Bundaran HI",
    "DKI2": "DKI2 – Kelapa Gading",
    "DKI3": "DKI3 – Jagakarsa",
    "DKI4": "DKI4 – Lubang Buaya",
    "DKI5": "DKI5 – Kebon Jeruk",
}
WARNA = {
    "Baik": "#2e9e4f",
    "Sedang": "#2f6fd1",
    "Tidak Sehat": "#e0a800",
    "Sangat Tidak Sehat": "#d9534f",
    "Berbahaya": "#222222",
}
HORIZON = tuple(range(1, 8))
ALGORITMA = ("XGBoost", "LightGBM")

BATAS_CELAH = 3
MIN_HARI_BARU = 90
BATAS_LATIH_AWAL = pd.Timestamp("2024-06-30")
AKHIR_LATIH_AWAL = pd.Timestamp("2023-12-31")
STRUKTURAL = (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-11-30"))

KOLOM_DATASET = (
    ["tanggal", "stasiun", "bagian", "periode_struktural_2022"]
    + list(PARAM)
    + [f"{p}_flag" for p in PARAM]
    + [f"{p}_pencilan" for p in PARAM]
    + [f"{p}_tersedia_sejak" for p in PARAM]
    + list(METEO)
)
KOLOM_UNGGAH = ["tanggal"] + list(PARAM) + list(METEO)
KOLOM_BARU = ["stasiun"] + KOLOM_UNGGAH
RENTANG_METEO = {"suhu": (-10, 60), "rh": (0, 100), "angin": (0, 100), "hujan": (0, 1000)}

BOBOT_STASIUN_LAIN = 0.3  # bobot sampel baris 4 stasiun lain; baris stasiun sendiri berbobot 1
OBJEKTIF = {"XGBoost": "reg:absoluteerror", "LightGBM": "l1"}
FITUR_DASAR = [f"{p}_t{L}" for p in PARAM for L in (0, 1, 2)] + [f"{m}_t{L}" for m in METEO for L in (0, 1)]
FITUR_TAMBAHAN = [f"{p}_rm{w}" for p in PARAM for w in (7, 30)] + ["musim_sin", "musim_cos", "hari_minggu"]
KOLOM_STASIUN = [f"st_{s}" for s in STASIUN]
FITUR_MODEL = FITUR_DASAR + FITUR_TAMBAHAN + KOLOM_STASIUN

SEED = 42
N_KOMBINASI = 30

HP_DEFAULT = {
    "XGBoost": dict(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        min_child_weight=3,
        n_jobs=1,
        verbosity=0,
        random_state=SEED,
    ),
    "LightGBM": dict(
        n_estimators=300,
        num_leaves=8,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5,
        min_child_samples=20,
        n_jobs=1,
        verbose=-1,
        random_state=SEED,
    ),
}


def tgl_id(t):
    return f"{t.day} {BULAN[t.month - 1]} {t.year}"


def kategori(v):
    if pd.isna(v):
        return "-"
    return (
        "Baik"
        if v <= 50
        else "Sedang" if v <= 100 else "Tidak Sehat" if v <= 200 else "Sangat Tidak Sehat" if v <= 300 else "Berbahaya"
    )


def normalkan_dataset(d):
    """Menyeragamkan himpunan data ke 32 kolom (Tabel 3.1). Kolom yang tidak ada dilengkapi nilai bawaan."""
    d = d.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    d["tanggal"] = pd.to_datetime(d["tanggal"])
    d["stasiun"] = d["stasiun"].astype(str).str.strip().str.upper()
    for p in PARAM:
        d[p] = pd.to_numeric(d[p], errors="coerce")
        if f"{p}_flag" not in d:
            d[f"{p}_flag"] = np.where(d[p].notna(), 0, 9)
        if f"{p}_pencilan" not in d:
            d[f"{p}_pencilan"] = 0
        c = f"{p}_tersedia_sejak"
        d[c] = pd.to_datetime(d[c]) if c in d else pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    for m in METEO:
        d[m] = pd.to_numeric(d[m], errors="coerce")
    if "periode_struktural_2022" not in d:
        d["periode_struktural_2022"] = d["tanggal"].between(*STRUKTURAL).astype(int)
    if "bagian" not in d:
        d["bagian"] = np.where(
            d["tanggal"] <= AKHIR_LATIH_AWAL, "latih", np.where(d["tanggal"] <= BATAS_LATIH_AWAL, "validasi", "uji")
        )
    return d[KOLOM_DATASET].sort_values(["stasiun", "tanggal"]).reset_index(drop=True)


def tanda_hampel(s, jendela=15, k=4.0, eps=0.05, min_obs=5, maks_run=2, return_arah=False):
    """Kode kandidat pencilan Hampel: 0 normal, 1 terisolasi, 2 episode. Nilai tidak diubah."""
    v = np.log1p(np.asarray(s, float))
    n, sisi = len(v), jendela // 2
    kand = np.zeros(n, bool)
    arah = np.zeros(n, int)
    for t in range(n):
        if np.isnan(v[t]):
            continue
        w = v[max(0, t - sisi) : t + sisi + 1]
        w = w[~np.isnan(w)]
        if len(w) < min_obs:
            continue
        m = np.median(w)
        kand[t] = abs(v[t] - m) > k * 1.4826 * max(np.median(np.abs(w - m)), eps)
        if kand[t]:
            arah[t] = int(np.sign(v[t] - m))
    kode, i = np.zeros(n, int), 0
    while i < n:
        if kand[i]:
            j = i
            while j + 1 < n and kand[j + 1]:
                j += 1
            kode[i : j + 1] = 2 if (j - i + 1) > maks_run else 1
            i = j + 1
        else:
            i += 1
    return (kode, arah) if return_arah else kode


def isi_celah_pendek(s, maks=BATAS_CELAH, desimal=1, larang=None):
    """Interpolasi linear celah <= maks hari yang diapit observasi; mengembalikan (nilai, penanda, tersedia_sejak)."""
    v = np.asarray(s, float).copy()
    n, idx = len(v), s.index
    larang = np.zeros(n, bool) if larang is None else np.asarray(larang, bool)
    flag = np.where(np.isnan(v), 9, 0)
    ts = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    i = 0
    while i < n:
        if np.isnan(v[i]):
            j = i
            while j + 1 < n and np.isnan(v[j + 1]):
                j += 1
            if i > 0 and j < n - 1 and (j - i + 1) <= maks and not larang[i : j + 1].any():
                v[i : j + 1] = np.interp(np.arange(i, j + 1), [i - 1, j + 1], [v[i - 1], v[j + 1]])
                if desimal is not None:
                    v[i : j + 1] = np.round(v[i : j + 1], desimal)
                flag[i : j + 1] = 1
                ts[i : j + 1] = idx[j + 1].to_datetime64().astype("datetime64[ns]")
            i = j + 1
        else:
            i += 1
    return pd.Series(v, index=idx), pd.Series(flag, index=idx), pd.Series(ts, index=idx)


def praproses_data_baru(d, data_lama):
    """Berkas Admin yang sudah divalidasi -> baris berskema 32 kolom (Hampel, interpolasi, penanda)."""
    keluar = []
    for s, b in d.groupby("stasiun"):
        lama = data_lama.loc[data_lama["stasiun"] == s, "tanggal"]
        mulai = lama.max() + pd.Timedelta(days=1) if len(lama) else b["tanggal"].min()
        idx = pd.date_range(mulai, b["tanggal"].max(), freq="D")
        b = b.set_index("tanggal").reindex(idx)
        o = pd.DataFrame({"tanggal": idx, "stasiun": s, "bagian": "uji", "periode_struktural_2022": 0})
        for p in PARAM:
            nilai, flag, ts = isi_celah_pendek(b[p])
            o[p] = nilai.to_numpy()
            o[f"{p}_flag"] = flag.to_numpy()
            o[f"{p}_pencilan"] = tanda_hampel(b[p])
            o[f"{p}_tersedia_sejak"] = ts.to_numpy()
        for m in METEO:
            o[m] = b[m].interpolate(limit=BATAS_CELAH, limit_area="inside").to_numpy()
        keluar.append(o)
    return pd.concat(keluar, ignore_index=True)[KOLOM_DATASET]


def _periksa_tabel(df, kolom_wajib, galat):
    """Pemeriksaan umum: kolom, tanggal, tipe angka, dan rentang nilai. Mengembalikan tabel bersih atau None."""
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    hilang = [c for c in kolom_wajib if c not in d.columns]
    if hilang:
        galat.append("Kolom wajib tidak ditemukan: " + ", ".join(hilang) + ".")
        return None
    d = d[kolom_wajib].copy()
    if d.empty:
        galat.append("Berkas tidak berisi baris data.")
        return None
    d["tanggal"] = pd.to_datetime(d["tanggal"], errors="coerce")
    if d["tanggal"].isna().any():
        galat.append(
            f"{int(d['tanggal'].isna().sum())} baris memiliki tanggal yang tidak dapat dibaca (gunakan format"
            " YYYY-MM-DD)."
        )
        return None
    for c in list(PARAM) + list(METEO):
        asli = d[c]
        d[c] = pd.to_numeric(asli, errors="coerce")
        rusak = int((d[c].isna() & asli.notna() & (asli.astype(str).str.strip() != "")).sum())
        if rusak:
            galat.append(f"Kolom {c}: {rusak} nilai bukan angka.")
    for c in PARAM:
        luar = int(((d[c] < 0) | (d[c] > 500)).sum())
        if luar:
            galat.append(f"Kolom {c}: {luar} nilai berada di luar rentang indeks ISPU (0 sampai 500).")
    for c, (lo, hi) in RENTANG_METEO.items():
        luar = int(((d[c] < lo) | (d[c] > hi)).sum())
        if luar:
            galat.append(f"Kolom {c}: {luar} nilai berada di luar rentang wajar ({lo} sampai {hi}).")
    return d


def validasi_unggahan(df):
    """Berkas User: minimal 3 hari terakhir berurutan dengan keenam parameter, dan meteorologi 2 hari terakhir."""
    galat = []
    d = _periksa_tabel(df, KOLOM_UNGGAH, galat)
    if d is None or galat:
        return None, galat
    if d["tanggal"].duplicated().any():
        galat.append("Terdapat tanggal ganda pada berkas.")
        return None, galat
    d = d.sort_values("tanggal").reset_index(drop=True)
    if len(d) < 3:
        galat.append(f"Jumlah hari tidak mencukupi: berkas berisi {len(d)} hari, minimal 3 hari terakhir diperlukan.")
        return None, galat
    if not (d["tanggal"].tail(3).diff().dropna() == pd.Timedelta(days=1)).all():
        galat.append("Tiga hari terakhir pada berkas harus berurutan tanpa celah tanggal.")
        return None, galat
    kosong = [c for c in PARAM if d[c].tail(3).isna().any()]
    if kosong:
        galat.append("Nilai ISPU kosong pada tiga hari terakhir untuk: " + ", ".join(kosong) + ".")
    kosong = [c for c in METEO if d[c].tail(2).isna().any()]
    if kosong:
        galat.append("Nilai meteorologi kosong pada dua hari terakhir untuk: " + ", ".join(kosong) + ".")
    return (None if galat else d), galat


def validasi_data_baru(df, data_lama):
    """Berkas Admin: format, kode stasiun, tanggal ganda, nilai wajar, dan tidak tumpang tindih dengan data lama."""
    galat = []
    d = _periksa_tabel(df, KOLOM_BARU, galat)
    if d is None or galat:
        return None, galat
    d["stasiun"] = d["stasiun"].astype(str).str.strip().str.upper()
    asing = sorted(set(d["stasiun"]) - set(STASIUN))
    if asing:
        galat.append("Kode stasiun tidak dikenal: " + ", ".join(asing) + " (yang sah: DKI1 sampai DKI5).")
    if d.duplicated(["stasiun", "tanggal"]).any():
        galat.append("Terdapat kombinasi stasiun dan tanggal yang ganda pada berkas.")
    kosong = [c for c in METEO if d[c].isna().any()]
    if kosong:
        galat.append("Variabel meteorologi tidak boleh kosong: " + ", ".join(kosong) + ".")
    akhir = data_lama.groupby("stasiun")["tanggal"].max()
    for s, b in d.groupby("stasiun"):
        if s in akhir.index and (b["tanggal"] <= akhir[s]).any():
            galat.append(
                f"{s}: terdapat tanggal yang sudah ada pada data historis. Unggah hanya data setelah"
                f" {tgl_id(akhir[s])}."
            )
    return (None if galat else d.sort_values(["stasiun", "tanggal"]).reset_index(drop=True)), galat


def harian_penuh(g):
    """Memastikan deret satu stasiun harian tanpa celah tanggal (hari hilang menjadi baris kosong berpenanda 9)."""
    g = g.drop_duplicates("tanggal").sort_values("tanggal")
    idx = pd.date_range(g["tanggal"].min(), g["tanggal"].max(), freq="D")
    if len(idx) == len(g):
        return g.reset_index(drop=True)
    s = g["stasiun"].iloc[0]
    g = g.set_index("tanggal").reindex(idx)
    g.index.name = "tanggal"
    g["stasiun"] = s
    for p in PARAM:
        g[f"{p}_flag"] = g[f"{p}_flag"].fillna(9)
    return g.reset_index()


def bentuk_fitur(g):
    """Fitur time-safe hari t: 26 dasar + 15 tambahan; nilai interpolasi dipakai hanya bila tersedia_sejak <= t."""
    f = {}
    for p in PARAM:
        for L in (0, 1, 2):
            fl, ts = g[f"{p}_flag"].shift(L), g[f"{p}_tersedia_sejak"].shift(L)
            ok = (fl == 0) | ((fl == 1) & (ts <= g["tanggal"]))
            f[f"{p}_t{L}"] = g[p].shift(L).where(ok)
    for m in METEO:
        for L in (0, 1):
            f[f"{m}_t{L}"] = g[m].shift(L)
    for p in PARAM:
        obs = g[p].where(g[f"{p}_flag"] == 0)
        f[f"{p}_rm7"] = obs.rolling(7, min_periods=3).mean()
        f[f"{p}_rm30"] = obs.rolling(30, min_periods=10).mean()
    hari = pd.to_datetime(g["tanggal"])
    f["musim_sin"] = np.sin(2 * np.pi * hari.dt.dayofyear / 365.25)
    f["musim_cos"] = np.cos(2 * np.pi * hari.dt.dayofyear / 365.25)
    f["hari_minggu"] = hari.dt.dayofweek
    return pd.DataFrame(f, index=g.index)


def basis(X, p):
    """Nilai acuan target selisih: nilai hari t; bila kosong t-1, lalu t-2, lalu rata-rata 7 hari."""
    return X[f"{p}_t0"].fillna(X[f"{p}_t1"]).fillna(X[f"{p}_t2"]).fillna(X[f"{p}_rm7"])


def matriks_stasiun(X, stasiun):
    """Fitur model untuk satu stasiun: fitur dasar + tambahan + kode stasiun (1 untuk stasiun itu, 0 lainnya)."""
    Z = X.reindex(columns=FITUR_DASAR + FITUR_TAMBAHAN).copy()
    for s in STASIUN:
        Z[f"st_{s}"] = int(s == stasiun)
    return Z[FITUR_MODEL]


class ModelISPU:
    """Model satu (stasiun, algoritma, parameter, horizon); prediksi = basis + tebakan selisih."""

    def __init__(self, est, parameter, stasiun, target_selisih=True):
        self.est, self.parameter, self.stasiun, self.target_selisih = est, parameter, stasiun, target_selisih

    def __getstate__(self):
        # Isi memori XGBoost tidak terjamin terbaca di sistem operasi lain; simpan dalam format model resminya.
        state = self.__dict__.copy()
        if type(self.est).__module__.startswith("xgboost"):
            state["est"] = None
            state["xgb_model"] = bytes(self.est.get_booster().save_raw("ubj"))
            state["xgb_params"] = self.est.get_params()
        return state

    def __setstate__(self, state):
        state = dict(state)
        raw, params = state.pop("xgb_model", None), state.pop("xgb_params", None)
        self.__dict__.update(state)
        if raw is not None:
            import xgboost as xgb

            self.est = xgb.XGBRegressor(**params)
            self.est.load_model(bytearray(raw))

    def predict(self, X):
        yh = np.asarray(self.est.predict(matriks_stasiun(X, self.stasiun)), float)
        if self.target_selisih:
            yh = yh + basis(X, self.parameter).to_numpy(float)
        return yh


def prediksi_model(m, X):
    if m is None:
        return np.full(len(X), np.nan)
    return m.predict(X)


def fitur_stasiun(g):
    g = harian_penuh(g)
    return g, bentuk_fitur(g)


def target(g, p, h):
    """ISPU observasi asli (penanda 0) pada hari t+h, sejajar dengan baris hari t."""
    return g[p].where(g[f"{p}_flag"] == 0).shift(-h)


def _hp_kustom():
    f = DIR_MODEL / "hiperparameter.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def hiperparameter(nama, p, h):
    """Bawaan, lalu ditimpa 'hiperparameter.json' (kunci 'XGBoost' atau 'XGBoost|pm25|3' hasil penalaan notebook)."""
    hp = dict(HP_DEFAULT[nama])
    kus = _hp_kustom()
    hp.update(kus.get(nama, {}))
    hp.update(kus.get(f"{nama}|{p}|{h}", {}))
    return hp


def buat_model(nama, p, h):
    hp = hiperparameter(nama, p, h)
    hp["objective"] = OBJEKTIF[nama]
    if nama == "XGBoost":
        import xgboost as xgb

        return xgb.XGBRegressor(**hp)
    import lightgbm as lgb

    return lgb.LGBMRegressor(**hp)


def baris_gabungan(siap, p, h, batas=None, bagian=None):
    """Baris latih kelima stasiun untuk (parameter, horizon): (stasiun, fitur, target selisih, basis)."""
    st, Z, y, b = [], [], [], []
    for s, (g, X) in siap.items():
        tgt, bs = target(g, p, h), basis(X, p)
        ok = tgt.notna() & bs.notna()
        if bagian is not None:
            ok &= (g["bagian"] == bagian) & (g["bagian"].shift(-h) == bagian)
        if batas is not None:
            ok &= (g["tanggal"] + pd.Timedelta(days=h)) <= pd.Timestamp(batas)
        st.append(np.full(int(ok.sum()), s))
        Z.append(matriks_stasiun(X[ok], s))
        y.append(tgt[ok].to_numpy(float))
        b.append(bs[ok].to_numpy(float))
    st, y, b = np.concatenate(st), np.concatenate(y), np.concatenate(b)
    return st, pd.concat(Z, ignore_index=True), y - b, b


def bobot_sampel(st, stasiun, bobot=None):
    """Bobot 1 untuk baris stasiun sendiri, BOBOT_STASIUN_LAIN untuk baris stasiun lain."""
    w = BOBOT_STASIUN_LAIN if bobot is None else bobot
    return np.where(st == stasiun, 1.0, w)


def latih_berbobot(data, batas_latih, min_baris=30, progres=None, bobot=None):
    """420 model: tiap stasiun dilatih dari kelima stasiun dengan bobot sampel 1 (sendiri) dan BOBOT_STASIUN_LAIN."""
    daftar = _stasiun_ada(data)
    siap = {s: fitur_stasiun(data[data["stasiun"] == s]) for s in daftar}
    model = {s: {} for s in daftar}
    tugas = [(a, p, h) for p in PARAM for h in HORIZON for a in ALGORITMA]
    for i, (a, p, h) in enumerate(tugas):
        if progres:
            progres(
                i / len(tugas), f"Melatih {a} {PARAM[p]} H+{h} untuk {len(daftar)} stasiun ({i + 1} dari {len(tugas)})…"
            )
        st, Z, y, _ = baris_gabungan(siap, p, h, batas=batas_latih)
        for s in daftar:
            if (st == s).sum() < min_baris:
                model[s][(a, p, h)] = None
                continue
            m = buat_model(a, p, h)
            w = bobot_sampel(st, s, bobot)
            pakai = w > 0  # bobot 0 = baris stasiun lain tidak dipakai (per stasiun murni)
            m.fit(Z[pakai], y[pakai], sample_weight=w[pakai])
            model[s][(a, p, h)] = ModelISPU(m, p, s)
    return model


def simpan_model(folder, kode, model):
    import joblib

    Path(folder).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(folder) / f"{kode}.joblib", compress=3)


def muat_model(folder, kode):
    import joblib

    return joblib.load(Path(folder) / f"{kode}.joblib")


def jumlah_model(folder):
    """Jumlah objek model terlatih yang benar-benar tersimpan (bukan None) di folder versi."""
    n = 0
    for f in sorted(Path(folder).glob("DKI*.joblib")):
        n += sum(m is not None for m in muat_model(folder, f.stem).values())
    return n


def metrik(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    e = y - yh
    sst = ((y - y.mean()) ** 2).sum()
    return dict(
        mae=float(np.abs(e).mean()),
        rmse=float(np.sqrt((e**2).mean())),
        r2=float(1 - (e**2).sum() / sst) if len(y) > 1 and sst > 0 else np.nan,
    )


def evaluasi_stasiun(g, model, batas_latih, kode, kode_buang=()):
    """MAE, RMSE, R2 pada data uji (hanya target observasi asli), beserta persistensi."""
    g, X = fitur_stasiun(g)
    uji = g["tanggal"] > pd.Timestamp(batas_latih)
    baris = []
    for p in PARAM:
        for h in HORIZON:
            y, t0 = target(g, p, h), X[f"{p}_t0"]
            ok = uji & y.notna() & t0.notna()
            if kode_buang:
                ok &= ~g[f"{p}_pencilan"].shift(-h).isin(kode_buang)
            if not ok.any():
                continue
            yo = y[ok].to_numpy()
            pred = {"Persistensi": t0[ok].to_numpy()}
            for a in ALGORITMA:
                if model.get((a, p, h)) is not None:
                    pred[a] = np.clip(prediksi_model(model[(a, p, h)], X[ok]), 0, None)
            for a, yh in pred.items():
                baris.append(dict(stasiun=kode, parameter=p, horizon=h, algoritma=a, **metrik(yo, yh), n=int(ok.sum())))
    return baris


def _stasiun_ada(data):
    return [s for s in STASIUN if (data["stasiun"] == s).any()]


def latih_semua(data, batas_latih, folder, progres=None):
    """Melatih dan menyimpan model seluruh stasiun, lalu mengevaluasinya pada data setelah batas latih."""
    skala = (lambda f, t: progres(0.9 * f, t)) if progres else None
    semua, hasil = latih_berbobot(data, batas_latih, progres=skala), []
    for s, model in semua.items():
        if progres:
            progres(0.92, f"Menyimpan dan mengevaluasi model {s}…")
        simpan_model(folder, s, model)
        hasil += evaluasi_stasiun(data[data["stasiun"] == s], model, batas_latih, s)
    if progres:
        progres(1.0, "Pelatihan selesai.")
    return pd.DataFrame(hasil)


def evaluasi_model_lama(data, folder_lama, batas_uji):
    """Mengevaluasi model versi lama pada segmen uji yang sama dengan model baru (tanpa persistensi)."""
    hasil = []
    for s in _stasiun_ada(data):
        hasil += evaluasi_stasiun(data[data["stasiun"] == s], muat_model(folder_lama, s), batas_uji, s)
    d = pd.DataFrame(hasil)
    return d[d["algoritma"] != "Persistensi"]


# jenis: "int" = bilangan bulat seragam (batas termasuk), "uni" = seragam, "log" = seragam pada skala logaritma
RUANG_CARI = {
    "XGBoost": dict(
        n_estimators=("int", 100, 600),
        learning_rate=("log", 0.01, 0.10),
        max_depth=("int", 2, 5),
        min_child_weight=("int", 1, 10),
        subsample=("uni", 0.6, 1.0),
        colsample_bytree=("uni", 0.6, 1.0),
        reg_lambda=("log", 1.0, 20.0),
    ),
    "LightGBM": dict(
        n_estimators=("int", 100, 600),
        learning_rate=("log", 0.01, 0.10),
        num_leaves=("int", 4, 24),
        min_child_samples=("int", 10, 40),
        subsample=("uni", 0.6, 1.0),
        colsample_bytree=("uni", 0.6, 1.0),
        reg_lambda=("log", 1.0, 20.0),
    ),
}
SKENARIO = {"semua": (), "tanpa_kode1": (1,), "tanpa_kode2": (2,), "tanpa_kode1_2": (1, 2)}


def sampel_hiperparameter(nama, rng):
    """Satu kombinasi acak dari ruang pencarian. Subsampel baris dan kolom diundi terpisah."""
    hp = {}
    for k, (jenis, lo, hi) in RUANG_CARI[nama].items():
        if jenis == "int":
            hp[k] = int(rng.integers(lo, hi + 1))
        elif jenis == "log":
            hp[k] = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 5)
        else:
            hp[k] = round(float(rng.uniform(lo, hi)), 5)
    return hp


def kombinasi_acak(nama, p, h, n=N_KOMBINASI, seed=SEED):
    """n kombinasi untuk satu pencarian. Aliran acaknya ditentukan oleh (benih, algoritma, parameter, horizon)."""
    rng = np.random.default_rng([seed, ALGORITMA.index(nama), list(PARAM).index(p), int(h)])
    return [sampel_hiperparameter(nama, rng) for _ in range(n)]


def model_dari_hp(nama, hp):
    dasar = dict(HP_DEFAULT[nama])
    dasar.update(hp)
    dasar["objective"] = OBJEKTIF[nama]
    if nama == "XGBoost":
        import xgboost as xgb

        return xgb.XGBRegressor(**dasar)
    import lightgbm as lgb

    return lgb.LGBMRegressor(**dasar)


def _bagi_penalaan(siap, p, h, min_baris=30):
    """Baris penalaan: latih kelima stasiun di segmen latih, dan baris validasi per stasiun."""
    st, Z, y, _ = baris_gabungan(siap, p, h, bagian="latih")
    vst, vZ, vy, vb = baris_gabungan(siap, p, h, bagian="validasi")
    val = {}
    for s in siap:
        m = vst == s
        if (st == s).sum() >= min_baris and m.sum() > 0:
            asli = vy[m] + vb[m]
            val[s] = (vZ[m], asli, vb[m])
    return (st, Z, y, val)


def _cari_satu(nama, p, h, bagi, n, seed):
    """Satu pencarian acak; tiap kombinasi dinilai dari MAE validasi rata-rata kelima stasiun."""
    st, Z, y, val = bagi
    log = []
    for i, hp in enumerate(kombinasi_acak(nama, p, h, n, seed), 1):
        mae = {}
        for s, (Zv, yv, bv) in val.items():
            m = model_dari_hp(nama, hp)
            m.fit(Z, y, sample_weight=bobot_sampel(st, s))
            yh = m.predict(Zv) + bv
            mae[s] = float(np.abs(yv - np.clip(yh, 0, None)).mean())
        log.append(
            dict(
                algoritma=nama,
                parameter=p,
                horizon=h,
                kombinasi=i,
                **hp,
                **{f"mae_{s}": v for s, v in mae.items()},
                mae_rata=float(np.mean(list(mae.values()))) if mae else np.nan,
                n_stasiun=len(mae),
            )
        )
    return log


def tala_hiperparameter(
    data,
    n_kombinasi=N_KOMBINASI,
    seed=SEED,
    n_proses=1,
    simpan=True,
    progres=None,
    algoritma=ALGORITMA,
    param=tuple(PARAM),
    horizon=HORIZON,
):
    """Pencarian acak (Tabel 3.3) untuk 2 algoritma x 6 parameter x 7 horizon = 84 pencarian."""
    from joblib import Parallel, delayed

    siap = {s: fitur_stasiun(data[data["stasiun"] == s]) for s in _stasiun_ada(data)}
    tugas = [(a, p, h) for a in algoritma for p in param for h in horizon]
    titik = DIR_MODEL / f"penalaan_sebagian_n{n_kombinasi}_seed{seed}"
    log = []
    if simpan and titik.exists():
        tersimpan = sorted(titik.glob("*.csv"))
        for f in tersimpan:
            log += pd.read_csv(f).to_dict("records")
        selesai = {(r["algoritma"], r["parameter"], int(r["horizon"])) for r in log}
        tugas = [t for t in tugas if t not in selesai]
        if progres and tersimpan:
            progres(0.0, f"Melanjutkan penalaan: {len(selesai)} pencarian sudah selesai, sisa {len(tugas)}")
    sudah = len({(r["algoritma"], r["parameter"], int(r["horizon"])) for r in log})
    total = sudah + len(tugas)
    kerja = (delayed(_cari_satu)(a, p, h, _bagi_penalaan(siap, p, h), n_kombinasi, seed) for a, p, h in tugas)
    try:
        aliran = Parallel(n_jobs=n_proses, return_as="generator")(kerja)
    except TypeError:  # joblib lama tanpa return_as
        aliran = iter(Parallel(n_jobs=n_proses)(kerja))
    if simpan:
        titik.mkdir(parents=True, exist_ok=True)
    for i, ((a, p, h), hasil) in enumerate(zip(tugas, aliran), 1):
        log += hasil
        if simpan:
            sementara = titik / f"{a}_{p}_H{h}.tmp"
            pd.DataFrame(hasil).to_csv(sementara, index=False)
            sementara.replace(titik / f"{a}_{p}_H{h}.csv")
        if progres:
            progres((sudah + i) / total, f"Pencarian {sudah + i} dari {total}: {a} {p} H+{h}")
    urutan = {t: i for i, t in enumerate((a, p, h) for a in algoritma for p in param for h in horizon)}
    log = pd.DataFrame(
        sorted(log, key=lambda r: (urutan[(r["algoritma"], r["parameter"], int(r["horizon"]))], r["kombinasi"]))
    )
    log["horizon"] = log["horizon"].astype(int)
    terbaik = {}
    for (a, p, h), g in log[log["n_stasiun"] > 0].groupby(["algoritma", "parameter", "horizon"], sort=False):
        r = g.loc[g["mae_rata"].idxmin()]
        terbaik[f"{a}|{p}|{int(h)}"] = {k: _asli(r[k], jenis[0]) for k, jenis in RUANG_CARI[a].items()}
    if simpan:
        (DIR_MODEL / "hiperparameter.json").write_text(
            json.dumps(terbaik, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        log.to_csv(DIR_MODEL / "penalaan_log.csv", index=False)
    return terbaik, log


def _asli(v, jenis):
    """Nilai dari CSV/numpy ke tipe Python sesuai ruang pencarian, agar hiperparameter.json tetap sama."""
    return int(v) if jenis == "int" else float(v)


def batas_validasi(data):
    """Tanggal terakhir segmen validasi = batas pelatihan akhir (latih + validasi), sebelum data uji."""
    v = data.loc[data["bagian"] == "validasi", "tanggal"]
    return v.max() if len(v) else BATAS_LATIH_AWAL


def evaluasi_skenario(data, folder, batas_latih, skenario=SKENARIO):
    """Mengevaluasi model tersimpan pada data uji untuk tiap skenario sensitivitas (kolom 'skenario')."""
    hasil = []
    for s in _stasiun_ada(data):
        g, m = data[data["stasiun"] == s], muat_model(folder, s)
        for nama, buang in skenario.items():
            for r in evaluasi_stasiun(g, m, batas_latih, s, buang):
                hasil.append(dict(skenario=nama, **r))
    return pd.DataFrame(hasil)


def latih_akhir(data, folder=None, progres=None):
    """Pelatihan akhir pada latih + validasi, evaluasi sekali pada data uji, dan analisis sensitivitas pencilan."""
    folder = Path(folder or DIR_MODEL / "v1")
    batas = batas_validasi(data)
    ev = latih_semua(data, batas, folder, progres)
    sens = evaluasi_skenario(data, folder, batas)
    meta = dict(
        batas_latih=str(pd.Timestamp(batas).date()),
        data_hingga=str(pd.Timestamp(data["tanggal"].max()).date()),
        dibuat=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        seed=SEED,
        n_sel_model=int(len(ev[ev["algoritma"] != "Persistensi"])),
        jumlah_model=jumlah_model(folder),
        desain="berbobot",
        bobot_stasiun_lain=BOBOT_STASIUN_LAIN,
        loss="l1",
        target_selisih=True,
        jumlah_fitur=len(FITUR_MODEL),
        hiperparameter_json=(DIR_MODEL / "hiperparameter.json").exists(),
    )
    ev.to_csv(folder / "evaluasi.csv", index=False)
    sens.to_csv(folder / "evaluasi_sensitivitas.csv", index=False)
    (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return ev, sens, meta


def prediksi_terbaru(g, model):
    """Prediksi H+1..H+7 dari baris terakhir yang memiliki data. Mengembalikan (tabel panjang, fitur, tanggal t)."""
    g, X = fitur_stasiun(g)
    ada = g[list(PARAM)].notna().any(axis=1)
    idx = ada[ada].index[-1]
    x = X.loc[[idx]]
    baris = []
    for (a, p, h), m in model.items():
        v = float(prediksi_model(m, x)[0])
        nilai = np.nan if np.isnan(v) else max(v, 0.0)
        baris.append(dict(algoritma=a, parameter=PARAM[p], kode=p, horizon=h, nilai=nilai))
    return pd.DataFrame(baris), x, g.loc[idx, "tanggal"]


def prediksi_unggahan(d, model):
    """Prediksi dari berkas User yang sudah lolos validasi. Semua nilai dianggap observasi asli (penanda 0)."""
    g = d.copy()
    g["stasiun"] = "UNGGAH"
    for p in PARAM:
        g[f"{p}_flag"] = np.where(g[p].notna(), 0, 9)
        g[f"{p}_tersedia_sejak"] = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns]")
    return prediksi_terbaru(g, model)


def prediksi_uji(g, model, p, h, batas_latih):
    """Aktual dan prediksi kedua algoritma pada periode uji, diindeks menurut tanggal target (t+h)."""
    g, X = fitur_stasiun(g)
    y = target(g, p, h)
    uji = (g["tanggal"] > pd.Timestamp(batas_latih)) & ((g["tanggal"] + pd.Timedelta(days=h)) <= g["tanggal"].max())
    out = pd.DataFrame({"tanggal_target": g.loc[uji, "tanggal"] + pd.Timedelta(days=h), "aktual": y[uji]})
    for a in ALGORITMA:
        m = model.get((a, p, h))
        out[a] = np.nan if m is None else np.clip(prediksi_model(m, X[uji]), 0, None)
    return out.reset_index(drop=True)


def prediksi_lampau(g, model, p, h, mulai, akhir):
    """Aktual dan prediksi H+h per tanggal target pada [mulai, akhir], untuk grafik Beranda."""
    g, X = fitur_stasiun(g)
    tgl_target = g["tanggal"] + pd.Timedelta(days=h)
    ok = (tgl_target >= pd.Timestamp(mulai)) & (tgl_target <= pd.Timestamp(akhir))
    out = pd.DataFrame({"tanggal_target": tgl_target[ok]})
    idx_target = g.set_index("tanggal")
    out["aktual"] = idx_target[p].reindex(out["tanggal_target"]).to_numpy()
    out["penanda"] = idx_target[f"{p}_flag"].reindex(out["tanggal_target"]).fillna(9).astype(int).to_numpy()
    for a in ALGORITMA:
        m = model.get((a, p, h))
        out[a] = np.nan if m is None else np.clip(prediksi_model(m, X[ok]), 0, None)
    return out.reset_index(drop=True)


LEVEL = {
    "Parameter": ["parameter"],
    "Horizon": ["horizon"],
    "Stasiun": ["stasiun"],
    "Stasiun × parameter": ["stasiun", "parameter"],
    "Parameter × horizon": ["parameter", "horizon"],
    "Stasiun × parameter × horizon": ["stasiun", "parameter", "horizon"],
}


def model_terbaik(ev, level, metrik="r2"):
    """Peringkat model per kelompok pada level tertentu menurut metrik, tanpa persistensi."""
    e = ev[ev["algoritma"].isin(ALGORITMA)]
    agg = (
        e.groupby(level + ["algoritma"])
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean"), n=("n", "sum"))
        .reset_index()
    )
    besar = metrik == "r2"
    agg = agg.sort_values(metrik, ascending=not besar)
    terbaik = agg.groupby(level, as_index=False).head(1)
    terbaik = terbaik.sort_values(metrik, ascending=not besar).reset_index(drop=True)
    terbaik.insert(0, "peringkat", np.arange(1, len(terbaik) + 1))
    return terbaik


def peringkat_semua(ev, metrik="r2"):
    """Seluruh model (stasiun x parameter x horizon x algoritma) tanpa persistensi, diurutkan menurut metrik."""
    e = ev[ev["algoritma"].isin(ALGORITMA)].copy()
    e = e.sort_values(metrik, ascending=metrik != "r2").reset_index(drop=True)
    e.insert(0, "peringkat", np.arange(1, len(e) + 1))
    return e


def horizon_terjauh(ev, ambang=0.5, per=("parameter",)):
    """Horizon terjauh per kelompok yang R2 rata-rata algoritma terbaiknya masih >= ambang."""
    e = ev[ev["algoritma"].isin(ALGORITMA)]
    r = e.groupby(list(per) + ["horizon", "algoritma"])["r2"].mean().reset_index()
    r = r.groupby(list(per) + ["horizon"])["r2"].max().reset_index()
    baris = []
    for k, gg in r.groupby(list(per)):
        gg = gg.sort_values("horizon")
        jauh = 0
        for _, x in gg.iterrows():
            if x["r2"] >= ambang:
                jauh = int(x["horizon"])
            else:
                break
        k = k if isinstance(k, tuple) else (k,)
        baris.append(
            {
                **dict(zip(per, k)),
                "horizon_terjauh": jauh,
                **{f"R² H+{int(x['horizon'])}": x["r2"] for _, x in gg.iterrows()},
            }
        )
    return pd.DataFrame(baris)


def tabel_prediksi_uji(data, folder, batas_latih):
    """Seluruh pasangan aktual-prediksi pada periode uji, satu baris per (stasiun, parameter, horizon, tanggal)."""
    keluar = []
    for s in _stasiun_ada(data):
        g, X = fitur_stasiun(data[data["stasiun"] == s])
        model = muat_model(folder, s)
        uji = g["tanggal"] > pd.Timestamp(batas_latih)
        for p in PARAM:
            for h in HORIZON:
                ok = uji & ((g["tanggal"] + pd.Timedelta(days=h)) <= g["tanggal"].max())
                d = pd.DataFrame(
                    {
                        "stasiun": s,
                        "parameter": p,
                        "horizon": h,
                        "tanggal_dasar": g.loc[ok, "tanggal"],
                        "tanggal_target": g.loc[ok, "tanggal"] + pd.Timedelta(days=h),
                        "aktual": target(g, p, h)[ok],
                        "Persistensi": X.loc[ok, f"{p}_t0"],
                        "pencilan": g[f"{p}_pencilan"].shift(-h)[ok],
                    }
                )
                for a in ALGORITMA:
                    d[a] = np.clip(prediksi_model(model.get((a, p, h)), X[ok]), 0, None)
                keluar.append(d)
    return pd.concat(keluar, ignore_index=True)


def analisis_puncak(pred, q=0.9, batas_tidak_sehat=100):
    """Bias pada hari puncak (>= P90) serta recall dan presisi hari tidak sehat (> 100) per parameter dan metode."""
    d = pred.dropna(subset=["aktual", "Persistensi"]).copy()
    d["batas_q"] = d.groupby(["stasiun", "parameter"])["aktual"].transform(lambda s: s.quantile(q))
    rows = []
    for p, g in d.groupby("parameter"):
        pk, ts = g[g["aktual"] >= g["batas_q"]], g["aktual"] > batas_tidak_sehat
        for a in ("Persistensi",) + ALGORITMA:
            e = g[a] - g["aktual"]
            ya = g[a] > batas_tidak_sehat
            rows.append(
                {
                    "parameter": p,
                    "metode": a,
                    "mae": e.abs().mean(),
                    "rmse": np.sqrt((e**2).mean()),
                    "bias": e.mean(),
                    "mae_hari_puncak": (pk[a] - pk["aktual"]).abs().mean(),
                    "bias_hari_puncak": (pk[a] - pk["aktual"]).mean(),
                    "hari_tidak_sehat": int(ts.sum()),
                    "recall_tidak_sehat_persen": 100 * (ya & ts).sum() / ts.sum() if ts.sum() else np.nan,
                    "presisi_tidak_sehat_persen": 100 * (ya & ts).sum() / ya.sum() if ya.sum() else np.nan,
                }
            )
    return pd.DataFrame(rows)
