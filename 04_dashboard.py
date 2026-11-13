# -*- coding: utf-8 -*-
"""
04_dashboard.py
ASI-EWS: Adaptive Agroclimatic Stress Intelligence –
Sistem Peringatan Dini Cekaman Atmosfer dan Lahan Pertanian
Berbasis Machine Learning dan Data Satelit di Jawa Barat

Fitur:
- Dual-Target ML (ASI & NDVI) terintegrasi pipeline 02_ml_v8.py
- Panduan Operasional Petani & Dinas Pertanian + Format Pesan WhatsApp
- Galeri Gambar Lokal & Ekspor Batch Confusion Matrix Resolusi Tinggi untuk KTI
- Kontrol Basemap Peta langsung di area utama (Desktop & Mobile Friendly)

Jalankan lokal:
py -m streamlit run D:\LOMBAAGRI\04_dashboard.py
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. KONFIGURASI PATH & JUDUL KTI
# ============================================================

JUDUL_SINGKAT = "ASI-EWS Jawa Barat"
JUDUL_KTI = (
    "ASI-EWS: Adaptive Agroclimatic Stress Intelligence – "
    "Sistem Peringatan Dini Cekaman Atmosfer dan Lahan Pertanian "
    "Berbasis Machine Learning dan Data Satelit di Jawa Barat"
)

BASE_DIR = Path(__file__).resolve().parent
LOKAL_OPTIONAL = Path(r"D:\LOMBAAGRI\OUTPUT_ASI_JABAR")
OUT = LOKAL_OPTIONAL if LOKAL_OPTIONAL.exists() else (BASE_DIR / "OUTPUT_ASI_JABAR")

st.set_page_config(
    page_title=JUDUL_KTI,
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "halaman" not in st.session_state:
    st.session_state.halaman = "beranda"


def pindah_ke(nama_halaman: str):
    st.session_state.halaman = nama_halaman


WARNA_RISIKO = {
    "Normal": "#239B75",
    "Waspada": "#E9A23B",
    "Tinggi": "#D9535D",
}

PUSAT_PETA = {"lat": -6.95, "lon": 107.60}

MAP_STYLES = {
    "Carto Positron (Terang & Bersih)": "carto-positron",
    "Carto Voyager (Jalan & Batas Wilayah)": "carto-voyager",
    "OpenStreetMap (Detail Lapangan)": "open-street-map",
    "Carto Dark Matter (Mode Gelap)": "carto-darkmatter",
}

LABEL_TARGET = {
    "asi": "Cekaman Agroklimat (ASI — Target Utama)",
    "ndvi": "Respons Vegetasi (Anomali NDVI — Target Pendukung)",
}


def konfig_unduh(nama_file: str) -> dict:
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "autoScale2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": nama_file,
            "scale": 3,
        },
    }


def rekom_aksi_lapangan(status_asi: str, status_ndvi: str, pemicu: str) -> dict:
    """Menerjemahkan output teknis ML menjadi panduan praktis Petani & Dinas."""
    pemicu_str = str(pemicu) if pd.notna(pemicu) and str(pemicu).strip() else "Kombinasi atmosfer & kelembapan tanah"

    if status_asi == "Tinggi" or status_ndvi == "Tinggi":
        return {
            "level": "SIAGA MERAH (PRIORITAS INTERVENSI)",
            "warna": "#D9535D",
            "bg": "#FDF0F1",
            "ringkas_petani": (
                "Risiko cekaman kekeringan/panas tinggi pada lahan sawah dalam 1–3 bulan ke depan. "
                "Tanaman rentan mengalami penurunan pertumbuhan vegetatif jika pasokan air terlambat."
            ),
            "aksi_petani": [
                "Terapkan irigasi berselang (AWD / macak-macak) dan hindari penggenangan berlebih agar stok air irigasi bertahan lebih lama.",
                "Siapkan pompa air kelompok tani (poktan) dan manfaatkan embung/sumur pantek di wilayah tadah hujan.",
                "Gunakan mulsa jerami di pematang/lahan untuk menekan penguapan air tanah, serta tunda pemupukan N dosis tinggi saat puncak panas.",
                "Untuk jadwal tanam berikutnya, utamakan varietas padi toleran kekeringan (misal: Inpari 38, Inpari 39, atau Situ Bagendit).",
            ],
            "aksi_dinas": [
                "Prioritaskan alokasi bantuan pompa air, BBM subsidi pertanian, dan rehabilitasi jaringan irigasi tersier di kabupaten ini.",
                "Koordinasikan jadwal buka-tutup pintu air (gilir giring) bersama BBWS/PSDA untuk wilayah hilir.",
                "Aktifkan pemantauan lapangan mingguan oleh PPL dan petugas POPT untuk antisipasi serangan hama sekunder (wereng/penggerek) saat cuaca panas.",
            ],
            "pemicu": pemicu_str,
        }
    elif status_asi == "Waspada" or status_ndvi == "Waspada":
        return {
            "level": "WASPADA KUNING (ANTISIPASI DINI)",
            "warna": "#D98A1C",
            "bg": "#FEF8EC",
            "ringkas_petani": (
                "Mulai terdeteksi peningkatan tekanan cuaca kering/hangat. Kondisi tanaman saat ini masih terkendali, "
                "namun perlu pengaturan air dan jadwal tanam lebih disiplin."
            ),
            "aksi_petani": [
                "Perbaiki saluran cacing dan tanggul petakan sawah untuk mencegah kebocoran air.",
                "Lakukan percepatan tanam (tanam serempak) memanfaatkan sisa lengas tanah yang masih tersedia.",
                "Gunakan pupuk organik/kompos untuk meningkatkan daya ikat air tanah dan ketahanan akar.",
            ],
            "aksi_dinas": [
                "Sebarkan informasi peringatan dini bulanan ke BPP kecamatan dan gabungan kelompok tani (Gapoktan).",
                "Inventarisasi kesiapan cadangan air embung desa dan pompa mobile di sentra padi wilayah ini.",
            ],
            "pemicu": pemicu_str,
        }
    else:
        return {
            "level": "NORMAL HIJAU (KONDISI KONDUSIF)",
            "warna": "#239B75",
            "bg": "#EDF8F4",
            "ringkas_petani": (
                "Kondisi agroklimat dan kehijauan vegetasi sawah berada dalam kisaran normal. "
                "Budidaya padi dapat berjalan sesuai kalender tanam setempat."
            ),
            "aksi_petani": [
                "Lanjutkan pola tanam dan pemupukan berimbang sesuai rekomendasi BPP setempat.",
                "Tetap jaga kebersihan saluran drainase untuk mengantisipasi hujan lebat sesaat.",
            ],
            "aksi_dinas": [
                "Lanjutkan pemantauan rutin bulanan dan pendampingan optimalisasi indeks pertanaman (IP).",
            ],
            "pemicu": pemicu_str,
        }


# ============================================================
# 2. DETEKSI FOLDER OUTPUT (PRIORITAS hasil_v8)
# ============================================================

def deteksi_folder_hasil(out_dir: Path) -> dict:
    kandidat = {}
    for folder_name, label in [
        ("hasil_v8", "Pipeline v8 (Dual-Target ASI & NDVI • B_CWB3)"),
        ("hasil_v7", "Pipeline v7 (D_ENSO_CH • Hindcast Des 2025)"),
        ("hasil_bmkg", "Pipeline v5 (BMKG per ZOM)"),
        ("hasil_chirps", "Pipeline v5 (CHIRPS Satelit)"),
    ]:
        res_dir = out_dir / folder_name
        dash_dir = res_dir / "DASHBOARD_DATA"
        if (dash_dir / "forecast_hindcast_des2025.csv").exists() or (dash_dir / "forecast_terbaru.csv").exists():
            kandidat[folder_name] = {"label": label, "res": res_dir, "dash": dash_dir}

    if not kandidat and (out_dir / "DASHBOARD_DATA").exists():
        kandidat["default"] = {
            "label": "Default Output",
            "res": out_dir,
            "dash": out_dir / "DASHBOARD_DATA",
        }
    return kandidat


FOLDER_TERSEDIA = deteksi_folder_hasil(OUT)
if not FOLDER_TERSEDIA:
    st.error(f"Folder hasil analisis belum ditemukan di: {OUT}")
    st.info(
        "Jalankan skrip `02_ml_v8.py` terlebih dahulu untuk menghasilkan folder "
        "`OUTPUT_ASI_JABAR/hasil_v8/DASHBOARD_DATA`."
    )
    st.stop()

if "folder_aktif" not in st.session_state or st.session_state.folder_aktif not in FOLDER_TERSEDIA:
    st.session_state.folder_aktif = "hasil_v8" if "hasil_v8" in FOLDER_TERSEDIA else next(iter(FOLDER_TERSEDIA))


# ============================================================
# 3. CUSTOM CSS (RESPONSIVE DESKTOP + MOBILE / HP)
# ============================================================

sidebar_css = ""
if st.session_state.halaman == "beranda":
    sidebar_css = """
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    button[kind="header"] {
        display: none !important;
    }
    """

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    .stApp {{
        background-color: #F4F7F5;
        color: #132A24;
    }}

    .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
        max-width: 1440px;
    }}

    {sidebar_css}

    section[data-testid="stSidebar"] {{
        background-color: #0F2E26 !important;
        border-right: 1px solid #1C453B;
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] span[data-testid="stMarkdownContainer"] {{
        color: #E8F3EE !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: #FFFFFF !important;
        border: 1.5px solid #B5D149 !important;
        border-radius: 10px !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="select"] div,
    section[data-testid="stSidebar"] div[data-baseweb="select"] input {{
        color: #0F2E26 !important;
        font-weight: 600 !important;
        -webkit-text-fill-color: #0F2E26 !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{
        fill: #0F2E26 !important;
    }}

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] {{
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 25px rgba(15, 46, 38, 0.15) !important;
    }}

    ul[role="listbox"] li {{
        background-color: #FFFFFF !important;
        color: #0F2E26 !important;
        font-weight: 500 !important;
    }}

    ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"] {{
        background-color: #E6F4EE !important;
        color: #0F2E26 !important;
        font-weight: 700 !important;
    }}

    section[data-testid="stSidebar"] button {{
        background-color: #B5D149 !important;
        color: #0F2E26 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}

    section[data-testid="stSidebar"] button * {{
        color: #0F2E26 !important;
    }}

    .navbar-landing {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #0D2B23;
        padding: 14px 26px;
        border-radius: 16px 16px 0 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        gap: 8px;
    }}

    .brand-logo {{
        font-size: 19px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.4px;
    }}

    .brand-logo span {{
        color: #B5D149;
    }}

    .brand-sub {{
        font-size: 12px;
        color: #9EC2B5;
        font-weight: 500;
    }}

    .landing-hero {{
        background: linear-gradient(130deg, #0D2B23 0%, #15473A 55%, #1E6351 100%);
        border-radius: 0 0 22px 22px;
        padding: 36px 36px;
        color: #FFFFFF;
        margin-bottom: 20px;
        box-shadow: 0 18px 36px rgba(13, 43, 35, 0.16);
    }}

    .landing-badge {{
        display: inline-block;
        background-color: rgba(181, 209, 73, 0.16);
        color: #D2EB73;
        border: 1px solid rgba(181, 209, 73, 0.45);
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }}

    .landing-hero h1 {{
        font-size: 29px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 12px 0;
        line-height: 1.28;
        letter-spacing: -0.4px;
    }}

    .landing-hero .accent {{
        color: #B5D149;
    }}

    .landing-hero p {{
        font-size: 14.5px;
        color: #D5E8E0;
        max-width: 920px;
        line-height: 1.65;
        margin: 0;
    }}

    div.stButton > button[kind="primary"] {{
        background-color: #B5D149 !important;
        color: #0D2B23 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 13px 22px !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 20px rgba(181, 209, 73, 0.35) !important;
        transition: all 0.2s ease !important;
        min-height: 48px !important;
    }}

    div.stButton > button[kind="primary"]:hover {{
        background-color: #C7E063 !important;
        transform: translateY(-1px) !important;
    }}

    div.stButton > button[kind="primary"] * {{
        color: #0D2B23 !important;
    }}

    .pillar-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 14px 0 22px 0;
    }}

    .pillar-card {{
        background: #FFFFFF;
        border: 1px solid #DCE8E2;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(15, 46, 38, 0.04);
    }}

    .pillar-icon {{
        font-size: 22px;
        background: #EAF5F0;
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 11px;
        margin-bottom: 12px;
    }}

    .pillar-title {{
        font-size: 15px;
        font-weight: 700;
        color: #0F2E26;
        margin-bottom: 5px;
    }}

    .pillar-text {{
        font-size: 13px;
        color: #4D665E;
        line-height: 1.5;
    }}

    .impact-strip {{
        background: #0F2E26;
        border-radius: 16px;
        padding: 20px 24px;
        color: #FFFFFF;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 20px;
    }}

    .impact-item h3 {{
        font-size: 20px;
        font-weight: 800;
        color: #B5D149;
        margin: 0 0 4px 0;
        line-height: 1.25;
    }}

    .impact-item p {{
        font-size: 12.5px;
        color: #D5E8E0;
        margin: 0;
        line-height: 1.4;
    }}

    .workspace-topbar {{
        background: linear-gradient(115deg, #0F2E26, #1C5747);
        padding: 16px 22px;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 14px;
    }}

    .workspace-topbar h2 {{
        margin: 0 0 5px 0;
        font-size: 17px;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.35;
    }}

    .workspace-topbar span {{
        font-size: 12px;
        color: #C4E0D5;
        display: block;
        line-height: 1.4;
    }}

    .alert-card-tani {{
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
        border-left: 7px solid;
        box-shadow: 0 6px 18px rgba(15, 46, 38, 0.06);
    }}

    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid #DCE8E2;
        padding: 13px 15px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(15, 46, 38, 0.04);
    }}

    div[data-testid="stMetric"] label {{
        color: #4D665E !important;
        font-weight: 600 !important;
        font-size: 12.5px !important;
    }}

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: #0F2E26 !important;
        font-weight: 800 !important;
        font-size: 19px !important;
    }}

    div[data-testid="stTabs"] [role="tablist"] {{
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        scrollbar-width: none;
        -ms-overflow-style: none;
        gap: 4px;
    }}

    div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {{
        display: none;
    }}

    div[data-testid="stTabs"] button {{
        font-weight: 600;
        font-size: 13.5px;
        padding: 9px 14px;
        border-radius: 10px 10px 0 0;
        color: #3E5750;
        flex-shrink: 0;
    }}

    div[data-testid="stTabs"] button[aria-selected="true"] {{
        background-color: #FFFFFF !important;
        color: #0F2E26 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #239B75 !important;
    }}

    @media (max-width: 900px) {{
        .pillar-grid, .impact-strip {{
            grid-template-columns: repeat(2, 1fr);
        }}
        .landing-hero {{
            padding: 26px 22px;
        }}
        .landing-hero h1 {{
            font-size: 23px;
        }}
    }}

    @media (max-width: 600px) {{
        .block-container {{
            padding-top: 0.75rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }}
        .navbar-landing {{
            flex-direction: column;
            align-items: flex-start;
            padding: 12px 16px;
        }}
        .landing-hero {{
            padding: 22px 18px;
            border-radius: 0 0 18px 18px;
        }}
        .landing-hero h1 {{
            font-size: 20px;
            line-height: 1.32;
        }}
        .landing-hero p {{
            font-size: 13.5px;
        }}
        .pillar-grid, .impact-strip {{
            grid-template-columns: 1fr;
            gap: 10px;
        }}
        .impact-strip {{
            padding: 16px 18px;
        }}
        .impact-item h3 {{
            font-size: 18px;
        }}
        .workspace-topbar {{
            padding: 14px 16px;
            border-radius: 14px;
        }}
        .workspace-topbar h2 {{
            font-size: 15px;
        }}
        div[data-testid="stMetric"] {{
            padding: 11px 12px;
        }}
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            font-size: 17px !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. PEMUATAN DATA DINAMIS SESUAI PIPELINE 02_ml_v8.py
# ============================================================

@st.cache_data(show_spinner=False)
def muat_data(res_dir_str: str, dash_dir_str: str, out_dir_str: str):
    res_dir = Path(res_dir_str)
    dash_dir = Path(dash_dir_str)
    out_dir = Path(out_dir_str)

    fc_path = dash_dir / "forecast_hindcast_des2025.csv"
    if not fc_path.exists():
        fc_path = dash_dir / "forecast_terbaru.csv"

    wajib = {
        "batas_kab_hasil.geojson": dash_dir / "batas_kab_hasil.geojson",
        fc_path.name: fc_path,
        "fitur_bulanan.csv": dash_dir / "fitur_bulanan.csv",
        "prediksi_oos.csv": dash_dir / "prediksi_oos.csv",
        "meta.json": dash_dir / "meta.json",
        "hasil_perbandingan_model.csv": res_dir / "hasil_perbandingan_model.csv",
    }

    belum_ada = [nama for nama, path in wajib.items() if not path.exists()]
    if belum_ada:
        raise FileNotFoundError(
            f"File berikut belum ditemukan di {res_dir}:\n- " + "\n- ".join(belum_ada)
        )

    data = {}
    data["geo"] = json.loads((dash_dir / "batas_kab_hasil.geojson").read_text(encoding="utf-8"))
    data["fc"] = pd.read_csv(fc_path)
    data["fc_filename"] = fc_path.name
    data["fit"] = pd.read_csv(dash_dir / "fitur_bulanan.csv", parse_dates=["tanggal"])
    data["oos"] = pd.read_csv(dash_dir / "prediksi_oos.csv", parse_dates=["tanggal", "tgt_date"])
    data["meta"] = json.loads((dash_dir / "meta.json").read_text(encoding="utf-8"))
    data["res"] = pd.read_csv(res_dir / "hasil_perbandingan_model.csv")

    for key in ("fc", "fit", "oos"):
        if "kab_code" in data[key].columns:
            data[key]["kab_code"] = pd.to_numeric(data[key]["kab_code"], errors="coerce")
            data[key] = data[key].dropna(subset=["kab_code"])
            data[key]["kab_code"] = data[key]["kab_code"].astype(int)

    oos_df = data["oos"]
    if "target" not in oos_df.columns:
        oos_df["target"] = "ndvi"
    if "aktual" not in oos_df.columns and "aktual_anom" in oos_df.columns:
        oos_df["aktual"] = oos_df["aktual_anom"]
    if "pred" not in oos_df.columns and "pred_anom" in oos_df.columns:
        oos_df["pred"] = oos_df["pred_anom"]

    if "target" not in data["res"].columns:
        data["res"]["target"] = "ndvi"

    fc_key = data["meta"].get("skenario_prakiraan", "B_CWB3")

    imp_dict = {}
    for tgt_k in ("asi", "ndvi"):
        p_tgt = res_dir / f"pentingnya_fitur_{tgt_k}_{fc_key}_h1.csv"
        if p_tgt.exists():
            imp_dict[tgt_k] = pd.read_csv(p_tgt, index_col=0)
    if not imp_dict:
        p_legacy = res_dir / f"pentingnya_fitur_{fc_key}_h1.csv"
        if p_legacy.exists():
            imp_dict["ndvi"] = pd.read_csv(p_legacy, index_col=0)
        else:
            kandidat_imp = list(res_dir.glob("pentingnya_fitur_*_h1.csv"))
            if kandidat_imp:
                imp_dict["utama"] = pd.read_csv(kandidat_imp[0], index_col=0)
    data["imp_dict"] = imp_dict

    optional = {
        "eks": dash_dir / "eksposur_sawah.csv",
        "musim": res_dir / "hasil_per_musim.csv",
        "boot": res_dir / "bootstrap_selisih_RMSE.csv",
        "det": res_dir / "deteksi_ambang_risiko.csv",
        "hind_eval": res_dir / "evaluasi_hindcast_des2025.csv",
        "hind_met": res_dir / "metrik_hindcast_des2025.csv",
        "hind_comp_kab": res_dir / "perbandingan_hindcast_des2025_kabupaten.csv",
        "hind_comp_met": res_dir / "perbandingan_metrik_hindcast_des2025.csv",
        "valbps": res_dir / "validasi_BPS.json",
        "sens": res_dir / "sensitivitas_bobot_ASI.json",
        "bps": out_dir / "bps_padi_jabar.csv",
    }

    for key, path in optional.items():
        if not path.exists():
            data[key] = None
        elif path.suffix.lower() == ".json":
            data[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            df_opt = pd.read_csv(path)
            if "kab_code" in df_opt.columns:
                df_opt["kab_code"] = pd.to_numeric(df_opt["kab_code"], errors="coerce")
            data[key] = df_opt

    return data


RES_AKTIF = FOLDER_TERSEDIA[st.session_state.folder_aktif]["res"]
DASH_AKTIF = FOLDER_TERSEDIA[st.session_state.folder_aktif]["dash"]

try:
    D = muat_data(str(RES_AKTIF), str(DASH_AKTIF), str(OUT))
except Exception as exc:
    st.error("Data dashboard belum dapat dimuat.")
    st.code(str(exc))
    st.stop()

geo = D["geo"]
fc = D["fc"].copy()
fit = D["fit"]
oos = D["oos"]
meta = D["meta"]
res = D["res"]
eks = D["eks"]

IS_DUAL_TARGET = "pred_asi" in fc.columns and "peringatan_asi" in fc.columns


def normalisasi_kolom_fc(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    if "indikasi_vegetasi_ndvi" not in df_out.columns:
        if "risiko_penurunan_vegetasi" in df_out.columns:
            df_out["indikasi_vegetasi_ndvi"] = df_out["risiko_penurunan_vegetasi"]
        elif "risiko" in df_out.columns:
            df_out["indikasi_vegetasi_ndvi"] = df_out["risiko"]
    if "indikasi_ndvi_relatif" not in df_out.columns:
        if "risiko_vegetasi_relatif" in df_out.columns:
            df_out["indikasi_ndvi_relatif"] = df_out["risiko_vegetasi_relatif"]
        elif "risiko_relatif" in df_out.columns:
            df_out["indikasi_ndvi_relatif"] = df_out["risiko_relatif"]
    return df_out


fc = normalisasi_kolom_fc(fc)
if D["hind_comp_kab"] is not None:
    D["hind_comp_kab"] = normalisasi_kolom_fc(D["hind_comp_kab"])

for df_tmp, kol in ((oos, "tgt_date"), (fit, "tanggal")):
    df_tmp[kol] = pd.to_datetime(df_tmp[kol], errors="coerce")

nama_kab = fit.drop_duplicates("kab_code").set_index("kab_code")["kabupaten"].to_dict()

if "kabupaten" not in oos.columns:
    oos["kabupaten"] = oos["kab_code"].map(nama_kab)
oos["kabupaten"] = oos["kabupaten"].fillna(oos["kab_code"].map(nama_kab)).fillna(oos["kab_code"].astype(str))

geo_props = {}
for feat in geo.get("features", []):
    p = feat.get("properties", {})
    if "kab_code" in p:
        geo_props[int(p["kab_code"])] = p

if geo_props:
    dom_map = {k: v.get("DOMINAN") for k, v in geo_props.items() if "DOMINAN" in v}
    if dom_map:
        fc["DOMINAN"] = fc["kab_code"].map(dom_map)
        if D["hind_comp_kab"] is not None:
            D["hind_comp_kab"]["DOMINAN"] = D["hind_comp_kab"]["kab_code"].map(dom_map)

if eks is not None and {"kab_code", "SAWAH_KM2"}.issubset(eks.columns):
    sawah_map = eks.dropna(subset=["kab_code"]).set_index("kab_code")["SAWAH_KM2"].to_dict()
    fc["SAWAH_KM2"] = fc["kab_code"].map(sawah_map)
    if D["hind_comp_kab"] is not None:
        D["hind_comp_kab"]["SAWAH_KM2"] = D["hind_comp_kab"]["kab_code"].map(sawah_map)


# ============================================================
# BAGIAN 1: HALAMAN BERANDA / SELAMAT DATANG (LANDING PAGE)
# ============================================================

if st.session_state.halaman == "beranda":
    periode_info = meta.get("periode_data", ["-", "-"])
    fc_key_aktif = meta.get("skenario_prakiraan", "B_CWB3")
    versi_aktif = str(meta.get("versi", "v8")).upper()
    basis_hindcast = meta.get("basis_hindcast", "2025-12")
    target_hindcast = ", ".join(meta.get("target_hindcast", ["2026-01", "2026-02", "2026-03"]))

    fc_h1 = fc[fc["horizon"] == 1] if "horizon" in fc.columns else fc

    if IS_DUAL_TARGET:
        kol_cek = "peringatan_asi"
        label_cek = "Peringatan ASI Waspada/Tinggi (H+1)"
        sw_tinggi_col, sw_waspada_col = "SW_ATI_H1", "SW_AWA_H1"
    else:
        kol_cek = "indikasi_vegetasi_ndvi"
        label_cek = "Indikasi Penurunan NDVI Waspada/Tinggi (H+1)"
        sw_tinggi_col, sw_waspada_col = "SW_TIN_H1", "SW_WAS_H1"

    jml_tinggi = int((fc_h1[kol_cek] == "Tinggi").sum()) if kol_cek in fc_h1.columns else 0
    jml_waspada = int((fc_h1[kol_cek] == "Waspada").sum()) if kol_cek in fc_h1.columns else 0

    sawah_berisiko_str = f"{jml_tinggi + jml_waspada} Kab/Kota"
    sub_berisiko = label_cek
    if eks is not None and {sw_tinggi_col, sw_waspada_col}.issubset(eks.columns):
        tot_berisiko = float(eks[sw_tinggi_col].sum() + eks[sw_waspada_col].sum())
        tot_sawah = float(eks["SAWAH_KM2"].sum()) if "SAWAH_KM2" in eks.columns else 0
        sawah_berisiko_str = f"{tot_berisiko:,.0f} km²"
        sub_berisiko = f"Sawah Wilayah {label_cek} (dari {tot_sawah:,.0f} km²)"

    st.markdown(
        f"""
        <div class="navbar-landing">
            <div class="brand-logo">🌾 ASI-EWS <span>JABAR</span></div>
            <div class="brand-sub">Versi: <strong>{versi_aktif} (Dual-Target)</strong> • Model Utama: <strong>{fc_key_aktif}</strong> • Basis Hindcast: <strong>{basis_hindcast}</strong></div>
        </div>
        <div class="landing-hero">
            <div class="landing-badge">🌱 Karya Tulis Ilmiah • {versi_aktif} • Dual-Target Machine Learning (ASI & NDVI)</div>
            <h1><span class="accent">ASI-EWS:</span> Adaptive Agroclimatic Stress Intelligence – Sistem Peringatan Dini Cekaman Atmosfer dan Lahan Pertanian Berbasis Machine Learning dan Data Satelit di Jawa Barat</h1>
            <p>
                Sistem peringatan dini berbasis <strong>Random Forest</strong> dengan desain keluaran ganda:
                <strong>(1) Target Utama:</strong> memprediksi <em>Atmospheric Stress Index (ASI)</em> untuk deteksi dini potensi
                peningkatan cekaman agroklimat, dan <strong>(2) Target Pendukung:</strong> memprediksi <em>anomali NDVI</em>
                sebagai indikator respons vegetasi lahan sawah hingga 3 bulan ke depan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(FOLDER_TERSEDIA) > 1:
        col_src1, _ = st.columns([2.2, 2.8])
        with col_src1:
            pilihan_fld = st.selectbox(
                "Pilih Versi Output Model:",
                list(FOLDER_TERSEDIA.keys()),
                index=list(FOLDER_TERSEDIA.keys()).index(st.session_state.folder_aktif),
                format_func=lambda k: FOLDER_TERSEDIA[k]["label"],
                key="select_fld_beranda",
            )
            if pilihan_fld != st.session_state.folder_aktif:
                st.session_state.folder_aktif = pilihan_fld
                st.rerun()

    col_cta1, col_cta2, _ = st.columns([1.6, 1.5, 1.9])
    with col_cta1:
        st.button(
            "🚀 Buka Dashboard & Panduan Lapangan →",
            type="primary",
            use_container_width=True,
            on_click=pindah_ke,
            args=("dashboard",),
        )
    with col_cta2:
        st.download_button(
            "📥 Unduh Data Hindcast (CSV)",
            fc.to_csv(index=False).encode("utf-8-sig"),
            D["fc_filename"],
            "text/csv",
            use_container_width=True,
        )

    st.markdown(
        f"""
        <div style="margin-top: 18px;"></div>
        <div class="impact-strip">
            <div class="impact-item">
                <h3>{len(nama_kab)} Wilayah</h3>
                <p>Sentra Padi Jawa Barat (LBS ≥ {meta.get('min_sawah_km2', 10)} km²)</p>
            </div>
            <div class="impact-item">
                <h3>{sawah_berisiko_str}</h3>
                <p>{sub_berisiko}</p>
            </div>
            <div class="impact-item">
                <h3>Dual-Target ML</h3>
                <p>Target Utama: ASI | Pendukung: NDVI ({fc_key_aktif})</p>
            </div>
            <div class="impact-item">
                <h3>{basis_hindcast}</h3>
                <p>Simulasi Hindcast (Target: {target_hindcast})</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Pilar Metodologi & Desain Dual-Target ASI-EWS")
    st.markdown(
        f"""
        <div class="pillar-grid">
            <div class="pillar-card">
                <div class="pillar-icon">🌡️</div>
                <div class="pillar-title">Target 1: Cekaman ASI</div>
                <div class="pillar-text">
                    Bobot dasar transparan 25% (VPD, Tmax, defisit CWB3, defisit kelembapan tanah) dengan pembelajaran nonlinear oleh Random Forest.
                </div>
            </div>
            <div class="pillar-card">
                <div class="pillar-icon">🌿</div>
                <div class="pillar-title">Target 2: Respons NDVI</div>
                <div class="pillar-text">
                    Memprediksi anomali kehijauan vegetasi sawah (MODIS MOD13Q1) hingga H+3 sebagai validasi biofisik cekaman agroklimat.
                </div>
            </div>
            <div class="pillar-card">
                <div class="pillar-icon">🚜</div>
                <div class="pillar-title">Ramah Petani & Dinas</div>
                <div class="pillar-text">
                    Terjemahan otomatis status peringatan menjadi langkah irigasi, pompa air, dan template pesan WhatsApp PPL.
                </div>
            </div>
            <div class="pillar-card">
                <div class="pillar-icon">🧪</div>
                <div class="pillar-title">Validasi Bebas Kebocoran</div>
                <div class="pillar-text">
                    Standardisasi klimatologi 2001–2018, Expanding-Window CV, Leave-ZOM-Out, dan Year-Cluster Bootstrap CI 95%.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "ℹ️ **Klarifikasi Interpretasi:** Hasil berbasis Desember 2025 adalah **simulasi prakiraan retrospektif (hindcast)**. "
        "Peringatan ASI menunjukkan potensi peningkatan cekaman agroklimat (bukan kerusakan langsung tanaman), sedangkan "
        "indikasi NDVI menunjukkan potensi penurunan respons vegetasi (bukan prediksi gagal panen)."
    )

    st.stop()


# ============================================================
# BAGIAN 2: RUANG KERJA DASHBOARD ANALISIS (WORKSPACE)
# ============================================================

fc_key_aktif = meta.get("skenario_prakiraan", "B_CWB3")
basis_hindcast = meta.get("basis_hindcast", "2025-12")
target_hindcast = ", ".join(meta.get("target_hindcast", ["2026-01", "2026-02", "2026-03"]))

if "pilihan_peta_aktif" not in st.session_state:
    st.session_state.pilihan_peta_aktif = list(MAP_STYLES.keys())[0]

with st.sidebar:
    st.markdown("## 🌾 ASI-EWS JABAR")
    st.caption(
        f"Model Utama: **{fc_key_aktif}** | Basis: **{basis_hindcast}**"
    )

    st.button(
        "🏠 Kembali ke Beranda",
        use_container_width=True,
        on_click=pindah_ke,
        args=("beranda",),
    )

    if len(FOLDER_TERSEDIA) > 1:
        st.divider()
        st.markdown("### 📂 Versi Output Model")
        pilihan_fld_sb = st.selectbox(
            "Pilih Folder Hasil",
            list(FOLDER_TERSEDIA.keys()),
            index=list(FOLDER_TERSEDIA.keys()).index(st.session_state.folder_aktif),
            format_func=lambda k: FOLDER_TERSEDIA[k]["label"],
            key="select_fld_sidebar",
        )
        if pilihan_fld_sb != st.session_state.folder_aktif:
            st.session_state.folder_aktif = pilihan_fld_sb
            st.rerun()

    st.divider()
    st.markdown("### 📍 Cakupan Spasial")
    st.write(
        f"**{len(nama_kab)} Kabupaten/Kota** sentra sawah "
        f"(luas sawah LBS ≥ {meta.get('min_sawah_km2', 10)} km²)."
    )

    st.divider()
    if st.button("🔄 Muat Ulang Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Folder aktif: {st.session_state.folder_aktif}")


def peta(df, kolom_warna, judul, tinggi=510, **kwargs):
    if df.empty:
        return None

    style_aktif = MAP_STYLES[st.session_state.pilihan_peta_aktif]
    argumen = dict(
        data_frame=df,
        geojson=geo,
        locations="kab_code",
        featureidkey="properties.kab_code",
        color=kolom_warna,
        hover_name="kabupaten",
        opacity=0.86,
        center=PUSAT_PETA,
        zoom=6.8,
        **kwargs,
    )

    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(**argumen, map_style=style_aktif)
    else:
        fig = px.choropleth_mapbox(**argumen, mapbox_style=style_aktif)

    fig.update_layout(
        height=tinggi,
        margin=dict(l=0, r=0, t=44, b=0),
        title=dict(text=judul, x=0.015, xanchor="left", font=dict(size=14, color="#0F2E26")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=11.5, color="#132A24"),
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.90)",
            title_text="",
        ),
    )
    return fig


def tampilkan_peta(fig, key=None, nama_unduh="peta_asi_ews_jabar"):
    if fig is None:
        st.info("Tidak ada data untuk ditampilkan pada peta.")
        return
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": nama_unduh,
                "scale": 3,
            },
        },
        key=key,
    )


col_hdr1, col_hdr2 = st.columns([4.5, 0.9])
with col_hdr1:
    st.markdown(
        f"""
        <div class="workspace-topbar">
            <h2>🌾 {JUDUL_KTI}</h2>
            <span>Desain Dual-Target (ASI & NDVI) • Model Utama: <strong>{fc_key_aktif}</strong> • Basis Hindcast: <strong>{basis_hindcast}</strong> (Target: {target_hindcast})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_hdr2:
    st.button(
        "🏠 Beranda",
        use_container_width=True,
        on_click=pindah_ke,
        args=("beranda",),
        key="btn_home_top",
    )


tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "🚜 Panduan Petani & Dinas",
        "🗺️ Peta Hindcast",
        "📍 Detail Wilayah",
        "📊 Evaluasi & Bootstrap",
        "🌾 Validasi BPS",
        "⬇️ Unduh & Metode",
        "📉 Analisis Galat",
        "🖼️ Galeri Gambar KTI",
    ]
)


# ============================================================
# TAB 0 - PANDUAN OPERASIONAL PETANI, PPL & DINAS PERTANIAN
# ============================================================

with tab0:
    st.subheader("Sistem Peringatan Dini & Rekomendasi Lapangan (Petani, PPL & Dinas Pertanian)")
    st.caption(
        "Pilih Kabupaten/Kota dan bulan ke depan untuk melihat status peringatan dalam bahasa praktis, "
        "langkah antisipasi budidaya, serta teks laporan siap kirim ke grup WhatsApp kelompok tani."
    )

    c_tani1, c_tani2 = st.columns([2.2, 1.3])
    kode_tani = c_tani1.selectbox(
        "📍 Pilih Kabupaten / Kota Anda:",
        sorted(nama_kab, key=nama_kab.get),
        format_func=lambda k: nama_kab[k],
        key="kab_tani_sel",
    )
    h_tani = c_tani2.selectbox(
        "📅 Pilih Periode Peringatan:",
        [1, 2, 3],
        format_func=lambda x: f"H+{x} Bulan ({(pd.Timestamp(basis_hindcast + '-01') + pd.DateOffset(months=x)).strftime('%B %Y')})",
        key="h_tani_sel",
    )

    baris_tani = fc[(fc["kab_code"] == kode_tani) & (fc["horizon"] == h_tani)]
    if baris_tani.empty:
        st.warning("Data peringatan untuk wilayah dan periode ini belum tersedia.")
    else:
        row_t = baris_tani.iloc[0]
        nm_wil = nama_kab[kode_tani]
        bln_tgt_str = str(row_t.get("bulan_target", f"H+{h_tani}"))
        st_asi = str(row_t.get("peringatan_asi", "Normal")) if "peringatan_asi" in row_t else "Normal"
        st_ndvi = str(row_t.get("indikasi_vegetasi_ndvi", "Normal")) if "indikasi_vegetasi_ndvi" in row_t else "Normal"
        pemicu_str = row_t.get("DOMINAN", "Kombinasi suhu, penguapan & defisit hujan")
        luas_sw = float(row_t["SAWAH_KM2"]) if "SAWAH_KM2" in row_t and pd.notna(row_t["SAWAH_KM2"]) else None

        panduan = rekom_aksi_lapangan(st_asi, st_ndvi, pemicu_str)

        st.markdown(
            f"""
            <div class="alert-card-tani" style="background-color: {panduan['bg']}; border-left-color: {panduan['warna']};">
                <div style="font-size: 12px; font-weight: 800; color: {panduan['warna']}; letter-spacing: 0.5px;">
                    {panduan['level']} • TARGET PERIODE: {bln_tgt_str}
                </div>
                <h3 style="margin: 6px 0 8px 0; color: #0F2E26; font-size: 21px;">
                    {nm_wil} {f"• Luas Sawah Baku: {luas_sw:,.0f} km²" if luas_sw else ""}
                </h3>
                <p style="margin: 0; color: #1E3A32; font-size: 14.5px; line-height: 1.6;">
                    {panduan['ringkas_petani']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m_t1, m_t2, m_t3 = st.columns(3)
        m_t1.metric("🌡️ Status Cekaman Cuaca (ASI)", st_asi, f"Skor ASI: {row_t['pred_asi']:.2f}" if "pred_asi" in row_t else None)
        m_t2.metric("🌿 Status Kehijauan Padi (NDVI)", st_ndvi, f"Anomali: {row_t['pred_anom_ndvi']:.2f} SD" if "pred_anom_ndvi" in row_t else None)
        m_t3.metric("⚡ Faktor Pemicu Dominan", str(panduan["pemicu"]))

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            st.markdown("#### 👨‍🌾 Rekomendasi Praktis untuk Petani & Poktan")
            for idx_a, item_a in enumerate(panduan["aksi_petani"], 1):
                st.markdown(f"**{idx_a}.** {item_a}")

        with col_act2:
            st.markdown("#### 🏛️ Rekomendasi Tindak Lanjut Dinas Pertanian & PPL")
            for idx_d, item_d in enumerate(panduan["aksi_dinas"], 1):
                st.markdown(f"**{idx_d}.** {item_d}")

        st.divider()
        st.markdown("#### 📲 Pesan Peringatan Siap Kirim WhatsApp (Grup Poktan / BPP)")
        teks_wa = (
            f"*INFO PERINGATAN DINI PERTANIAN (ASI-EWS JABAR)*\n"
            f"Wilayah: *{nm_wil}*\n"
            f"Periode Target: *{bln_tgt_str} (H+{h_tani})*\n"
            f"Status: *{panduan['level']}*\n"
            f"• Cekaman Cuaca/Atmosfer (ASI): *{st_asi}*\n"
            f"• Indikasi Vegetasi Sawah (NDVI): *{st_ndvi}*\n"
            f"• Faktor Utama: {panduan['pemicu']}\n\n"
            f"*Saran Tindakan Lapangan:*\n"
            + "\n".join([f"- {x}" for x in panduan["aksi_petani"][:3]])
            + f"\n\n_Sumber: Sistem ASI-EWS Jawa Barat (Basis {basis_hindcast})_"
        )
        st.text_area(
            "Salin teks di bawah ini untuk dibagikan ke WhatsApp Penyuluh / Kelompok Tani:",
            value=teks_wa,
            height=220,
            key="wa_text_box",
        )

    st.divider()
    st.markdown("#### 🚨 Daftar Kabupaten Prioritas Intervensi Dinas Pertanian Jawa Barat")
    fc_prio = fc[fc["horizon"] == h_tani].copy()
    kol_prio = "peringatan_asi" if "peringatan_asi" in fc_prio.columns else "indikasi_vegetasi_ndvi"
    fc_prio["Skor_Prioritas"] = fc_prio[kol_prio].map({"Tinggi": 3, "Waspada": 2, "Normal": 1}).fillna(0)
    fc_prio = fc_prio.sort_values(
        by=["Skor_Prioritas", "SAWAH_KM2"] if "SAWAH_KM2" in fc_prio.columns else ["Skor_Prioritas"],
        ascending=[False, False] if "SAWAH_KM2" in fc_prio.columns else [False],
    )
    kol_tampil_prio = [
        c for c in [
            "kabupaten",
            "SAWAH_KM2",
            "peringatan_asi",
            "indikasi_vegetasi_ndvi",
            "DOMINAN",
        ]
        if c in fc_prio.columns
    ]
    st.dataframe(
        fc_prio[kol_tampil_prio].rename(
            columns={
                "kabupaten": "Kabupaten/Kota",
                "SAWAH_KM2": "Luas Sawah (km²)",
                "peringatan_asi": "Status Cuaca (ASI)",
                "indikasi_vegetasi_ndvi": "Status Tanaman (NDVI)",
                "DOMINAN": "Pemicu Dominan",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# TAB 1 - PETA HINDCAST DUAL-TARGET (ASI & NDVI)
# ============================================================

with tab1:
    st.subheader("Peta Simulasi Hindcast Retrospektif (Keluaran Ganda: ASI & NDVI)")
    st.caption(
        f"📌 Simulasi retrospektif berbasis **{basis_hindcast}** untuk target **{target_hindcast}** (bukan prakiraan real-time). "
        "Gunakan tombol kamera di pojok kanan atas peta untuk mengunduh gambar PNG resolusi tinggi untuk KTI."
    )

    col_map, col_control = st.columns([3.2, 1.3])

    with col_control:
        st.markdown("##### ⚙️ Kontrol Peta")

        # Pemilih jenis basemap langsung muncul di sini (tidak tersembunyi di sidebar)
        gaya_dipilih = st.selectbox(
            "🗺️ Jenis Tampilan Peta (Basemap)",
            list(MAP_STYLES.keys()),
            index=list(MAP_STYLES.keys()).index(st.session_state.pilihan_peta_aktif),
            key="pilih_basemap_tab1",
        )
        if gaya_dipilih != st.session_state.pilihan_peta_aktif:
            st.session_state.pilihan_peta_aktif = gaya_dipilih

        mode = st.radio(
            "Mode pemetaan",
            ["Simulasi hindcast (peringatan)", "Kondisi historis (bulanan)"],
            key="mode_peta",
        )

        if mode == "Simulasi hindcast (peringatan)":
            fc_sumber = fc
            if D["hind_comp_kab"] is not None and "skenario_prakiraan" in D["hind_comp_kab"].columns:
                daftar_sk_hc = D["hind_comp_kab"]["skenario_prakiraan"].unique().tolist()
                idx_def = daftar_sk_hc.index(fc_key_aktif) if fc_key_aktif in daftar_sk_hc else 0
                sk_peta = st.selectbox(
                    "Model hindcast",
                    daftar_sk_hc,
                    index=idx_def,
                    key="sk_peta_hc",
                )
                fc_sumber = D["hind_comp_kab"][D["hind_comp_kab"]["skenario_prakiraan"] == sk_peta].copy()

            if IS_DUAL_TARGET and "peringatan_asi" in fc_sumber.columns:
                target_peta = st.radio(
                    "Keluaran target",
                    ["asi", "ndvi"],
                    format_func=lambda k: "Target 1: Cekaman ASI" if k == "asi" else "Target 2: Respons NDVI",
                    key="target_peta_hc",
                )
            else:
                target_peta = "ndvi"

            h = st.selectbox(
                "Horizon simulasi",
                [1, 2, 3],
                format_func=lambda x: f"H+{x} ({(pd.Timestamp(basis_hindcast + '-01') + pd.DateOffset(months=x)).strftime('%Y-%m')})",
            )
            jenis_ambang = st.radio(
                "Klasifikasi ambang",
                ["Absolut (Utama)", "Relatif Persentil (Latih ≤2018)"],
                key="jenis_ambang_peta",
                help="Absolut: ASI >= 0.35/0.75 atau NDVI <= -0.5/-1.0 SD. Relatif: dikalibrasi hanya dari OOS periode latih <= 2018.",
            )
        else:
            bulan = sorted(
                fit.dropna(subset=["asi"])["tanggal"].dt.strftime("%Y-%m").unique()
            )
            if not bulan:
                st.warning("Tidak ditemukan data historis ASI.")
                st.stop()
            pilih = st.select_slider("Bulan historis", options=bulan, value=bulan[-1])
            var = st.selectbox(
                "Variabel indikator",
                [
                    "ASI (BMKG)",
                    "ASI (CHIRPS)",
                    "Anomali NDVI",
                    "Anomali VPD",
                    "Anomali Tmax",
                    "Anomali kelembapan tanah",
                ],
            )

    with col_map:
        if mode == "Simulasi hindcast (peringatan)":
            d = fc_sumber[fc_sumber["horizon"] == h].copy()
            if d.empty:
                st.warning(f"Data simulasi horizon H+{h} belum tersedia.")
            else:
                if target_peta == "asi" and "peringatan_asi" in d.columns:
                    kol_kat = (
                        "peringatan_asi_relatif"
                        if (jenis_ambang.startswith("Relatif") and "peringatan_asi_relatif" in d.columns)
                        else "peringatan_asi"
                    )
                    judul_kat = "Peringatan Cekaman ASI"
                    hover_dict = {
                        "kab_code": False,
                        "pred_asi": ":.2f",
                    }
                    if "sd_antar_pohon_asi" in d.columns:
                        hover_dict["sd_antar_pohon_asi"] = ":.2f"
                    if "aktual_asi" in d.columns and d["aktual_asi"].notna().any():
                        hover_dict["aktual_asi"] = ":.2f"
                else:
                    kol_kat = (
                        "indikasi_ndvi_relatif"
                        if (jenis_ambang.startswith("Relatif") and "indikasi_ndvi_relatif" in d.columns)
                        else "indikasi_vegetasi_ndvi"
                    )
                    judul_kat = "Indikasi Respons Vegetasi NDVI"
                    hover_dict = {
                        "kab_code": False,
                        "pred_anom_ndvi": ":.2f" if "pred_anom_ndvi" in d.columns else False,
                    }
                    if "sd_antar_pohon_ndvi" in d.columns:
                        hover_dict["sd_antar_pohon_ndvi"] = ":.2f"
                    if "aktual_anom_ndvi" in d.columns and d["aktual_anom_ndvi"].notna().any():
                        hover_dict["aktual_anom_ndvi"] = ":.2f"

                if "SAWAH_KM2" in d.columns:
                    hover_dict["SAWAH_KM2"] = ":.1f"
                if "DOMINAN" in d.columns:
                    hover_dict["DOMINAN"] = True

                target_bln = d["bulan_target"].iloc[0] if "bulan_target" in d else ""
                basis_bln = str(d["tanggal_basis"].iloc[0])[:7] if "tanggal_basis" in d else ""
                model_lbl = d["skenario_prakiraan"].iloc[0] if "skenario_prakiraan" in d else fc_key_aktif

                fig = peta(
                    d,
                    kol_kat,
                    f"{judul_kat} ({model_lbl}) | Target: {target_bln} | Basis: {basis_bln}",
                    color_discrete_map=WARNA_RISIKO,
                    category_orders={kol_kat: ["Normal", "Waspada", "Tinggi"]},
                    hover_data=hover_dict,
                )
                tampilkan_peta(
                    fig,
                    key="map_forecast",
                    nama_unduh=f"peta_hindcast_{target_peta}_{model_lbl}_H{h}",
                )
        else:
            kolom_var = {
                "ASI (BMKG)": "asi",
                "ASI (CHIRPS)": "asi_c",
                "Anomali NDVI": "ndvi_z",
                "Anomali VPD": "vpd_z",
                "Anomali Tmax": "tmax_z",
                "Anomali kelembapan tanah": "soil1_z",
            }[var]

            if kolom_var not in fit.columns:
                st.warning(f"Kolom {kolom_var} tidak tersedia pada fitur_bulanan.csv.")
                d = pd.DataFrame()
            else:
                d = fit[fit["tanggal"].dt.strftime("%Y-%m") == pilih].copy()

            if not d.empty:
                skala = "RdBu" if kolom_var in ("ndvi_z", "soil1_z") else "RdBu_r"
                fig = peta(
                    d,
                    kolom_var,
                    f"{var} | Periode {pilih}",
                    color_continuous_scale=skala,
                    range_color=(-2.5, 2.5),
                    hover_data={"kab_code": False},
                )
                tampilkan_peta(
                    fig,
                    key="map_historical",
                    nama_unduh=f"peta_historis_{kolom_var}_{pilih}",
                )

    if mode == "Simulasi hindcast (peringatan)" and not d.empty:
        kol_ringkas = (
            "peringatan_asi"
            if (target_peta == "asi" and "peringatan_asi" in d.columns)
            else "indikasi_vegetasi_ndvi"
        )
        judul_ringkas = (
            "Ringkasan Peringatan Cekaman Agroklimat (ASI)"
            if target_peta == "asi"
            else "Ringkasan Indikasi Respons Vegetasi (NDVI)"
        )
        st.markdown(f"#### {judul_ringkas}")

        has_sawah = "SAWAH_KM2" in d.columns and d["SAWAH_KM2"].notna().any()
        if has_sawah:
            sw_tinggi = float(d.loc[d[kol_ringkas] == "Tinggi", "SAWAH_KM2"].sum())
            sw_waspada = float(d.loc[d[kol_ringkas] == "Waspada", "SAWAH_KM2"].sum())
            sw_normal = float(d.loc[d[kol_ringkas] == "Normal", "SAWAH_KM2"].sum())
            sw_total = float(d["SAWAH_KM2"].sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "🔴 Kategori Tinggi",
                f"{int((d[kol_ringkas] == 'Tinggi').sum())} Wil ({sw_tinggi:,.0f} km²)",
            )
            k2.metric(
                "🟡 Kategori Waspada",
                f"{int((d[kol_ringkas] == 'Waspada').sum())} Wil ({sw_waspada:,.0f} km²)",
            )
            k3.metric(
                "🟢 Kategori Normal",
                f"{int((d[kol_ringkas] == 'Normal').sum())} Wil ({sw_normal:,.0f} km²)",
            )
            k4.metric(
                "🌾 Total Sawah Wilayah",
                f"{sw_total:,.0f} km² ({len(d)} Wil)",
            )
            st.caption(
                "Catatan: Luas sawah di atas merupakan total luas sawah LBS di kabupaten yang masuk kategori peringatan, "
                "bukan luas sawah yang terbukti mengalami kerusakan."
            )
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🔴 Kategori Tinggi", f"{int((d[kol_ringkas] == 'Tinggi').sum())} Wilayah")
            k2.metric("🟡 Kategori Waspada", f"{int((d[kol_ringkas] == 'Waspada').sum())} Wilayah")
            k3.metric("🟢 Kategori Normal", f"{int((d[kol_ringkas] == 'Normal').sum())} Wilayah")
            k4.metric("📍 Total Cakupan", f"{len(d)} Kab/Kota")

        kandidat_kolom = [
            "kabupaten",
            "SAWAH_KM2",
            "DOMINAN",
            "pred_asi",
            "aktual_asi",
            "peringatan_asi",
            "pred_anom_ndvi",
            "aktual_anom_ndvi",
            "indikasi_vegetasi_ndvi",
        ]
        kolom_tabel = [c for c in kandidat_kolom if c in d.columns]
        tbl = d[kolom_tabel].copy().rename(
            columns={
                "kabupaten": "Kabupaten/Kota",
                "SAWAH_KM2": "Luas Sawah Wilayah (km²)",
                "DOMINAN": "Pemicu Stres Basis",
                "pred_asi": "Prediksi ASI",
                "aktual_asi": "Aktual ASI",
                "peringatan_asi": "Peringatan ASI",
                "pred_anom_ndvi": "Prediksi NDVI (SD)",
                "aktual_anom_ndvi": "Aktual NDVI (SD)",
                "indikasi_vegetasi_ndvi": "Indikasi NDVI",
            }
        )
        if "Prediksi ASI" in tbl.columns:
            tbl = tbl.sort_values("Prediksi ASI", ascending=False)
        elif "Prediksi NDVI (SD)" in tbl.columns:
            tbl = tbl.sort_values("Prediksi NDVI (SD)")
        st.dataframe(tbl, hide_index=True, use_container_width=True)

        if D["hind_comp_met"] is not None and isinstance(D["hind_comp_met"], pd.DataFrame):
            with st.expander("🎯 Perbandingan Akurasi Hindcast Jan–Mar 2026 (A_dasar vs B_CWB3 vs D_ENSO_CH)", expanded=True):
                st.dataframe(D["hind_comp_met"].round(3), hide_index=True, use_container_width=True)
        elif D["hind_met"] is not None and isinstance(D["hind_met"], pd.DataFrame):
            with st.expander("🎯 Evaluasi Akurasi Hindcast Jan–Mar 2026 vs Aktual", expanded=True):
                st.dataframe(D["hind_met"].round(3), hide_index=True, use_container_width=True)


# ============================================================
# TAB 2 - DETAIL KABUPATEN & SINYAL ENSO/IOD
# ============================================================

with tab2:
    st.subheader("Analisis Detail Kabupaten/Kota & Telekoneksi Global")

    kode = st.selectbox(
        "Pilih kabupaten/kota",
        sorted(nama_kab, key=nama_kab.get),
        format_func=lambda k: nama_kab[k],
    )
    dk = fit[fit["kab_code"] == kode].sort_values("tanggal")

    if dk.empty:
        st.warning("Data kabupaten/kota ini tidak tersedia.")
    else:
        tahun_min = int(dk["tanggal"].dt.year.min())
        tahun_max = int(dk["tanggal"].dt.year.max())
        awal = min(max(2018, tahun_min), tahun_max)
        rentang = st.slider(
            "Rentang tahun pengamatan",
            min_value=tahun_min,
            max_value=tahun_max,
            value=(awal, tahun_max),
        )
        dk = dk[
            (dk["tanggal"].dt.year >= rentang[0])
            & (dk["tanggal"].dt.year <= rentang[1])
        ].copy()

        a, b = st.columns(2)
        with a:
            st.markdown("#### ASI Bulanan dan Komponen Penyusun (Bobot 25%)")
            gunakan_ch = fc_key_aktif.endswith("_CH") or fc_key_aktif == "G_gabung"
            cwb_col = "c_cwb_c" if (gunakan_ch and "c_cwb_c" in dk.columns) else "c_cwb"
            asi_col = "asi_c" if (gunakan_ch and "asi_c" in dk.columns) else "asi"

            komponen = [c for c in ["c_vpd", "c_tmax", cwb_col, "c_soil"] if c in dk.columns]
            if komponen:
                kom = dk.melt(
                    id_vars="tanggal",
                    value_vars=komponen,
                    var_name="komponen",
                    value_name="kontribusi",
                )
                kom["komponen"] = kom["komponen"].map({
                    "c_vpd": "VPD",
                    "c_tmax": "Suhu Maks",
                    "c_cwb": "Defisit CWB3 (BMKG)",
                    "c_cwb_c": "Defisit CWB3 (CHIRPS)",
                    "c_soil": "Kering Tanah",
                })
                f1 = px.bar(
                    kom,
                    x="tanggal",
                    y="kontribusi",
                    color="komponen",
                    barmode="relative",
                )
                if asi_col in dk.columns:
                    f1.add_scatter(
                        x=dk["tanggal"],
                        y=dk[asi_col],
                        mode="lines",
                        name="ASI Total",
                        line=dict(color="#0F2E26", width=2),
                    )
                f1.add_hline(y=0.75, line_dash="dash", line_color="#D9535D", annotation_text="Ambang Tinggi (0.75)")
                f1.update_layout(
                    height=380,
                    margin=dict(t=15, b=10, l=10, r=10),
                    legend=dict(orientation="h", y=1.12, x=0),
                )
                st.plotly_chart(f1, use_container_width=True, config=konfig_unduh(f"asi_komponen_{nama_kab[kode]}"))
            else:
                st.info("Kolom komponen ASI tidak tersedia.")

        with b:
            st.markdown("#### Anomali NDVI Sawah (Target Pendukung)")
            if "ndvi_z" in dk.columns:
                f2 = px.line(dk, x="tanggal", y="ndvi_z")
                f2.update_traces(line_color="#239B75")
                f2.add_hline(
                    y=-1,
                    line_dash="dash",
                    line_color="#D9535D",
                    annotation_text="Ambang Tinggi (-1.0 SD)",
                )
                f2.add_hline(y=0, line_dash="dot", line_color="gray")
                f2.update_layout(
                    height=380,
                    margin=dict(t=15, b=10, l=10, r=10),
                    yaxis_title="Anomali NDVI (SD)",
                )
                st.plotly_chart(f2, use_container_width=True, config=konfig_unduh(f"ndvi_historis_{nama_kab[kode]}"))
            else:
                st.info("Kolom ndvi_z tidak tersedia.")

        if {"nino34", "dmi"}.issubset(dk.columns) and dk[["nino34", "dmi"]].notna().any().any():
            st.markdown("#### Indeks Iklim Global: ENSO (Niño 3.4) & Indian Ocean Dipole (DMI)")
            enso_melt = dk.melt(
                id_vars="tanggal",
                value_vars=["nino34", "dmi"],
                var_name="Indeks",
                value_name="Anomali (°C)",
            )
            enso_melt["Indeks"] = enso_melt["Indeks"].map({
                "nino34": "Niño 3.4 (ENSO)",
                "dmi": "DMI (IOD)",
            })
            f_enso = px.line(
                enso_melt,
                x="tanggal",
                y="Anomali (°C)",
                color="Indeks",
                color_discrete_map={"Niño 3.4 (ENSO)": "#D9535D", "DMI (IOD)": "#E9A23B"},
            )
            f_enso.add_hline(y=0.5, line_dash="dot", line_color="#D9535D", annotation_text="El Niño / IOD+ (+0.5)")
            f_enso.add_hline(y=-0.5, line_dash="dot", line_color="#239B75", annotation_text="La Niña / IOD- (-0.5)")
            f_enso.update_layout(
                height=310,
                margin=dict(t=15, b=10, l=10, r=10),
                legend=dict(orientation="h", y=1.15, x=0),
            )
            st.plotly_chart(f_enso, use_container_width=True, config=konfig_unduh("indeks_enso_iod"))

        st.markdown("#### Prediksi vs Aktual (Expanding-Window Temporal CV)")
        cc1, cc2, cc3 = st.columns([1.2, 1, 2])
        target_opsi = sorted(oos["target"].dropna().unique().tolist())
        tgt_det = cc1.selectbox(
            "Target model",
            target_opsi,
            format_func=lambda t: "Target 1: ASI" if t == "asi" else "Target 2: NDVI",
            key="tgt_detail",
        )
        horizon_opsi = sorted(oos["horizon"].dropna().unique().tolist())
        if horizon_opsi:
            hh = cc2.selectbox("Horizon prediksi", horizon_opsi, key="h_detail")
            skenario_opsi = sorted(oos["skenario"].dropna().unique().tolist())
            default_skenario = [
                s for s in ["A_dasar", "B_CWB3", "D_ENSO_CH", "G_gabung"]
                if s in skenario_opsi
            ][:3]
            if not default_skenario:
                default_skenario = skenario_opsi[:2]
            skn = cc3.multiselect(
                "Skenario model",
                skenario_opsi,
                default=list(dict.fromkeys(default_skenario)),
                key="sk_detail",
            )
            o = oos[
                (oos["kab_code"] == kode)
                & (oos["target"] == tgt_det)
                & (oos["horizon"] == hh)
                & (oos["skenario"].isin(skn))
            ].copy()
            o = o[
                (o["tgt_date"].dt.year >= rentang[0])
                & (o["tgt_date"].dt.year <= rentang[1])
            ]
            if o.empty:
                st.info("Tidak ada data prediksi untuk pilihan ini.")
            elif {"pred", "aktual"}.issubset(o.columns):
                fig_oos = px.line(
                    o.sort_values("tgt_date"),
                    x="tgt_date",
                    y="pred",
                    color="skenario",
                )
                aktual = o.drop_duplicates("tgt_date").sort_values("tgt_date")
                fig_oos.add_scatter(
                    x=aktual["tgt_date"],
                    y=aktual["aktual"],
                    mode="lines",
                    name="Aktual",
                    line=dict(color="#0F2E26", width=2.2),
                )
                fig_oos.update_layout(
                    height=380,
                    margin=dict(t=15, b=10, l=10, r=10),
                    xaxis_title="Bulan target",
                    yaxis_title="Indeks ASI" if tgt_det == "asi" else "Anomali NDVI (SD)",
                    legend=dict(orientation="h", y=1.15, x=0),
                )
                st.plotly_chart(
                    fig_oos,
                    use_container_width=True,
                    config=konfig_unduh(f"prediksi_vs_aktual_{nama_kab[kode]}_{tgt_det}_H{hh}"),
                )

        st.markdown("#### Hasil Simulasi Hindcast Wilayah Ini (Dual-Target)")
        kolom_fc = [
            c for c in [
                "bulan_target",
                "horizon",
                "pred_asi",
                "aktual_asi",
                "peringatan_asi",
                "pred_anom_ndvi",
                "aktual_anom_ndvi",
                "indikasi_vegetasi_ndvi",
            ]
            if c in fc.columns
        ]
        st.dataframe(
            fc[fc["kab_code"] == kode][kolom_fc],
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# TAB 3 - EVALUASI MODEL (DUAL-TARGET) & BOOTSTRAP
# ============================================================

with tab3:
    st.subheader("Evaluasi Komparatif Model pada Sampel Identik (BMKG, CHIRPS, ENSO/IOD & Gabungan)")
    st.caption(
        "Seluruh 11 skenario model dievaluasi pada himpunan kabupaten dan tanggal yang identik agar perbandingan adil (apple-to-apple)."
    )

    target_eval_opsi = sorted(res["target"].dropna().unique().tolist())
    tgt_tab3 = st.radio(
        "Pilih Target Evaluasi:",
        target_eval_opsi,
        horizontal=True,
        format_func=lambda t: LABEL_TARGET.get(t, t.upper()),
        key="tgt_tab3_radio",
    )

    res_t = res[res["target"] == tgt_tab3].copy()

    if "skema" not in res_t.columns:
        st.warning("Kolom skema tidak tersedia pada hasil_perbandingan_model.csv.")
    else:
        skema_opsi = sorted(res_t["skema"].dropna().unique().tolist())
        if skema_opsi:
            skema = st.radio("Skema validasi", skema_opsi, horizontal=True)
            metrik_opsi = [m for m in ["RMSE", "R2", "AUC_event", "r", "MAE"] if m in res_t.columns]
            if metrik_opsi:
                metrik = st.selectbox("Pilih metrik evaluasi", metrik_opsi)
                r = res_t[res_t["skema"] == skema]
                if r.empty:
                    st.info("Tidak ada hasil untuk skema validasi ini.")
                else:
                    fig_res = px.bar(
                        r,
                        x="horizon",
                        y=metrik,
                        color="skenario" if "skenario" in r.columns else None,
                        barmode="group",
                        text_auto=".3f",
                    )
                    fig_res.update_layout(
                        height=440,
                        margin=dict(t=20, b=10, l=10, r=10),
                        xaxis=dict(dtick=1, title="Horizon (bulan)"),
                        yaxis_title=metrik,
                        legend=dict(orientation="h", y=1.18, x=0),
                    )
                    st.plotly_chart(
                        fig_res,
                        use_container_width=True,
                        config=konfig_unduh(f"perbandingan_{metrik}_{tgt_tab3}_{skema}"),
                    )

        if {"skema", "horizon", "skenario", "RMSE"}.issubset(res_t.columns):
            st.markdown(f"#### Matriks Perbandingan RMSE Seluruh Skenario ({tgt_tab3.upper()})")
            pv = res_t.pivot_table(
                index=["skema", "horizon"], columns="skenario", values="RMSE"
            ).round(3)
            st.dataframe(pv, use_container_width=True)

    if D["musim"] is not None and isinstance(D["musim"], pd.DataFrame):
        st.divider()
        st.subheader(f"Performa Model Berdasarkan Musim ({tgt_tab3.upper()})")
        musim_df = D["musim"].copy()
        if "target" in musim_df.columns:
            musim_df = musim_df[musim_df["target"] == tgt_tab3]
        if not musim_df.empty:
            h_musim = st.selectbox(
                "Pilih Horizon untuk Analisis Musim",
                sorted(musim_df["horizon"].unique()),
                key="h_musim_sel",
            )
            sub_m = musim_df[musim_df["horizon"] == h_musim]
            fig_musim = px.bar(
                sub_m,
                x="skenario",
                y="RMSE",
                color="musim",
                barmode="group",
                text_auto=".3f",
                color_discrete_map={"hujan_Nov-Apr": "#239B75", "kemarau_Mei-Okt": "#E9A23B"},
                title=f"RMSE Expanding-Window CV Menurut Musim ({tgt_tab3.upper()}, H+{h_musim})",
            )
            fig_musim.update_layout(
                height=410,
                margin=dict(t=40, b=10, l=10, r=10),
                legend=dict(orientation="h", y=1.12, x=0),
            )
            st.plotly_chart(
                fig_musim,
                use_container_width=True,
                config=konfig_unduh(f"rmse_musim_{tgt_tab3}_H{h_musim}"),
            )

    if D["boot"] is not None and isinstance(D["boot"], pd.DataFrame):
        st.divider()
        st.subheader(f"Uji Signifikansi Year-Cluster Bootstrap Selisih RMSE ({tgt_tab3.upper()}, CI 95%)")
        st.caption(
            "Resampling berbasis kelompok tahun untuk seluruh horizon (H+1, H+2, H+3). "
            "Selisih = RMSE(Sederhana) - RMSE(Lengkap). Nilai positif berarti model kedua lebih akurat; "
            "**nyata = True** menandakan interval kepercayaan 95% tidak memuat nol."
        )
        boot_df = D["boot"].copy()
        if "target" in boot_df.columns:
            boot_df = boot_df[boot_df["target"] == tgt_tab3]
        if not boot_df.empty:
            c_b1, c_b2 = st.columns(2)
            filter_musim = c_b1.radio(
                "Kelompok musim",
                boot_df["musim"].unique().tolist(),
                horizontal=True,
                key="boot_musim_radio",
            )
            filter_h_boot = c_b2.multiselect(
                "Filter horizon",
                sorted(boot_df["horizon"].unique().tolist()),
                default=sorted(boot_df["horizon"].unique().tolist()),
                key="boot_h_multi",
            )
            sub_boot = boot_df[
                (boot_df["musim"] == filter_musim)
                & (boot_df["horizon"].isin(filter_h_boot))
            ]
            st.dataframe(sub_boot, hide_index=True, use_container_width=True)

    imp_aktif = D["imp_dict"].get(tgt_tab3)
    if imp_aktif is None or (isinstance(imp_aktif, pd.DataFrame) and imp_aktif.empty):
        imp_aktif = next(iter(D["imp_dict"].values()), None)

    if imp_aktif is not None and isinstance(imp_aktif, pd.DataFrame):

        st.divider()
        st.subheader(f"Fitur Terpenting Model Utama ({fc_key_aktif}, Target {tgt_tab3.upper()}, H+1)")
        st.caption("Nilai SHAP / importance menunjukkan asosiasi prediktif fitur dalam model, bukan bukti hubungan sebab-akibat.")
        imp = imp_aktif.copy()
        numerik = imp.select_dtypes(include=np.number).columns
        kolom_nilai = "nilai" if "nilai" in imp.columns else (numerik[0] if len(numerik) else None)
        if kolom_nilai:
            imp = imp.sort_values(kolom_nilai)
            fig_imp = px.bar(
                imp,
                x=kolom_nilai,
                y=imp.index.astype(str),
                orientation="h",
                color_discrete_sequence=["#239B75"],
            )
            fig_imp.update_layout(height=460, margin=dict(t=15, b=10, l=10, r=10), yaxis_title="")
            st.plotly_chart(
                fig_imp,
                use_container_width=True,
                config=konfig_unduh(f"feature_importance_{fc_key_aktif}_{tgt_tab3}_H1"),
            )

    if D["sens"] is not None:
        with st.expander("Sensitivitas Bobot ASI (Simulasi Dirichlet)"):
            st.json(D["sens"])


# ============================================================
# TAB 4 - VALIDASI BPS
# ============================================================

with tab4:
    st.subheader("ASI dan NDVI vs Produktivitas Padi BPS")
    if D["bps"] is None:
        st.info("File bps_padi_jabar.csv belum tersedia.")
    else:
        bps = D["bps"].copy()
        kolom_bps = {"kabupaten", "tahun", "produktivitas_ton_ha"}
        if not kolom_bps.issubset(bps.columns):
            st.warning("Kolom BPS harus memuat: kabupaten, tahun, produktivitas_ton_ha.")
        else:
            bps["kabupaten"] = bps["kabupaten"].astype(str).str.upper().str.strip()
            fitur_tahunan = (
                fit.assign(
                    kabupaten=fit["kabupaten"].astype(str).str.upper().str.strip(),
                    tahun=fit["tanggal"].dt.year,
                )
                .groupby(["kabupaten", "tahun"], as_index=False)
                .agg(asi=("asi", "mean"), ndvi_z=("ndvi_z", "mean"))
            )
            gabung = fitur_tahunan.merge(
                bps, on=["kabupaten", "tahun"], how="inner"
            ).dropna(subset=["asi", "ndvi_z", "produktivitas_ton_ha"])

            if gabung.empty:
                st.warning("Tidak ada pasangan data yang cocok antara fitur bulanan dan data BPS.")
            else:
                for kol in ["asi", "ndvi_z", "produktivitas_ton_ha"]:
                    gabung[kol + "_dm"] = (
                        gabung[kol] - gabung.groupby("kabupaten")[kol].transform("mean")
                    )
                pilihan_x = st.radio(
                    "Variabel sumbu X",
                    ["asi_dm", "ndvi_z_dm"],
                    horizontal=True,
                    format_func=lambda v: (
                        "ASI (selisih dari rerata kabupaten)"
                        if v == "asi_dm"
                        else "Anomali NDVI (selisih dari rerata kabupaten)"
                    ),
                )
                fig_bps = px.scatter(
                    gabung,
                    x=pilihan_x,
                    y="produktivitas_ton_ha_dm",
                    color="kabupaten",
                    hover_data=["tahun"],
                )
                fig_bps.update_layout(
                    height=440,
                    margin=dict(t=15, b=10, l=10, r=10),
                    yaxis_title="Produktivitas (selisih rerata, ton/ha)",
                    showlegend=False,
                )
                st.plotly_chart(
                    fig_bps,
                    use_container_width=True,
                    config=konfig_unduh(f"validasi_bps_{pilihan_x}"),
                )
                rho_asi = gabung["asi_dm"].corr(
                    gabung["produktivitas_ton_ha_dm"], method="spearman"
                )
                rho_ndvi = gabung["ndvi_z_dm"].corr(
                    gabung["produktivitas_ton_ha_dm"], method="spearman"
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Jumlah Pasangan Data", len(gabung))
                m2.metric("Spearman ASI vs Produktivitas", f"{rho_asi:.2f}" if pd.notna(rho_asi) else "N/A")
                m3.metric("Spearman NDVI vs Produktivitas", f"{rho_ndvi:.2f}" if pd.notna(rho_ndvi) else "N/A")
                st.caption("Korelasi bersifat deskriptif dan tidak membuktikan sebab-akibat.")


# ============================================================
# TAB 5 - UNDUH DAN METODOLOGI
# ============================================================

with tab5:
    st.subheader("Unduh Hasil Analisis & Data Spasial")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(
        "⬇️ Data Hindcast (CSV)",
        fc.to_csv(index=False).encode("utf-8-sig"),
        D["fc_filename"],
        "text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "⬇️ Fitur Bulanan (CSV)",
        fit.to_csv(index=False).encode("utf-8-sig"),
        "fitur_bulanan.csv",
        "text/csv",
        use_container_width=True,
    )
    if eks is not None:
        c3.download_button(
            "⬇️ Eksposur Sawah (CSV)",
            eks.to_csv(index=False).encode("utf-8-sig"),
            "eksposur_sawah.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        c3.info("eksposur_sawah.csv belum ada.")

    shp_files = [
        p for p in RES_AKTIF.glob("ASI_EWS_KAB.*")
        if p.suffix.lower() in {".shp", ".shx", ".dbf", ".prj", ".cpg"}
    ]
    if shp_files:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as arsip:
            for path in shp_files:
                arsip.write(path, path.name)
        c4.download_button(
            "⬇️ Shapefile Wilayah (ZIP)",
            buffer.getvalue(),
            "ASI_EWS_KAB_shp.zip",
            "application/zip",
            use_container_width=True,
        )
    else:
        c4.info("Shapefile belum tersedia di folder output.")

    st.divider()
    st.subheader("Metodologi Penelitian & Klarifikasi Interpretasi (Pipeline v8)")
    periode = meta.get("periode_data", ["-", "-"])
    skenario = meta.get("skenario", [])
    comp_hc = meta.get("skenario_perbandingan_hindcast", ["A_dasar", "B_CWB3", "D_ENSO_CH"])

    penjelasan_asi = meta.get(
        "penjelasan_indeks_asi",
        "ASI menggunakan empat komponen terstandardisasi (2001-2018) dengan bobot dasar 25% per komponen; "
        "sifat adaptif dijalankan oleh Random Forest.",
    )
    catatan_meta = meta.get("catatan_interpretasi", meta.get("catatan", []))
    catatan_teks = (
        "\n  - " + "\n  - ".join(catatan_meta)
        if isinstance(catatan_meta, list)
        else str(catatan_meta)
    )

    st.markdown(
        f"""
- **Judul Karya Tulis Ilmiah:** *{JUDUL_KTI}*
- **Desain Keluaran Ganda (Dual-Target Machine Learning):**
  - **Target Utama:** Prediksi *Atmospheric Stress Index (ASI)* H+1, H+2, H+3 (cekaman agroklimat).
  - **Target Pendukung:** Prediksi *anomali NDVI* (`ndvi_z`) H+1, H+2, H+3 (respons vegetasi sawah).
- **Wilayah Kajian:** {len(nama_kab)} kabupaten/kota sentra produksi padi di Provinsi Jawa Barat (luas sawah LBS ≥ **{meta.get('min_sawah_km2', 10)} km²**).
- **Sumber Data:** {meta.get('sumber_hujan', 'BMKG per ZOM dan CHIRPS')}, ERA5-Land, MODIS MOD13Q1 & MOD11A2, serta ENSO (Niño 3.4) & IOD (DMI) periode **{periode[0]}** s.d. **{periode[1]}**.
- **Struktur Indeks ASI:** {penjelasan_asi}
- **Ambang Peringatan & Indikasi:**
  - **Peringatan Cekaman ASI (Target Utama):** Waspada **≥ 0.35**, Tinggi **≥ 0.75** (cekaman agroklimat meningkat).
  - **Indikasi Respons Vegetasi NDVI (Target Pendukung):** Waspada **≤ -0.5 SD**, Tinggi **≤ -1.0 SD** (penurunan respons vegetasi).
- **Simulasi Hindcast Retrospektif:** Basis **{basis_hindcast}** untuk target **{target_hindcast}** menggunakan model utama **{fc_key_aktif}** (dikomparasikan dengan **{", ".join(comp_hc)}**).
- **Skenario Model Teruji (11 Skenario pada Sampel Identik):** {", ".join(map(str, skenario)) if skenario else "-"}.
- **Klarifikasi Metodologi & Interpretasi:**{catatan_teks}
"""
    )


# ============================================================
# TAB 6 - EVALUASI ERROR MODEL DAN CONFUSION MATRIX KTI
# ============================================================

with tab6:
    st.subheader("Evaluasi Error Model & Confusion Matrix (Expanding-Window Temporal CV)")

    if D["det"] is not None and isinstance(D["det"], pd.DataFrame):
        with st.expander("📋 Ringkasan Kinerja Deteksi Ambang Peringatan ASI & Indikasi NDVI (deteksi_ambang_risiko.csv)", expanded=True):
            st.dataframe(D["det"].round(3), hide_index=True, use_container_width=True)

    kol_oos_eval = {"skenario", "horizon", "aktual", "pred", "tgt_date"}
    if not kol_oos_eval.issubset(oos.columns):
        st.error(
            "Kolom prediksi OOS belum lengkap untuk TAB 6. "
            f"Kolom yang diperlukan: {', '.join(sorted(kol_oos_eval))}"
        )
    elif oos.empty:
        st.warning("Data prediksi OOS kosong.")
    else:
        opsi = sorted(oos["skenario"].dropna().unique().tolist())
        horizon_tersedia = sorted(
            pd.to_numeric(oos["horizon"], errors="coerce").dropna().unique().tolist()
        )
        target_oos_opsi = sorted(oos["target"].dropna().unique().tolist())

        if not opsi or not horizon_tersedia:
            st.warning("Skenario atau horizon evaluasi tidak tersedia.")
        else:
            idx_skenario = opsi.index(fc_key_aktif) if fc_key_aktif in opsi else 0

            e1, e2, e3 = st.columns(3)
            tgt_eval = e1.selectbox(
                "Target Evaluasi",
                target_oos_opsi,
                format_func=lambda t: "Target 1: ASI" if t == "asi" else "Target 2: NDVI",
                key="tgt_eval_tab6",
            )
            hh = e2.selectbox("Horizon (bulan)", horizon_tersedia, key="h_eval")
            sk = e3.selectbox("Skenario", opsi, index=idx_skenario, key="sk_eval")

            o = oos[
                (oos["target"] == tgt_eval)
                & (oos["horizon"] == hh)
                & (oos["skenario"] == sk)
            ].dropna(subset=["aktual", "pred", "tgt_date"]).copy()

            if o.empty:
                st.warning("Tidak ada pasangan aktual dan prediksi untuk pilihan ini.")
            else:
                o["aktual"] = pd.to_numeric(o["aktual"], errors="coerce")
                o["pred"] = pd.to_numeric(o["pred"], errors="coerce")
                o = o.dropna(subset=["aktual", "pred"])
                o["galat"] = o["pred"] - o["aktual"]

                if o.empty:
                    st.warning("Seluruh nilai aktual/prediksi tidak valid.")
                else:
                    mse = float((o["galat"] ** 2).mean())
                    sse = float((o["galat"] ** 2).sum())
                    sst = float(((o["aktual"] - o["aktual"].mean()) ** 2).sum())
                    r2 = 1 - sse / sst if sst > 0 else np.nan
                    korelasi = o["pred"].corr(o["aktual"])

                    k = st.columns(6)
                    k[0].metric("MSE", f"{mse:.3f}")
                    k[1].metric("RMSE", f"{np.sqrt(mse):.3f}")
                    k[2].metric("MAE", f"{o['galat'].abs().mean():.3f}")
                    k[3].metric("Bias", f"{o['galat'].mean():.3f}")
                    k[4].metric("R²", f"{r2:.3f}" if pd.notna(r2) else "N/A")
                    k[5].metric("Korelasi r", f"{korelasi:.3f}" if pd.notna(korelasi) else "N/A")

                    satuan_lbl = "Indeks ASI" if tgt_eval == "asi" else "Anomali NDVI (SD)"
                    st.caption(
                        f"Satuan galat = **{satuan_lbl}**. "
                        "Gunakan ikon kamera di pojok kanan atas grafik untuk mengunduh gambar PNG resolusi tinggi (~300 DPI)."
                    )

                    a, b = st.columns(2)
                    with a:
                        samp = o.sample(min(4000, len(o)), random_state=1)
                        fs = px.scatter(
                            samp,
                            x="aktual",
                            y="pred",
                            opacity=0.3,
                            labels={
                                "aktual": f"Aktual ({satuan_lbl})",
                                "pred": f"Prediksi ({satuan_lbl})",
                            },
                            title="Prediksi vs Aktual",
                        )
                        lo = float(min(o["aktual"].min(), o["pred"].min()))
                        hi_ = float(max(o["aktual"].max(), o["pred"].max()))
                        fs.add_shape(
                            type="line",
                            x0=lo,
                            y0=lo,
                            x1=hi_,
                            y1=hi_,
                            line=dict(color="black", dash="dash"),
                        )
                        fs.update_layout(height=380, margin=dict(t=38, b=10, l=10, r=10))
                        st.plotly_chart(
                            fs,
                            use_container_width=True,
                            config=konfig_unduh(f"scatter_aktual_pred_{tgt_eval}_{sk}_H{hh}"),
                        )

                    with b:
                        fh = px.histogram(
                            o,
                            x="galat",
                            nbins=60,
                            title="Sebaran Galat (Prediksi - Aktual)",
                            labels={"galat": f"Galat ({satuan_lbl})"},
                            color_discrete_sequence=["#239B75"],
                        )
                        fh.add_vline(x=0, line_dash="dash")
                        fh.update_layout(height=380, margin=dict(t=38, b=10, l=10, r=10))
                        st.plotly_chart(
                            fh,
                            use_container_width=True,
                            config=konfig_unduh(f"histogram_galat_{tgt_eval}_{sk}_H{hh}"),
                        )

                    if "kabupaten" in o.columns:
                        c, d_ = st.columns(2)
                        with c:
                            pk = (
                                o.groupby("kabupaten")["galat"]
                                .apply(lambda s: float(np.sqrt(np.mean(s ** 2))))
                                .sort_values()
                                .reset_index(name="RMSE")
                            )
                            fk = px.bar(
                                pk,
                                x="RMSE",
                                y="kabupaten",
                                orientation="h",
                                title="RMSE per Kabupaten/Kota",
                                color_discrete_sequence=["#1E6351"],
                            )
                            fk.update_layout(height=500, margin=dict(t=38, b=10, l=10, r=10))
                            st.plotly_chart(
                                fk,
                                use_container_width=True,
                                config=konfig_unduh(f"rmse_kabupaten_{tgt_eval}_{sk}_H{hh}"),
                            )

                        with d_:
                            o["bln_target"] = o["tgt_date"].dt.month
                            pb = (
                                o.groupby("bln_target")["galat"]
                                .apply(lambda s: float(np.sqrt(np.mean(s ** 2))))
                                .reset_index(name="RMSE")
                            )
                            fb = px.bar(
                                pb,
                                x="bln_target",
                                y="RMSE",
                                title="RMSE per Bulan Target",
                                color_discrete_sequence=["#239B75"],
                            )
                            fb.update_layout(
                                height=500,
                                margin=dict(t=38, b=10, l=10, r=10),
                                xaxis=dict(dtick=1, title="Bulan target"),
                            )
                            st.plotly_chart(
                                fb,
                                use_container_width=True,
                                config=konfig_unduh(f"rmse_bulanan_{tgt_eval}_{sk}_H{hh}"),
                            )

                    # Confusion matrix 3 kelas peringatan/indikasi
                    st.subheader(
                        "Confusion Matrix Kelas Peringatan ASI"
                        if tgt_eval == "asi"
                        else "Confusion Matrix Kelas Indikasi Respons Vegetasi NDVI"
                    )
                    fc_h = fc[fc["horizon"] == hh].copy()

                    if tgt_eval == "asi":
                        t_hi = float(fc_h["ambang_asi_tinggi"].dropna().iloc[0]) if "ambang_asi_tinggi" in fc_h.columns else 0.75
                        t_wa = float(fc_h["ambang_asi_waspada"].dropna().iloc[0]) if "ambang_asi_waspada" in fc_h.columns else 0.35

                        def kelas(v):
                            return np.select(
                                [v >= t_hi, v >= t_wa],
                                ["Tinggi", "Waspada"],
                                default="Normal",
                            )
                    else:
                        t_hi = float(fc_h["ambang_ndvi_tinggi"].dropna().iloc[0]) if "ambang_ndvi_tinggi" in fc_h.columns else -1.0
                        t_wa = float(fc_h["ambang_ndvi_waspada"].dropna().iloc[0]) if "ambang_ndvi_waspada" in fc_h.columns else -0.5

                        def kelas(v):
                            return np.select(
                                [v <= t_hi, v <= t_wa],
                                ["Tinggi", "Waspada"],
                                default="Normal",
                            )

                    urut = ["Normal", "Waspada", "Tinggi"]
                    o["kls_aktual"] = kelas(o["aktual"].to_numpy())
                    o["kls_pred"] = kelas(o["pred"].to_numpy())
                    cm = pd.crosstab(
                        o["kls_aktual"], o["kls_pred"]
                    ).reindex(index=urut, columns=urut, fill_value=0)

                    norm = st.checkbox(
                        "Tampilkan persentase per baris (= recall)", value=True
                    )
                    tampil = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) if norm else cm
                    fcm = px.imshow(
                        tampil,
                        text_auto=".2f" if norm else "d",
                        color_continuous_scale="Greens",
                        labels=dict(x="Kelas prediksi", y="Kelas aktual", color=""),
                    )
                    fcm.update_layout(
                        height=440,
                        margin=dict(t=45, b=20, l=20, r=20),
                        title=f"Confusion Matrix ({tgt_eval.upper()} | {sk} | H+{hh})",
                        font=dict(size=13, color="#0F2E26"),
                    )
                    st.plotly_chart(
                        fcm,
                        use_container_width=True,
                        config=konfig_unduh(f"confusion_matrix_{tgt_eval}_{sk}_H{hh}"),
                    )

                    rows = []
                    for cl in urut:
                        tp_k = int(cm.loc[cl, cl])
                        pr = tp_k / int(cm[cl].sum()) if int(cm[cl].sum()) else np.nan
                        rc = tp_k / int(cm.loc[cl].sum()) if int(cm.loc[cl].sum()) else np.nan
                        f1_k = (
                            2 * pr * rc / (pr + rc)
                            if pd.notna(pr) and pd.notna(rc) and (pr + rc) > 0
                            else np.nan
                        )
                        rows.append({
                            "Kelas": cl,
                            "Jumlah Aktual": int(cm.loc[cl].sum()),
                            "Precision": round(pr, 3) if pd.notna(pr) else np.nan,
                            "Recall": round(rc, 3) if pd.notna(rc) else np.nan,
                            "F1": round(f1_k, 3) if pd.notna(f1_k) else np.nan,
                        })
                    st.dataframe(
                        pd.DataFrame(rows), hide_index=True, use_container_width=True
                    )
                    total_cm = int(cm.to_numpy().sum())
                    akurasi = float(np.trace(cm.to_numpy()) / total_cm) if total_cm else np.nan
                    st.write(
                        f"**Akurasi keseluruhan 3 kelas:** {akurasi:.3f}"
                        if pd.notna(akurasi)
                        else "**Akurasi keseluruhan 3 kelas:** N/A"
                    )


# ============================================================
# TAB 7 - GALERI GAMBAR & GRAFIK LOKAL UNTUK KTI
# ============================================================

with tab7:
    st.subheader("Galeri Gambar & Grafik Hasil Analisis Lokal (Siap KTI)")
    st.caption(
        f"Memindai otomatis seluruh file gambar (.png, .jpg, .jpeg) di folder hasil aktif (`{RES_AKTIF}`) "
        f"dan folder utama (`{OUT}`)."
    )

    ekstensi_gambar = {".png", ".jpg", ".jpeg", ".webp"}
    semua_gambar = sorted(
        [
            p for p in OUT.rglob("*")
            if p.is_file() and p.suffix.lower() in ekstensi_gambar
        ],
        key=lambda p: (0 if RES_AKTIF in p.parents else 1, p.name.lower()),
    )

    if not semua_gambar:
        st.warning(
            f"Belum ditemukan file gambar (.png / .jpg) di dalam `{OUT}`. "
            "Pastikan skrip `02_ml_v8.py` menyimpan plot ke folder tersebut, atau gunakan "
            "ikon kamera di pojok kanan atas grafik pada Tab 1, Tab 3, dan Tab 6 untuk mengunduh gambar PNG resolusi tinggi."
        )
    else:
        col_f1, col_f2, col_f3 = st.columns([2.2, 1.0, 1.4])
        kata_kunci = col_f1.text_input(
            "Filter nama gambar (contoh: confusion, cm, rmse, shap, peta, asi, ndvi)",
            value="",
            key="filter_gambar_kti",
        ).strip().lower()

        kolom_grid = col_f2.selectbox(
            "Jumlah kolom tampilan",
            [1, 2, 3],
            index=1,
            key="kolom_galeri_kti",
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for g_path in semua_gambar:
                zf.write(g_path, arcname=str(g_path.relative_to(OUT)))

        with col_f3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.download_button(
                f"📦 Unduh Semua ({len(semua_gambar)} Gambar ZIP)",
                data=zip_buf.getvalue(),
                file_name=f"Gambar_KTI_{st.session_state.folder_aktif}.zip",
                mime="application/zip",
                use_container_width=True,
            )

        gambar_terfilter = [
            p for p in semua_gambar
            if not kata_kunci or kata_kunci in p.name.lower() or kata_kunci in str(p.parent.name).lower()
        ]

        st.write(f"Menampilkan **{len(gambar_terfilter)}** dari **{len(semua_gambar)}** file gambar lokal.")

        if not gambar_terfilter:
            st.info("Tidak ada gambar yang cocok dengan kata kunci filter.")
        else:
            for idx in range(0, len(gambar_terfilter), kolom_grid):
                cols = st.columns(kolom_grid)
                for col_i, img_path in enumerate(gambar_terfilter[idx : idx + kolom_grid]):
                    with cols[col_i]:
                        rel_name = str(img_path.relative_to(OUT))
                        st.markdown(f"**`{rel_name}`**")
                        st.image(str(img_path), use_container_width=True)
                        st.download_button(
                            label=f"⬇️ Unduh {img_path.name}",
                            data=img_path.read_bytes(),
                            file_name=img_path.name,
                            mime=f"image/{img_path.suffix.lstrip('.').lower()}",
                            key=f"dl_img_{idx}_{col_i}_{rel_name}",
                            use_container_width=True,
                        )
                        st.divider()