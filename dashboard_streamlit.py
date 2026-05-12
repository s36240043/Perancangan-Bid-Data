import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Konfigurasi Halaman Dasar
st.set_page_config(page_title="Ad-Fraud ML Simulator", page_icon="🛡️", layout="wide")

# ==========================================
# FUNGSI MUAT DATA (Cached untuk kecepatan)
# ==========================================
@st.cache_data
def load_agg_data():
    # Mencoba memuat file CSV Anda. Jika tidak ada di direktori yang sama, 
    # kita gunakan data fallback berdasarkan gambar yang Anda berikan agar aplikasi tetap berjalan.
    try:
        df = pd.read_csv("aggregation_results.csv")
    except FileNotFoundError:
        data = {
            "threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "bot_count": [14709432, 14574304, 13643546, 11928112, 10950962, 10560644, 10092536, 9752599, 9206215],
            "human_count": [290568, 425696, 1356454, 3071888, 4049038, 4439356, 4907464, 5247401, 5793785]
        }
        df = pd.DataFrame(data)
    return df

df_agg = load_agg_data()

# ==========================================
# NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("🛡️ Fraud Defense Command")
st.sidebar.markdown("---")
page = st.sidebar.radio("Pilih Modul Presentasi:", [
    "1. Lanskap Makro",
    "2. Arsitektur ML & Threshold",
    "3. Bukti Forensik (CTIT)",
    "4. Simulator ROI & Bisnis"
])

# ==========================================
# HALAMAN 1: LANSKAP MAKRO
# ==========================================
if page == "1. Lanskap Makro":
    st.title("Lanskap Makro: Skala Penipuan")
    st.markdown("Melihat seberapa besar serangan *bot* mendistorsi trafik organik kita sebelum *Machine Learning* diterapkan.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trafik Masuk", "15,000,000", "Klik")
    col2.metric("Total Konversi", "27,666", "Instalasi")
    col3.metric("Rasio Konversi (CR) Mentah", "0.18%", "- Kritis", delta_color="off")
    
    st.markdown("### Funnel Leakage")
    fig = go.Figure(go.Funnel(
        y = ["Klik Total", "Lolos Aturan Dasar", "Konversi Asli"],
        x = [15000000, 2675282, 17896],
        textinfo = "value+percent initial"
    ))
    fig.update_layout(margin={"l": 0, "r": 0, "t": 30, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# HALAMAN 2: ARSITEKTUR ML & THRESHOLD
# ==========================================
elif page == "2. Arsitektur ML & Threshold":
    st.title("Simulasi Kebijakan Pemblokiran (Thresholding)")
    st.markdown("Mengatur seberapa agresif model Semi-Supervised XGBoost kita dalam memblokir *Channel Spraying*.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Kontrol Kebijakan")
    # Slider mengambil nilai dari data agregasi Anda (0.1 hingga 0.9)
    selected_threshold = st.sidebar.select_slider(
        "Ambang Batas Probabilitas Bot (Threshold):",
        options=df_agg['threshold'].tolist(),
        value=0.8
    )
    
    # Filter data berdasarkan threshold
    current_data = df_agg[df_agg['threshold'] == selected_threshold].iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.error(f"🤖 Bot Diblokir: **{current_data['bot_count']:,}**")
    with col2:
        st.success(f"👤 Manusia Diselamatkan: **{current_data['human_count']:,}**")
        
    # Visualisasi Donut Chart
    fig = px.pie(
        values=[current_data['bot_count'], current_data['human_count']], 
        names=["Bot (Dibuang)", "Manusia (Dipertahankan)"],
        color_discrete_sequence=["#EF553B", "#00CC96"],
        hole=0.4,
        title=f"Distribusi Trafik pada Threshold {selected_threshold}"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Insight Bisnis:** Menurunkan threshold membuat sistem lebih aman dari bot, tetapi mengorbankan lebih banyak pengguna publik (False Positives).")

# ==========================================
# HALAMAN 3: BUKTI FORENSIK (CTIT)
# ==========================================
elif page == "3. Bukti Forensik (CTIT)":
    st.title("Validasi Forensik: Click-to-Install Time")
    st.markdown("Bukti independen mengapa klik yang ditandai sebagai bot oleh ML kita memang bukan manusia.")
    
    # Karena kita tidak meload 15 juta baris, kita gunakan dummy distribusi statistik untuk Boxplot
    # yang mencerminkan tabel temuan Anda sebelumnya.
    np.random.seed(42)
    data_organic = np.random.lognormal(mean=np.log(60), sigma=0.8, size=1000)
    data_bot = np.random.lognormal(mean=np.log(1800), sigma=1.5, size=1000)
    data_cleared = np.random.lognormal(mean=np.log(1700), sigma=1.2, size=1000)
    
    df_box = pd.DataFrame({
        "Grup": ["PASS_ORGANIC"]*1000 + ["ML_DETECTED_BOT"]*1000 + ["CLEARED_BY_ML"]*1000,
        "CTIT_Detik": np.concatenate([data_organic, data_bot, data_cleared])
    })
    
    fig = px.box(
        df_box, x="Grup", y="CTIT_Detik", color="Grup",
        log_y=True, # SANGAT PENTING: Menggunakan skala Logaritmik untuk visibilitas
        title="Distribusi Waktu Instalasi (Skala Logaritmik)",
        color_discrete_sequence=["#00CC96", "#EF553B", "#FFA15A"]
    )
    fig.update_layout(yaxis_title="Detik menuju Instalasi (Log Scale)")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# HALAMAN 4: SIMULATOR ROI (DENGAN GUARDRAILS)
# ==========================================
elif page == "4. Simulator ROI & Bisnis":
    st.title("Simulator Dampak Finansial (ROI)")
    st.markdown("Kalkulasi penghematan anggaran secara seketika berdasarkan temuan model.")
    
    # ---------------------------------------------------------
    # MATHEMATICAL GUARDRAILS (PENGAMAN)
    # Kita menggunakan parameter min_value, max_value, dan step
    # agar manajemen tidak bisa memasukkan angka "halu".
    # ---------------------------------------------------------
    st.sidebar.markdown("### Asumsi Finansial")
    cpc = st.sidebar.number_input(
        "Harga Per Klik (CPC) dalam USD:", 
        min_value=0.01,   # Guardrail: Harga klik tidak mungkin gratis/negatif
        max_value=2.00,   # Guardrail: Harga klik massal tidak mungkin > $2 di industri game/app
        value=0.05, 
        step=0.01
    )
    
    cpa = st.sidebar.number_input(
        "Komisi Per Instalasi (CPA/CPI) dalam USD:", 
        min_value=0.50,   # Guardrail: Komisi tidak mungkin sangat kecil
        max_value=50.00,  # Guardrail: Batas wajar
        value=2.50, 
        step=0.50
    )
    
    # Ambil data pemblokiran bot rata-rata dari agregasi Anda (misal kita ambil threshold 0.8)
    bot_clicks_blocked = df_agg[df_agg['threshold'] == 0.8]['bot_count'].values[0]
    bot_installs_blocked = 14164 # Angka dari presentasi kita sebelumnya
    
    # Kalkulasi ROI
    saved_cpc = bot_clicks_blocked * cpc
    saved_cpa = bot_installs_blocked * cpa
    total_saved = saved_cpc + saved_cpa
    
    # Layout Metrik Finansial
    col1, col2 = st.columns(2)
    col1.metric("Anggaran Klik Terselamatkan", f"${saved_cpc:,.2f}")
    col2.metric("Pencurian Komisi Dicegah", f"${saved_cpa:,.2f}")
    
    st.markdown("---")
    st.metric("Total Penyelamatan Anggaran (Net ROI)", f"${total_saved:,.2f}", "+ Laba Bersih Operasional")
    
    # Logical Guardrail Warning
    # if total_saved > 1000000:
    #     st.warning("⚠️ **Peringatan Simulasi:** Nilai CPC/CPA yang Anda masukkan menghasilkan estimasi penghematan di atas \$1 Juta. Di dunia nyata, angka sebesar ini akan memicu teguran otomatis dari jaringan iklan untuk meninjau ulang kontrak lelang.")
    
    # Visualisasi Waterfall
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "total"],
        x = ["Pencegahan Pemborosan Klik", "Pencegahan Pencurian Komisi", "Total Anggaran Diselamatkan"],
        textposition = "outside",
        text = [f"${saved_cpc/1000:.0f}K", f"${saved_cpa/1000:.0f}K", f"${total_saved/1000:.0f}K"],
        y = [saved_cpc, saved_cpa, total_saved],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#EF553B"}},
        increasing = {"marker":{"color":"#00CC96"}},
        totals = {"marker":{"color":"#1f77b4"}}
    ))
    fig.update_layout(title="Aliran Dana Terselamatkan", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)