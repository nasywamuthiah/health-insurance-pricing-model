"""
Health Insurance Pricing Dashboard
====================================
Dashboard interaktif untuk estimasi biaya asuransi kesehatan berdasarkan
profil risiko, menggunakan model GLM Gamma (log link) yang sudah kita
bangun sebelumnya.

CARA JALANIN (di laptop kamu):
1. Pastikan file ini dan 'insurance.csv' ada di folder yang sama
2. Install library yang dibutuhkan (sekali aja):
   pip install streamlit pandas numpy statsmodels matplotlib seaborn
3. Buka terminal di folder itu, jalanin:
   streamlit run dashboard.py
4. Browser otomatis kebuka di localhost:8501

CARA DEPLOY (biar bisa diakses online, gratis):
1. Push file ini + insurance.csv ke GitHub repo kamu
2. Buka https://share.streamlit.io, login pakai akun GitHub
3. Klik "New app", pilih repo kamu, pilih file "dashboard.py"
4. Klik Deploy - selesai, dapat link publik!
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
# set_page_config harus dipanggil pertama kali, sebelum elemen lain
st.set_page_config(
    page_title="Health Insurance Pricing Dashboard",
    page_icon="🏥",
    layout="wide"
)

sns.set_style("whitegrid")


# ============================================================
# LOAD DATA & TRAIN MODEL
# ============================================================
# @st.cache_data artinya: fungsi ini cuma dijalankan SEKALI, hasilnya
# disimpan di memori. Jadi walaupun user gonta-ganti slider berkali-kali,
# Streamlit nggak perlu load ulang data & training ulang model tiap kali
# (yang bikin app jadi lambat). Ini best practice penting di Streamlit.
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df


@st.cache_resource  # khusus buat cache objek model (bukan data biasa)
def train_model(df):
    formula = "charges ~ age + bmi + children + C(smoker) + C(sex) + C(region)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Gamma(link=sm.families.links.Log())
    ).fit()
    return model


df = load_data()
model = train_model(df)


# ============================================================
# HEADER
# ============================================================
st.title("🏥 Health Insurance Pricing Dashboard")
st.markdown(
    "Dashboard ini mengestimasi biaya asuransi kesehatan berdasarkan "
    "profil risiko individu, menggunakan **GLM Gamma (log link)** — "
    "pendekatan standar aktuaria untuk data klaim yang *right-skewed*."
)
st.divider()


# ============================================================
# SIDEBAR: INPUT PROFIL RISIKO
# ============================================================
# Semua widget input (slider, radio, dll) ditaruh di sidebar biar
# dashboard-nya rapi: input di kiri, hasil di kanan/tengah.
st.sidebar.header("Profil Peserta")

age = st.sidebar.slider("Usia", min_value=18, max_value=64, value=30)
bmi = st.sidebar.slider("BMI", min_value=15.0, max_value=53.0, value=25.0, step=0.1)
children = st.sidebar.slider("Jumlah anak/tanggungan", min_value=0, max_value=5, value=0)
sex = st.sidebar.radio("Jenis kelamin", options=["female", "male"])
smoker = st.sidebar.radio("Status merokok", options=["no", "yes"])
region = st.sidebar.selectbox(
    "Region",
    options=["northeast", "northwest", "southeast", "southwest"]
)


# ============================================================
# PREDIKSI
# ============================================================
# Bikin 1 baris data baru sesuai input user, formatnya harus sama
# persis dengan kolom yang dipakai waktu training model
input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex": [sex],
    "smoker": [smoker],
    "region": [region],
})

predicted_charges = model.predict(input_data)[0]

# Hitung juga confidence interval sederhana pakai standard error prediksi
pred_summary = model.get_prediction(input_data).summary_frame()
lower = pred_summary["mean_ci_lower"].values[0]
upper = pred_summary["mean_ci_upper"].values[0]


# ============================================================
# TAMPILAN UTAMA: HASIL PREDIKSI
# ============================================================
col1, col2, col3 = st.columns(3)

col1.metric(
    label="Estimasi Charges Tahunan",
    value=f"${predicted_charges:,.0f}"
)
col2.metric(
    label="Rentang Estimasi (95% CI)",
    value=f"${lower:,.0f} - ${upper:,.0f}"
)
col3.metric(
    label="Persentil dibanding rata-rata dataset",
    value=f"{(df['charges'] < predicted_charges).mean() * 100:.0f}%"
)

st.divider()


# ============================================================
# VISUALISASI: POSISI PREDIKSI DI DISTRIBUSI DATA
# ============================================================
st.subheader("Posisi Estimasi dalam Distribusi Charges")

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["charges"], bins=50, color="#93c5fd", ax=ax)
ax.axvline(predicted_charges, color="#ef4444", linestyle="--", linewidth=2,
           label=f"Estimasi kamu: ${predicted_charges:,.0f}")
ax.set_xlabel("Charges ($)")
ax.set_ylabel("Jumlah orang")
ax.legend()
st.pyplot(fig)

st.caption(
    "Garis merah menunjukkan posisi estimasi charges untuk profil yang kamu "
    "input, dibandingkan dengan distribusi seluruh data historis."
)

st.divider()


# ============================================================
# SIMULASI: EFEK MENGUBAH STATUS MEROKOK
# ============================================================
# Ini bagian yang paling "actuarial" - nunjukin efek 1 faktor risiko
# secara terisolasi, dengan semua faktor lain tetap sama (ceteris paribus)
st.subheader("Simulasi: Efek Status Merokok")

sim_data = pd.DataFrame({
    "age": [age, age],
    "bmi": [bmi, bmi],
    "children": [children, children],
    "sex": [sex, sex],
    "smoker": ["no", "yes"],
    "region": [region, region],
})
sim_pred = model.predict(sim_data)

col_a, col_b = st.columns(2)
col_a.metric("Jika non-smoker", f"${sim_pred[0]:,.0f}")
col_b.metric(
    "Jika smoker",
    f"${sim_pred[1]:,.0f}",
    delta=f"+{(sim_pred[1] / sim_pred[0] - 1) * 100:.0f}%"
)

st.caption(
    "Dengan profil usia, BMI, jumlah anak, dan region yang sama, status "
    "merokok saja bisa mengubah estimasi charges secara signifikan."
)

st.divider()


# ============================================================
# TABEL KOEFISIEN MODEL (buat yang mau lihat detail teknis)
# ============================================================
with st.expander("Lihat detail koefisien model GLM"):
    coef_table = pd.DataFrame({
        "Koefisien (log scale)": model.params,
        "Faktor multiplikatif (exp)": np.exp(model.params),
        "P-value": model.pvalues
    }).round(4)
    st.dataframe(coef_table, width="stretch")
    st.caption(
        "Faktor multiplikatif > 1 artinya variabel tersebut menaikkan "
        "charges; < 1 artinya menurunkan. P-value < 0.05 dianggap "
        "signifikan secara statistik."
    )

st.markdown(
    "---\n"
    "*Dashboard ini dibuat sebagai dummy project pembelajaran ilmu aktuaria, "
    "menggunakan dataset publik [Medical Cost Personal](https://www.kaggle.com/datasets/mirichoi0218/insurance) "
    "dari Kaggle.*"
)
