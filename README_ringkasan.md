# Health Insurance Pricing Model — Dummy Project

Dataset: Medical Cost Personal Dataset (Kaggle) — 1338 baris, 7 kolom, tanpa missing value.

## 1. Exploratory Data Analysis (EDA)

- Rata-rata charges: **$13,270** (median $9,382) — distribusi right-skewed, khas data klaim asuransi.
- **Smoker** adalah variabel paling berpengaruh: rata-rata charges perokok $32,050 vs bukan perokok $8,434 (~4x lipat). Korelasi dengan charges: 0.79.
- Age dan BMI berkorelasi positif tapi lebih lemah (0.30 dan 0.20).
- Charges antar region relatif mirip, sedikit lebih rendah di southwest/southeast.

**Chart tersedia:**
- `01_distribusi_charges.png` — distribusi charges asli & log-transform
- `02_charges_by_smoker.png` — boxplot smoker vs non-smoker
- `03_charges_region_sex.png` — boxplot by region & sex
- `04_scatter_age_bmi.png` — scatter age/bmi vs charges (colored by smoker)
- `05_correlation_heatmap.png` — correlation matrix

## 2. Pricing Model

Dua model dibandingkan:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| OLS (Linear Regression) | 4,181 | 5,796 | 0.784 |
| **GLM Gamma (log link)** | 4,155 | 7,306 | 0.656 |

**Kenapa GLM Gamma dipakai di aktuaria (bukan cuma OLS)?**
Data charges itu selalu positif dan right-skewed (banyak klaim kecil, sedikit klaim besar) — asumsi OLS (error normal, homoskedastik) sering dilanggar. GLM Gamma dengan log link memodelkan charges sebagai perkalian faktor risiko (multiplicative), yang lebih sesuai dengan cara kerja pricing asuransi (base rate × faktor usia × faktor smoker × dst).

Di percobaan ini, RMSE OLS memang lebih rendah — itu wajar karena OLS memang dioptimalkan buat minimalkan squared error, sementara GLM Gamma dioptimalkan lewat likelihood, bukan error kuadrat. Tapi struktur GLM lebih mudah diinterpretasi sebagai faktor pricing.

**Interpretasi koefisien GLM (dalam bentuk faktor pengali terhadap base rate):**

| Faktor | Pengali charges | Signifikan? |
|---|---|---|
| Smoker = yes | **×4.44** | ya (p<0.001) |
| Setiap +1 tahun usia | ×1.028 | ya |
| Setiap +1 BMI | ×1.014 | ya |
| Setiap +1 anak | ×1.077 | ya |
| Region southwest | ×0.861 (lebih murah dari base) | ya (p=0.012) |
| Sex male | ×0.938 | tidak signifikan |

Artinya: kalau base rate charges adalah $1,683 (intercept), seorang perokok usia tertentu punya charges kira-kira 4.4x lebih mahal dibanding non-smoker dengan profil sama.

**Chart tersedia:**
- `06_actual_vs_predicted.png` — perbandingan prediksi OLS vs GLM
- `07_residual_plot.png` — residual plot model GLM
- `glm_coefficients.csv` — tabel lengkap koefisien
- `glm_model.pkl` — model tersimpan (bisa dipakai buat dashboard)

## 3. Pengembangan: Perbandingan dengan Model Tweedie

Bagian ini adalah pengembangan dari analisis GLM Gamma sebelumnya — menambahkan model **Tweedie Regression** dan menguji ketiga model secara khusus di **kasus ekstrem (charges > $30,000)**, sesuai saran yang muncul di studi pembanding (Kelompok 6, UB) yang menemukan GLM Gamma kurang akurat di rentang biaya tinggi.

**Kenapa Tweedie?**
Tweedie adalah distribusi fleksibel yang berada "di antara" Poisson (var_power=1) dan Gamma (var_power=2) — di project ini dipakai var_power=1.5, nilai umum untuk data severity klaim asuransi. Kelebihannya: bisa menangkap kombinasi banyak klaim kecil + sedikit klaim besar sekaligus, yang sering terjadi di data asuransi kesehatan riil.

**Hasil evaluasi keseluruhan (test set, n=268):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| OLS | 4,181 | 5,796 | 0.784 |
| GLM Gamma | 4,155 | 7,307 | 0.656 |
| **Tweedie** | **3,929** | 6,149 | 0.756 |

Tweedie punya **MAE paling rendah** dari ketiganya — errornya rata-rata paling kecil.

**Hasil evaluasi khusus kasus ekstrem (charges > $30,000, n=33):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| OLS | 8,285 | 9,175 | -1.22 |
| GLM Gamma | 8,914 | 10,827 | -2.09 |
| **Tweedie** | **8,064** | 9,691 | -1.47 |

Di segmen ekstrem, ketiga model sama-sama **kesulitan** (R² negatif berarti prediksi lebih buruk dari sekadar menebak rata-rata kelompok ini) — konsisten dengan temuan studi pembanding. Tapi **Tweedie tetap paling unggul** dari sisi MAE dibanding OLS dan GLM Gamma, walau selisihnya tidak besar.

**Insight:** menambah model Tweedie memberi sedikit perbaikan, tapi tidak menyelesaikan masalah utama — model linear/GLM sederhana memang punya keterbatasan struktural untuk menangkap kasus ekstrem/outlier biaya tinggi. Ini membuka ruang riset lanjutan (GAM, model non-linear, atau segmentasi terpisah untuk kelompok risiko tinggi).

**Chart tambahan:**
- `08_perbandingan_rmse.png` — bar chart RMSE ketiga model, keseluruhan vs kasus ekstrem
- `09_prediksi_kasus_ekstrem.png` — scatter actual vs predicted untuk 3 model, khusus data charges > 30000
- `perbandingan_model_keseluruhan.csv` / `perbandingan_model_kasus_ekstrem.csv` — tabel angka lengkap

## Catatan/Limitasi

- Dataset dummy/publik dari Kaggle, bukan data riil perusahaan asuransi tertentu — tidak merepresentasikan pasar asuransi kesehatan Indonesia.
- Model belum memperhitungkan interaksi antar variabel (misal age × smoker, bmi × smoker) yang biasanya penting di pricing riil.
- Belum ada credibility weighting / loss ratio analysis yang biasa dipakai di actuarial pricing profesional.
