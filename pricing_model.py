# -*- coding: utf-8 -*-
"""
Health Insurance Pricing Model — Dummy Project (Aktuaria)
==========================================================
Dataset: Medical Cost Personal (Kaggle)
https://www.kaggle.com/datasets/mirichoi0218/insurance

Alur:
1. Exploratory Data Analysis (EDA)
2. Pricing Model: OLS vs GLM Gamma (log link)
3. Interpretasi koefisien GLM
4. Pengembangan: perbandingan dengan Tweedie + analisis kasus ekstrem

Catatan: jalankan di Google Colab dengan insurance.csv sudah di-upload
ke direktori kerja (root /content/), atau sesuaikan path read_csv di bawah.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 110

# Folder output (khusus Colab). Ganti ke folder lokal kalau run di Jupyter biasa.
OUTPUT_DIR = '/content/charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = 'insurance.csv'
FORMULA = "charges ~ age + bmi + children + C(smoker) + C(sex) + C(region)"


def evaluate(y_true, y_pred, name):
    """Hitung MAE, RMSE, R2 dan cetak hasilnya."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R2   : {r2:.4f}")
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'n': len(y_true)}


# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
df['smoker_num'] = df['smoker'].map({'yes': 1, 'no': 0})

print(f"Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
print(df.isnull().sum().rename('missing_values'))


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# --- 2.1 Distribusi charges (asli vs log) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(df['charges'], bins=40, kde=True, ax=axes[0], color='#2563eb')
axes[0].set_title('Distribusi Charges (skala asli)')
axes[0].set_xlabel('Charges ($)')

sns.histplot(np.log(df['charges']), bins=40, kde=True, ax=axes[1], color='#7c3aed')
axes[1].set_title('Distribusi log(Charges)')
axes[1].set_xlabel('log(Charges)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_distribusi_charges.png'), bbox_inches='tight')
plt.show()

# --- 2.2 Charges by smoker ---
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x='smoker', y='charges', hue='smoker',
            palette=['#22c55e', '#ef4444'], ax=ax, legend=False)
ax.set_title('Charges: Perokok vs Bukan Perokok')
ax.set_xlabel('Smoker')
ax.set_ylabel('Charges ($)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_charges_by_smoker.png'), bbox_inches='tight')
plt.show()

# --- 2.3 Charges by region & sex ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x='region', y='charges', ax=axes[0], color='#60a5fa')
axes[0].set_title('Charges by Region')
axes[0].tick_params(axis='x', rotation=15)

sns.boxplot(data=df, x='sex', y='charges', ax=axes[1], color='#f59e0b')
axes[1].set_title('Charges by Sex')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_charges_region_sex.png'), bbox_inches='tight')
plt.show()

# --- 2.4 Scatter: age & bmi vs charges (by smoker) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=df, x='age', y='charges', hue='smoker',
                 palette={'no': '#22c55e', 'yes': '#ef4444'}, alpha=0.6, ax=axes[0])
axes[0].set_title('Age vs Charges')

sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker',
                 palette={'no': '#22c55e', 'yes': '#ef4444'}, alpha=0.6, ax=axes[1])
axes[1].set_title('BMI vs Charges')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_scatter_age_bmi.png'), bbox_inches='tight')
plt.show()

# --- 2.5 Correlation heatmap ---
corr = df[['age', 'bmi', 'children', 'smoker_num', 'charges']].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_correlation_heatmap.png'), bbox_inches='tight')
plt.show()

# --- 2.6 Ringkasan angka penting ---
summary = {
    'n_rows': len(df),
    'avg_charges': df['charges'].mean(),
    'median_charges': df['charges'].median(),
    'avg_charges_smoker': df[df.smoker == 'yes']['charges'].mean(),
    'avg_charges_nonsmoker': df[df.smoker == 'no']['charges'].mean(),
    'pct_smoker': (df['smoker'] == 'yes').mean() * 100,
    'corr_age_charges': corr.loc['age', 'charges'],
    'corr_bmi_charges': corr.loc['bmi', 'charges'],
    'corr_smoker_charges': corr.loc['smoker_num', 'charges'],
}
print("\n" + "=" * 50)
print("RINGKASAN EDA")
print("=" * 50)
for k, v in summary.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")


# ============================================================
# 3. PRICING MODEL: OLS vs GLM GAMMA
# ============================================================
train, test = train_test_split(df, test_size=0.2, random_state=42)

ols_model = smf.ols(formula=FORMULA, data=train).fit()
glm_model = smf.glm(formula=FORMULA, data=train,
                     family=sm.families.Gamma(link=sm.families.links.Log())).fit()

pred_ols = ols_model.predict(test)
pred_glm = glm_model.predict(test)

print("\n" + "=" * 50)
print("EVALUASI MODEL (test set, 20% data)")
print("=" * 50)
res_ols = evaluate(test['charges'], pred_ols, "OLS (Linear Regression)")
res_glm = evaluate(test['charges'], pred_glm, "GLM Gamma (log link)")

# --- Simpan model GLM untuk dipakai dashboard nanti ---
with open(os.path.join(OUTPUT_DIR, 'glm_model.pkl'), 'wb') as f:
    pickle.dump(glm_model, f)

# --- Plot actual vs predicted ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(test['charges'], pred_ols, alpha=0.5, color='#f59e0b')
axes[0].plot([0, 65000], [0, 65000], 'k--', lw=1)
axes[0].set_title(f"OLS: Actual vs Predicted (R\u00b2={res_ols['R2']:.3f})")
axes[0].set_xlabel('Actual Charges')
axes[0].set_ylabel('Predicted Charges')

axes[1].scatter(test['charges'], pred_glm, alpha=0.5, color='#2563eb')
axes[1].plot([0, 65000], [0, 65000], 'k--', lw=1)
axes[1].set_title(f"GLM Gamma: Actual vs Predicted (R\u00b2={res_glm['R2']:.3f})")
axes[1].set_xlabel('Actual Charges')
axes[1].set_ylabel('Predicted Charges')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_actual_vs_predicted.png'), bbox_inches='tight')
plt.show()

# --- Residual plot GLM ---
residuals = test['charges'] - pred_glm
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(pred_glm, residuals, alpha=0.5, color='#7c3aed')
ax.axhline(0, color='black', linestyle='--', lw=1)
ax.set_title('Residual Plot - GLM Gamma Model')
ax.set_xlabel('Predicted Charges')
ax.set_ylabel('Residual (Actual - Predicted)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_residual_plot.png'), bbox_inches='tight')
plt.show()


# ============================================================
# 4. INTERPRETASI KOEFISIEN GLM
# ============================================================
coef_table = pd.DataFrame({
    'coef': glm_model.params,
    'exp_coef (faktor)': np.exp(glm_model.params),
    'p_value': glm_model.pvalues
})
print("\n" + "=" * 50)
print("GLM COEFFICIENTS (sebagai faktor multiplikatif, exp(coef))")
print("=" * 50)
print(coef_table.round(4))
coef_table.round(4).to_csv(os.path.join(OUTPUT_DIR, 'glm_coefficients.csv'))


# ============================================================
# 5. PENGEMBANGAN: OLS vs GLM GAMMA vs TWEEDIE
# ============================================================
tweedie_model = smf.glm(formula=FORMULA, data=train,
                         family=sm.families.Tweedie(
                             var_power=1.5,
                             link=sm.families.links.Log()
                         )).fit()

test = test.copy()
test['pred_ols'] = pred_ols
test['pred_glm_gamma'] = pred_glm
test['pred_tweedie'] = tweedie_model.predict(test)

# --- Evaluasi keseluruhan ---
df_all = pd.DataFrame([
    evaluate(test['charges'], test['pred_ols'], 'OLS'),
    evaluate(test['charges'], test['pred_glm_gamma'], 'GLM Gamma'),
    evaluate(test['charges'], test['pred_tweedie'], 'Tweedie'),
])

# --- Evaluasi khusus kasus ekstrem (charges > 30000) ---
extreme = test[test['charges'] > 30000].copy()
df_extreme = pd.DataFrame([
    evaluate(extreme['charges'], extreme['pred_ols'], 'OLS'),
    evaluate(extreme['charges'], extreme['pred_glm_gamma'], 'GLM Gamma'),
    evaluate(extreme['charges'], extreme['pred_tweedie'], 'Tweedie'),
])

print("\n" + "=" * 60)
print(f"EVALUASI KESELURUHAN (n={len(test)} data test)")
print("=" * 60)
print(df_all.to_string(index=False))

print("\n" + "=" * 60)
print(f"EVALUASI KASUS EKSTREM: charges > 30000 (n={len(extreme)} data)")
print("=" * 60)
print(df_extreme.to_string(index=False))

df_all.to_csv(os.path.join(OUTPUT_DIR, 'perbandingan_model_keseluruhan.csv'), index=False)
df_extreme.to_csv(os.path.join(OUTPUT_DIR, 'perbandingan_model_kasus_ekstrem.csv'), index=False)

# --- Chart: bar chart RMSE keseluruhan vs ekstrem ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['#f59e0b', '#2563eb', '#22c55e']

axes[0].bar(df_all['Model'], df_all['RMSE'], color=colors)
axes[0].set_title('RMSE - Keseluruhan Test Set')
axes[0].set_ylabel('RMSE ($)')
for i, v in enumerate(df_all['RMSE']):
    axes[0].text(i, v + 50, f'{v:,.0f}', ha='center')

axes[1].bar(df_extreme['Model'], df_extreme['RMSE'], color=colors)
axes[1].set_title('RMSE - Kasus Ekstrem (charges > 30000)')
axes[1].set_ylabel('RMSE ($)')
for i, v in enumerate(df_extreme['RMSE']):
    axes[1].text(i, v + 100, f'{v:,.0f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_perbandingan_rmse.png'), bbox_inches='tight')
plt.show()

# --- Chart: actual vs predicted, khusus kasus ekstrem, 3 model ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_cols = ['pred_ols', 'pred_glm_gamma', 'pred_tweedie']
model_names = ['OLS', 'GLM Gamma', 'Tweedie']

for ax, col, name, c in zip(axes, model_cols, model_names, colors):
    ax.scatter(extreme['charges'], extreme[col], alpha=0.6, color=c)
    lims = [20000, 65000]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_title(f'{name} - Kasus Ekstrem')
    ax.set_xlabel('Actual Charges')
    ax.set_ylabel('Predicted Charges')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_prediksi_kasus_ekstrem.png'), bbox_inches='tight')
plt.show()

print(f"\nSemua chart & tabel tersimpan di {OUTPUT_DIR}/")
