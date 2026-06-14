import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Membaca file Excel
data = pd.read_excel("TPAK_TPT_TABEL.xlsx")
data.to_csv("TPAK_TPT.csv", index=False)

print("Jumlah Data:", len(data))

# Variabel X dan Y
X = data[['TPAK']]
y = data['TPT']

# =====================
# SPLIT DATA
# =====================

# 70% Training, 30% sementara
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

# 15% Validation, 15% Testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

print("\n=== PEMBAGIAN DATA ===")
print("Training  :", len(X_train))
print("Validation:", len(X_val))
print("Testing   :", len(X_test))

# =====================
# TRAINING MODEL
# =====================

model = LinearRegression()
model.fit(X_train, y_train)

# =====================
# EVALUASI
# =====================

r2_train = model.score(X_train, y_train)
r2_val = model.score(X_val, y_val)
r2_test = model.score(X_test, y_test)

print("\n=== HASIL REGRESI ===")
print("Koefisien :", model.coef_[0])
print("Intercept :", model.intercept_)

print("\n=== R2 SCORE ===")
print("Training   :", r2_train)
print("Validation :", r2_val)
print("Testing    :", r2_test)

# Persamaan Regresi
print("\n=== PERSAMAAN REGRESI ===")
print(f"TPT = {model.coef_[0]:.4f} × TPAK + {model.intercept_:.4f}")

# Korelasi
korelasi = data['TPAK'].corr(data['TPT'])
print("\nKorelasi :", korelasi)

# =====================
# PREDIKSI SELURUH DATA
# =====================

data['Prediksi_TPT'] = model.predict(X)

print("\n=== HASIL PREDIKSI ===")
print(data)

# Simpan ke Excel
data.to_excel("Hasil_Regresi_TPAK_TPT.xlsx", index=False)

print("\nFile berhasil disimpan:")
print("Hasil_Regresi_TPAK_TPT.xlsx")

import matplotlib.pyplot as plt

plt.scatter(data['TPAK'], data['TPT'])
plt.plot(data['TPAK'], data['Prediksi_TPT'])
plt.xlabel("TPAK")
plt.ylabel("TPT")
plt.title("Regresi Linear TPAK vs TPT")
plt.show()

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='r2'
)

print("CV Scores:", cv_scores)
print("Rata-rata CV Score:", cv_scores.mean())