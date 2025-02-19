import pandas as pd 
import streamlit as st
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load model
df = pd.read_excel("D:\SEMESTER7/harga_tiket.xlsx")
df["depart_date"] = pd.to_datetime(df["depart_date"], format="%Y-%m-%d")

# Encode 'destination' using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
destination_encoded = encoder.fit_transform(df[["destination"]])
destination_encoded_df = pd.DataFrame(destination_encoded, columns=encoder.get_feature_names_out(["destination"]))

# Combine encoded 'destination' with the rest of the dataset
df_preprocessed = pd.concat([df.drop(columns=["origin","destination", "extract_timestamp", "depart_date"]), destination_encoded_df], axis=1)

#Split the dataset into training and testing sets
# Memisahkan kolom dependen dan independen
X = df_preprocessed.drop(columns=["best_price"])
y = df_preprocessed['best_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Buat objek Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Buat objek KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Inisialiasi metrik evaluasi
training_accuracy = []
testing_accuracy = []
mse_list = []
rmse_list = []
mae_list = []
r2_list = []

# Loop setiap fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
  # Gunakan .iloc untuk memilih baris berdasarkan indeks integer
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  # Latih model
  model_rf.fit(X_train, y_train)

  # Menghitung akurasi untuk Data Pelatihan dan Pengujian
  train_predictions = model_rf.predict(X_train)
  test_predictions = model_rf.predict(X_test)

  # Hitung ulang prediksi dari model Random Forest
  predictions_rf = model_rf.predict(X_test) 

  # Menghitung metrik evaluasi
  mse = mean_squared_error(y_test, predictions_rf)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_test, predictions_rf)
  r2 = r2_score(y_test, predictions_rf)

   # Menghitung akurasi
  training_accuracy.append(model_rf.score(X_train, y_train))
  testing_accuracy.append(model_rf.score(X_test, y_test))

  # Menambahkan list yang ada
  mse_list.append(mse)
  rmse_list.append(rmse)
  mae_list.append(mae)
  r2_list.append(r2)

  # Membulatkan dan mengubah tipe data
  df_pred_rf = pd.DataFrame({'predictions_rf' : predictions_rf})
  df_pred_rf["predictions_rf"] = df_pred_rf["predictions_rf"].round().astype(int)

  print(f"Fold {fold+1}:")
  print(f"  Training Accuracy: {training_accuracy[-1]:.2f}")
  print(f"  Testing Accuracy: {testing_accuracy[-1]:.2f}")
  print(f"  MSE: {mse:.2f}")
  print(f"  RMSE: {rmse:.2f}")
  print(f"  MAE: {mae:.2f}")
  print(f"  R2 Score: {r2:.2f}")
  print(df_pred_rf.head())
  print()

# Fungsi untuk memproses input tujuan
def preprocess_destination(destination):
    destination_encoded = encoder.transform([[destination]])
    return pd.DataFrame(destination_encoded, columns=encoder.get_feature_names_out(["destination"]))

# Fungsi untuk memprediksi harga berdasarkan tujuan
def predict_price(destination, year, month, day):
    destination_features = preprocess_destination(destination)
    input_data = pd.DataFrame({
        "year": [year],
        "month": [month],
        "day": [day]
    })
    processed_input = pd.concat([input_data, destination_features], axis=1)
    prediction = model_rf.predict(processed_input)
    return prediction[0]

# Input dan Prediksi
def input_destination():
    destination = input("Masukkan tujuan: ")

    # Validasi input tahun
    while True:
        year_input = input("Masukkan tahun (misal, 2023): ")
        if year_input.isdigit():
            year = int(year_input)
            break
        else:
            print("Input tidak valid! Harap masukkan tahun yang valid.")

    # Validasi input bulan
    while True:
        month_input = input("Masukkan bulan (1-12): ")
        if month_input.isdigit() and 1 <= int(month_input) <= 12:
            month = int(month_input)
            break
        else:
            print("Input tidak valid! Harap masukkan bulan yang valid (1-12).")

    # Validasi input tanggal
    while True:
        day_input = input("Masukkan tanggal: ")
        if day_input.isdigit() and 1 <= int(day_input) <= 31:  # Anda bisa menambahkan validasi tanggal berdasarkan bulan
            day = int(day_input)
            break
        else:
            print("Input tidak valid! Harap masukkan tanggal yang valid (1-31).")
    
    # Memanggil fungsi prediksi
    price_prediction = predict_price(destination, year, month, day)
    return price_prediction

# Penggunaan contoh
example_destination = input("Masukkan tujuan untuk prediksi: ")
predicted_price = input_destination()
print(f"Prediksi Harga untuk tujuan {example_destination}: {predicted_price}")

# Judul Web
st.title('Prediksi Harga Tiket Pesawat')

# Input tujuan
destination = st.text_input("Masukkan Tujuan (contoh: YIA, CGK, DPS):")
year = st.text_input("Masukkan Tahun (contoh: 2023):")
month = st.text_input("Masukkan Bulan (1-12):")
day = st.text_input("Masukkan Tanggal (1-31):")

if st.button("Prediksi"):
    # Validasi input
    if not destination or not year or not month or not day:
        st.warning("Harap isi semua kolom input!")
    else:
        try:
            # Konversi input angka
            year = int(year)
            month = int(month)
            day = int(day)
            if 1 <= month <= 12 and 1 <= day <= 31:  # Validasi bulan dan tanggal
                price, error = predict_price(destination, year, month, day)
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success(f"**Prediksi Harga:** {price}")
            else:
                st.error("Bulan harus antara 1-12 dan tanggal harus antara 1-31.")
        except ValueError:
            st.error("Tahun, bulan, dan tanggal harus berupa angka yang valid.")


