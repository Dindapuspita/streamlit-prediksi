import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.title("Prediksi Harga Tiket Pesawat")

# Load model
try:
    prediksi_rf = joblib.load('rf.joblib')
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# Load dataset untuk referensi encoding
try:
    df = pd.read_excel("D:/SEMESTER7/harga_tiket.xlsx")
    df["depart_date"] = pd.to_datetime(df["depart_date"], format="%Y-%m-%d")
    df["year"] = df["depart_date"].dt.year
    df["month"] = df["depart_date"].dt.month
    df["day"] = df["depart_date"].dt.day

    # Membuat daftar unik dari destinasi
    destination_list = sorted(df["destination"].unique().tolist())  # Urutkan agar konsisten

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
    st.stop()

# Input dari pengguna
destination = st.selectbox("Pilih Kota Tujuan", destination_list)
year = st.selectbox('Pilih Tahun Keberangkatan Anda', ['2023', '2024'])
month = st.number_input("Masukkan Bulan Yang Anda Inginkan", min_value=1, max_value=12, step=1)
day = st.number_input('Masukkan Tanggal Keberangkatan', min_value=1, max_value=31, step=1)

# Proses prediksi
if st.button('Predict Price'):
    try:
        # One-Hot Encoding manual untuk 'destination'
        destination_encoded = [1 if dest == destination else 0 for dest in destination_list]

        # Menyusun input data untuk model
        input_data = np.array([int(year), int(month), int(day)] + destination_encoded, dtype=float)
        input_data = input_data.reshape(1, -1)

        # **Cek Data Sebelum Prediksi**
        st.subheader("Data yang Dikirim ke Model:")
        df_input = pd.DataFrame(input_data, columns=["year", "month", "day"] + destination_list)
        st.write(df_input)  # Menampilkan tabel input data

        # Prediksi harga tiket
        prediction = prediksi_rf.predict(input_data)
        st.success(f"Prediksi harga tiket untuk {destination} adalah: Rp {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
