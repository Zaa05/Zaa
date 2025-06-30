import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st
import os

# --- Bagian Aplikasi Streamlit untuk Prediksi ---

st.set_page_config(page_title="Prediksi Diabetes", layout="centered")

st.title('Prediksi Diabetes')
st.markdown('Aplikasi ini memprediksi kemungkinan seseorang menderita diabetes berdasarkan beberapa parameter kesehatan.')

# Muat model dan scaler yang sudah dilatih
# Model akan digunakan untuk prediksi, scaler akan melakukan skala fitur input
@st.cache_resource # Cache resource untuk menghindari pemuatan ulang setiap kali ada interaksi
def load_model_and_scaler():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("File 'model.pkl' atau 'scaler.pkl' tidak ditemukan.")
        st.error("Pastikan Anda telah melatih model dan menyimpan file-file ini di direktori yang sama dengan 'app.py'.")
        st.stop() # Hentikan aplikasi jika file penting tidak ada

model, scaler = load_model_and_scaler()

st.sidebar.header('Parameter Input Pasien')

# Fungsi untuk mendapatkan input pengguna melalui slider
def user_input_features():
    pregnancies = st.sidebar.slider('Jumlah Kehamilan', 0, 17, 3)
    glucose = st.sidebar.slider('Kadar Glukosa (mg/dL)', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Tekanan Darah (mmHg)', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Ketebalan Kulit (mm)', 0, 99, 23)
    insulin = st.sidebar.slider('Kadar Insulin (muU/ml)', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('Fungsi Silsilah Diabetes', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Usia', 21, 81, 29)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    # Konversi input menjadi DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Dapatkan input dari pengguna
input_df = user_input_features()

st.subheader('Input Pengguna')
st.write(input_df)

# Pra-pemrosesan input pengguna dengan scaler yang sama
# Pastikan urutan kolom sesuai dengan yang digunakan saat pelatihan
numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
input_scaled = scaler.transform(input_df[numerical_cols])

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Hasil Prediksi')

    # Tampilkan hasil prediksi
    if prediction[0] == 0:
        st.success('**Pasien DIPREDIKSI TIDAK MENDERITA DIABETES.**')
    else:
        st.error('**Pasien DIPREDIKSI MENDERITA DIABETES.**')

    st.write(f"Probabilitas Tidak Diabetes (0): {prediction_proba[0][0]:.2f}")
    st.write(f"Probabilitas Diabetes (1): {prediction_proba[0][1]:.2f}")

st.markdown("""
---
**Catatan:**
* Model ini dilatih menggunakan algoritma Random Forest.
* Pra-pemrosesan data (scaling) diterapkan pada fitur input untuk memastikan konsistensi dengan data pelatihan.
* Gunakan slider di sidebar untuk memasukkan nilai parameter pasien.
""")
