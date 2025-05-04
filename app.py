import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd

# Memuat model dan scaler yang telah disimpan
model_classification = load_model('model_classification.h5')  # Model Klasifikasi
model_estimation_ann = load_model('model_estimasi.h5')  # Model Estimasi ANN
model_estimation_rnn = load_model('model_rnn.h5')  # Model Estimasi RNN

# Memuat scaler yang telah disimpan
scaler_classification = joblib.load('scaler_classification.save')
scaler_estimation = joblib.load('scaler_estimasi.save')
scaler_rnn = joblib.load('scaler_rnn.save')

# Judul aplikasi
st.title('Prediksi Lalu Lintas menggunakan Model ANN & RNN')

# Deskripsi aplikasi
st.write("""
Aplikasi ini memungkinkan Anda untuk memprediksi **Traffic Situation** dan **Total kendaraan**.
Pilih jenis prediksi yang ingin dilakukan:
""")

# Pilih jenis prediksi
prediction_type = st.selectbox('Pilih Tipe Prediksi', ['Klasifikasi (Traffic Situation)', 'Estimasi (Total Kendaraan) - ANN', 'Estimasi (Total Kendaraan) - RNN'])

# Input data untuk prediksi
if prediction_type == 'Klasifikasi (Traffic Situation)':
    st.subheader("Masukkan Data Lalu Lintas untuk Klasifikasi")
    car_count = st.number_input("Jumlah Mobil", min_value=0, step=1)
    bike_count = st.number_input("Jumlah Sepeda Motor", min_value=0, step=1)
    bus_count = st.number_input("Jumlah Bus", min_value=0, step=1)
    truck_count = st.number_input("Jumlah Truk", min_value=0, step=1)
    
    # Prediksi dengan model klasifikasi
    if st.button('Prediksi'):
        input_data = np.array([[car_count, bike_count, bus_count, truck_count]])
        input_scaled = scaler_classification.transform(input_data)
        prediction = model_classification.predict(input_scaled)
        traffic_situation = np.argmax(prediction, axis=1)
        
        # Mapping label ke Traffic Situation
        traffic_situation_map = {0: 'Heavy', 1: 'High', 2: 'Normal', 3: 'Low'}
        st.write(f"Prediksi Situasi Lalu Lintas: {traffic_situation_map[traffic_situation[0]]}")

elif prediction_type == 'Estimasi (Total Kendaraan) - ANN':
    st.subheader("Masukkan Data Lalu Lintas untuk Estimasi Total Kendaraan (ANN)")
    car_count = st.number_input("Jumlah Mobil", min_value=0, step=1)
    bike_count = st.number_input("Jumlah Sepeda Motor", min_value=0, step=1)
    bus_count = st.number_input("Jumlah Bus", min_value=0, step=1)
    truck_count = st.number_input("Jumlah Truk", min_value=0, step=1)
    
    # Prediksi dengan model estimasi ANN
    if st.button('Prediksi'):
        input_data = np.array([[car_count, bike_count, bus_count, truck_count]])
        input_scaled = scaler_estimation.transform(input_data)
        prediction = model_estimation_ann.predict(input_scaled)
        st.write(f"Prediksi Total Kendaraan: {prediction[0][0]:.2f}")

elif prediction_type == 'Estimasi (Total Kendaraan) - RNN':
    st.subheader("Masukkan Data Lalu Lintas untuk Estimasi Total Kendaraan (RNN)")
    car_count = st.number_input("Jumlah Mobil", min_value=0, step=1)
    bike_count = st.number_input("Jumlah Sepeda Motor", min_value=0, step=1)
    bus_count = st.number_input("Jumlah Bus", min_value=0, step=1)
    truck_count = st.number_input("Jumlah Truk", min_value=0, step=1)
    
    # Prediksi dengan model estimasi RNN
    if st.button('Prediksi'):
        # Menggabungkan input untuk menjadi array
        input_data = np.array([[car_count, bike_count, bus_count, truck_count]])
        
        # Skalakan input menggunakan scaler yang telah dilatih
        input_scaled = scaler_rnn.transform(input_data)

        # Ulangi data untuk membentuk window size 12
        # Pada contoh ini, kita ulangi data untuk membentuk sekuens waktu 12 langkah
        input_data_reshaped = np.repeat(input_scaled, 12, axis=0)  # Membentuk window size 12 dengan mengulang data

        # Mengubah bentuk input menjadi (1, 12, 4) sesuai dengan yang diinginkan oleh model RNN
        input_data_reshaped = input_data_reshaped.reshape(1, 12, 4)

        # Melakukan prediksi dengan model RNN
        prediction = model_estimation_rnn.predict(input_data_reshaped)
        
        # Menampilkan hasil prediksi
        st.write(f"Prediksi Total Kendaraan (RNN): {prediction[0][0]:.2f}")

