import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -- 1. FUNGSI UNTUK LOAD SEMUA ASSET --
def load_assets():
    # Load ketiga model
    with open('model_xgb.pkl', 'rb') as f:
        m_xgb = pickle.load(f)
    with open('model_rf.pkl', 'rb') as f:
        m_rf = pickle.load(f)
    with open('model_knn.pkl', 'rb') as f:
        m_knn = pickle.load(f)
    
    # Load Scaler (Penting untuk KNN)
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
        
    # Load Encoders dan Nama Fitur
    with open('assets.pkl', 'rb') as f:
        asst = pickle.load(f)
        
    return m_xgb, m_rf, m_knn, sc, asst

# Panggil fungsi load
try:
    model_xgb, model_rf, model_knn, scaler, assets = load_assets()
    encoders = assets['encoders']
    features = assets['features']
except FileNotFoundError as e:
    st.error(f"File model tidak ditemukan! Pastikan semua file .pkl sudah di-upload. Error: {e}")

# -- 2. UI STREAMLIT --
st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")

st.title("💎 Diamond Price Prediction App")
st.markdown("Aplikasi ini memprediksi harga berlian menggunakan 3 algoritma berbeda berdasarkan hasil eksperimen.")

# Sidebar untuk memilih Algoritma
st.sidebar.header("Konfigurasi Model")
selected_algo = st.sidebar.selectbox(
    "Pilih Algoritma ML:", 
    ["XGBoost", "Random Forest", "KNN"]
)

st.sidebar.info(f"Model aktif: **{selected_algo}**")

# -- 3. FORM INPUT DATA --
with st.form("input_diamond"):
    st.subheader("Karakteristik Berlian")
    col1, col2 = st.columns(2)
    
    with col1:
        carat = st.number_input("Carat Weight", min_value=0.1, max_value=5.0, value=0.7, step=0.01)
        cut = st.selectbox("Cut Quality", encoders['cut'].classes_)
        color = st.selectbox("Color Grade", encoders['color'].classes_)
        clarity = st.selectbox("Clarity Grade", encoders['clarity'].classes_)
        
    with col2:
        depth = st.number_input("Total Depth %", min_value=40.0, max_value=80.0, value=61.0)
        table = st.number_input("Table Width %", min_value=40.0, max_value=90.0, value=57.0)
        x = st.number_input("Length (X) mm", min_value=0.1, max_value=12.0, value=5.5)
        y = st.number_input("Width (Y) mm", min_value=0.1, max_value=12.0, value=5.5)
        z = st.number_input("Height (Z) mm", min_value=0.1, max_value=10.0, value=3.5)
        
    predict_btn = st.form_submit_button("Prediksi Harga Sekarang")

# -- 4. PROSES PREDIKSI --
if predict_btn:
    # Buat DataFrame dari input user
    input_data = pd.DataFrame([[
        carat,
        encoders['cut'].transform([cut])[0],
        encoders['color'].transform([color])[0],
        encoders['clarity'].transform([clarity])[0],
        depth, table, x, y, z
    ]], columns=features)
    
    # Pilih model dan tentukan apakah perlu scaling
    if selected_algo == "XGBoost":
        res = model_xgb.predict(input_data)[0]
    elif selected_algo == "Random Forest":
        res = model_rf.predict(input_data)[0]
    elif selected_algo == "KNN":
        # KNN WAJIB di-scaling sesuai training
        scaled_data = scaler.transform(input_data)
        res = model_knn.predict(scaled_data)[0]
    
    # Tampilkan hasil
    st.balloons()
    st.success(f"### Estimasi Harga: ${res:,.2f}")
    
    # Informasi tambahan
    st.write(f"Prediksi dihasilkan menggunakan algoritma **{selected_algo}**.")