import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -- 1. LOAD ASSETS --
def load_assets():
    with open('model_xgb.pkl', 'rb') as f:
        m_xgb = pickle.load(f)
    with open('model_rf.pkl', 'rb') as f:
        m_rf = pickle.load(f)
    with open('model_knn.pkl', 'rb') as f:
        m_knn = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    with open('assets.pkl', 'rb') as f:
        asst = pickle.load(f)
    return m_xgb, m_rf, m_knn, sc, asst

try:
    model_xgb, model_rf, model_knn, scaler, assets = load_assets()
    encoders = assets['encoders']
    features = assets['features']
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# -- 2. TAMPILAN UI --
st.set_page_config(page_title="Diamond Price Predictor", layout="wide")
st.title("Prediksi Harga Berlian 💎")

# Sidebar Konfigurasi
selected_algo = st.sidebar.selectbox("Pilih Algoritma ML:", ["XGBoost", "Random Forest", "KNN"])

# Data Akurasi (Silakan sesuaikan dengan hasil training kamu)
accuracy_stats = {
    "XGBoost": {"R2": "98.2%", "MAE": "$350"},
    "Random Forest": {"R2": "97.5%", "MAE": "$410"},
    "KNN": {"R2": "94.1%", "MAE": "$620"}
}

# -- 3. INPUT FORM (Sesuai Dataset Kamu) --
with st.form("input_diamond"):
    st.subheader("📋 Karakteristik Berlian")
    col1, col2 = st.columns(2)
    
    with col1:
        carat = st.number_input("Carat Weight", min_value=0.1, max_value=5.0, value=0.23, step=0.01)
        
        # Pilihan Manual agar PASTI muncul Teks sesuai dataset
        cut_opt = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
        color_opt = ["J", "I", "H", "G", "F", "E", "D"]
        clarity_opt = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
        
        cut = st.selectbox("Cut Quality", options=cut_opt)
        color = st.selectbox("Color Grade", options=color_opt)
        clarity = st.selectbox("Clarity Grade", options=clarity_opt)
        
    with col2:
        depth = st.number_input("Total Depth %", min_value=40.0, max_value=80.0, value=61.5)
        table = st.number_input("Table Width %", min_value=40.0, max_value=90.0, value=55.0)
        x = st.number_input("Length (X) mm", value=3.95)
        y = st.number_input("Width (Y) mm", value=3.98)
        z = st.number_input("Height (Z) mm", value=2.43)
        
    predict_btn = st.form_submit_button("💰 Prediksi Harga Sekarang")

# -- 4. PROSES PREDIKSI --
if predict_btn:
    # Mengubah teks pilihan menjadi angka menggunakan LabelEncoder yang sudah di-load
    # Jika LabelEncoder gagal, ini akan otomatis menggunakan urutan alfabet
    try:
        c_encoded = encoders['cut'].transform([cut])[0]
        col_encoded = encoders['color'].transform([color])[0]
        cla_encoded = encoders['clarity'].transform([clarity])[0]
    except:
        # Fallback jika encoder bermasalah
        c_encoded = cut_opt.index(cut)
        col_encoded = color_opt.index(color)
        cla_encoded = clarity_opt.index(clarity)

    input_df = pd.DataFrame([[
        carat, c_encoded, col_encoded, cla_encoded, depth, table, x, y, z
    ]], columns=features)
    
    # Pilih Model
    if selected_algo == "XGBoost":
        res = model_xgb.predict(input_df)[0]
    elif selected_algo == "Random Forest":
        res = model_rf.predict(input_df)[0]
    else:
        res = model_knn.predict(scaler.transform(input_df))[0]

    # Hasil
    st.balloons()
    st.success(f"### Estimasi Harga: ${res:,.2f}")
    
    # Metrik Akurasi
    m1, m2 = st.columns(2)
    m1.metric("Akurasi Model", accuracy_stats[selected_algo]["R2"])
    m2.metric("Rata-rata Error (MAE)", accuracy_stats[selected_algo]["MAE"])

    # Feature Importance
    if selected_algo in ["XGBoost", "Random Forest"]:
        st.divider()
        st.subheader(f"📊 Fitur Paling Berpengaruh ({selected_algo})")
        imp = model_xgb.feature_importances_ if selected_algo == "XGBoost" else model_rf.feature_importances_
        feat_df = pd.DataFrame({'Fitur': features, 'Skor': imp}).sort_values(by='Skor', ascending=True)
        st.bar_chart(data=feat_df, x='Fitur', y='Skor', horizontal=True)
