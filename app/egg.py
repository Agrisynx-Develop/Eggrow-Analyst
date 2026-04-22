import streamlit as st
st.set_page_config(page_title="EggRow AI System", layout="wide")
st.title("🐔 EggRow - Smart Poultry Decision Support System")

import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import google.generativeai as genai

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime
import re
import cv2
from tensorflow.keras.models import load_model

from PIL import Image
from scipy.optimize import linprog


# =====================================================
# CONFIGURATION
# =====================================================


os.makedirs("database", exist_ok=True)
db_path = "database/produktivitias_db.csv"

# =====================================================
# GEMINI CONFIG
# =====================================================
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", "YOUR_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# =====================================================
# NILAI STANDAR GLOBAL
# =====================================================
HDP_OPTIMAL = (90, 96)
HDP_ALERT = 85
FCR_OPTIMAL = (1.9, 2.2)
FCR_ALERT = 2.3

# =====================================================
# HELPER FUNCTION
# =====================================================
def format_rupiah(value):
    return f"Rp {value:,.0f}"

def clean_numeric(x):
    x = re.sub(r"[^\d.]", "", str(x))
    return pd.to_numeric(x, errors="coerce")

# -------------------- STATEFUL BUTTON --------------------
def stateful_button(*args, key=None, **kwargs):
    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False

    if st.button(*args, **kwargs):
        st.session_state[key] = not st.session_state[key]

    return st.session_state[key]



# =====================================================
# SIDEBAR
# =====================================================
menu = st.sidebar.selectbox(
    "Menu",
    ["Dashboard", "Analisis Prediksi", "Nutrisi", "Kesehatan", "Summary"]
)


# =====================================================
# DASHBOARD
# =====================================================
if menu == "Dashboard":
    with st.sidebar:
        st.header("Import Data")

        fileMethod = st.radio(
            "Select data source:",
            options=["Browse Files", "Seaborn Dataset"]
        )

        df = None

        if fileMethod == "Browse Files":
            dataupload = st.file_uploader("Upload CSV")
            if dataupload:
                df = pd.read_csv(dataupload)
                st.dataframe(df.head())

        if fileMethod == "Seaborn Dataset":
            dataset_name = st.text_input("Enter dataset name (e.g. iris)")
            if dataset_name:
                try:
                    df = sns.load_dataset(dataset_name)
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(str(e))      
    
    
    st.header("📊 Dashboard Monitoring")

    if df is None:
        st.warning("Please upload or select a dataset")
        st.stop()

    with st.expander("View Data"):
        st.dataframe(df)
   
    tab1 = st.tabs([
    "Produktivitas"
    ])
    
    # -------------------- TAB 1 --------------------
    with tab1:
        st.header("Filter Data Berdasarkan Tanggal")

        # Pastikan kolom tanggal ada
        if "tanggal" not in df.columns:
            st.error("Kolom 'tanggal' tidak ditemukan")
            st.stop()

        # Convert ke datetime
        df['tanggal'] = pd.to_datetime(df['tanggal']).dt.strftime('%Y-%m-%d')

        filter_type = st.radio(
            "Pilih Filter Tanggal",
            ["Semua Data", "Tanggal tertentu", "Range tanggal"]
        )

        # Default
        df_filtered = df.copy()
        
        if filter_type == "Tanggal tertentu":
            selected_date = st.date_input("Pilih tanggal")
            df_filtered = df[df["tanggal"] == (selected_date)]

        elif filter_type == "Range tanggal":
            df["tanggal"] = pd.to_datetime(df["tanggal"]).dt.date
    
            # input user
            start_date = st.date_input("Start date")
            end_date = st.date_input("End date")

            # filter
            df_filtered = df[
                (df["tanggal"] >= start_date) &
                (df["tanggal"] <= end_date)
            ]
        

        # Hindari warning pandas
        df_filtered = df_filtered.copy()

        st.write("Data setelah filter:")
        st.dataframe(df_filtered)


        # ==================== HITUNG INDIKATOR ====================
        st.subheader("Hitung Indikator Produksi")

        required_cols = [
            "konsumsi pakan",
            "jumlah telur",
            "berat telur rata-rata",
            "jumlah ternak",
            "harga pakan",
            "harga telur"
        ]

        # Cek kolom wajib
        missing_cols = [col for col in required_cols if col not in df_filtered.columns]
        if missing_cols:
            st.error(f"Kolom berikut belum ada: {missing_cols}")
            st.stop()

        if st.button("Hitung Indikator"):

            # Pastikan tipe data numerik
            df_filtered[required_cols] = df_filtered[required_cols].astype(float)

            # Hindari pembagian nol
            safe_divisor = (df_filtered["jumlah telur"] * df_filtered["berat telur rata-rata"]).replace(0, np.nan)

            # Perhitungan
            df_filtered["fcr"] = df_filtered["konsumsi pakan"] / safe_divisor

            df_filtered["hdp"] = (df_filtered["jumlah telur"] / df_filtered["jumlah ternak"]) * 100
            df_filtered["hdp"] = np.where(df_filtered["hdp"] > 100, np.nan,df_filtered["hdp"])
            df_filtered["feed cost"] = (df_filtered["konsumsi pakan"] * df_filtered["harga pakan"]) / safe_divisor

            df_filtered["revenue"] = (df_filtered["jumlah telur"] * df_filtered["berat telur rata-rata"]) * df_filtered["harga telur"]

            df_filtered["total feed cost"] = df_filtered["konsumsi pakan"] * df_filtered["harga pakan"]

            df_filtered["profit"] = df_filtered["revenue"] - df_filtered["total feed cost"]

            # Rapikan angka
            df_filtered = df_filtered.round(2)

            st.success("Indikator berhasil dihitung")

            # ==================== OUTPUT ====================
            st.session_state["df_filtered"] = df_filtered

            # ================== TAMPILKAN DATA ==================
            st.dataframe(df_filtered)

            # ================== SIMPAN KE DATABASE ==================
            db_path = "database/hasil_analisis.csv"

            if st.button("💾 Simpan ke Database"):

                os.makedirs("database", exist_ok=True)

                save_df = df_filtered.copy()

                if os.path.exists(db_path):
                    old = pd.read_csv(db_path)
                    save_df = pd.concat([old, save_df], ignore_index=True)

                save_df.to_csv(db_path, index=False)

                st.success("Data berhasil disimpan")

            st.subheader("Ringkasan")
            
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("FCR", round(df_filtered['fcr'].mean(), 3))

            with col2:
                st.metric("HDP (%)", round(df_filtered['hdp'].mean(), 2))

            with col3:
                st.metric("Feed Cost / kg egg", format_rupiah(df_filtered['feed cost'].mean()))

            with col4:
                st.metric("Total Profit", format_rupiah(df_filtered['profit'].sum()))

            with col5:
                st.metric("Rata-rata Profit", format_rupiah(df_filtered['profit'].mean()))

            # ================= ALERT =================
            hdp_value = df_filtered['hdp'].mean()
            fcr_value = df_filtered['fcr'].mean()

            if hdp_value < HDP_ALERT:
                st.error("⚠️ HDP di bawah standar!")
            elif HDP_OPTIMAL[0] <= hdp_value <= HDP_OPTIMAL[1]:
                st.success("✅ HDP optimal")
            else:
                st.warning("⚠️ HDP tidak ideal")

            if fcr_value > FCR_ALERT:
                st.error("⚠️ FCR melebihi standar!")
            elif FCR_OPTIMAL[0] <= fcr_value <= FCR_OPTIMAL[1]:
                st.success("✅ FCR optimal")
            else:
                st.warning("⚠️ FCR tidak ideal")
            # ==================== GRAFIK ====================
            import plotly.express as px

            fig = px.line(df_filtered, x="tanggal", y="profit", title="Profit over Time")
            st.plotly_chart(fig)



elif menu == "Analisis Prediksi":
    with st.sidebar:
         st.header("📊 Analisis Prediksi")

    df_sidebar = st.session_state.get("df_filtered")

    if df_sidebar is not None:
        st.write("Data terbaru:")
        st.dataframe(df_sidebar)
    else:
        st.info("Belum ada data hasil analisis")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Cleaning",
        "Visualization",
        "Model Building",
        "Prediction",
        "Chatbot"
    ])


    # -------------------- TAB 1 --------------------
    with tab1:
        st.header("Data Cleaning")

        st.write("Missing Values:")
        st.write(df_sidebar.isnull().sum())

        if st.button("Drop Null Rows"):
            df_sidebar = df_sidebar.dropna()
            st.success("Null rows removed")

        if st.button("Drop Duplicates"):
            df_sidebar = df_sidebar.drop_duplicates()
            st.success("Duplicates removed")

        if st.button("Encode Categorical Columns"):
            for col in df_sidebar.select_dtypes(include='object').columns:
                le = LabelEncoder()
                df_sidebar[col] = le.fit_transform(df_sidebar[col])
            st.success("Encoding completed")

        st.dataframe(df_sidebar)


    # -------------------- TAB 2 --------------------
    with tab2:
        st.header("Visualization")

        x = st.selectbox("X Axis", df_sidebar.columns)
        y = st.selectbox("Y Axis", df_sidebar.columns)

        plot_type = st.selectbox("Plot Type", [
            "Line", "Scatter", "Bar", "Histogram", "Box"
        ])

        try:
            if plot_type == "Line":
                fig = px.line(df_sidebar, x=x, y=y)

            elif plot_type == "Scatter":
                fig = px.scatter(df_sidebar, x=x, y=y)

            elif plot_type == "Bar":
                fig = px.bar(df_sidebar, x=x, y=y)

            elif plot_type == "Histogram":
                fig = px.histogram(df_sidebar, x=x)

            elif plot_type == "Box":
                fig = px.box(df_sidebar, x=x, y=y)

            st.plotly_chart(fig)

        except Exception as e:
            st.error(str(e))


    # -------------------- TAB 3 --------------------
    with tab3:
        st.header("Model Building")

        model_name = st.selectbox("Select Model", [
            "Linear Regression",
            "Logistic Regression"
        ])

        X_cols = st.multiselect("Select Features (X)", df_sidebar.columns)
        y_col = st.selectbox("Select Target (Y)", df_sidebar.columns)

        if st.button("Train Model"):

            X = df_sidebar[X_cols]
            y = df_sidebar[y_col]

            # Check categorical
            if X.select_dtypes(include='object').shape[1] > 0:
                st.error("Encode categorical columns first")
                st.stop()

            # Select model
            if model_name == "Linear Regression":
                ml_model = LinearRegression()
            else:
                model = LogisticRegression()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            ml_model.fit(X_train, y_train)
            y_pred = ml_model.predict(X_test)
            
            st.success("Model trained successfully")
            if np.issubdtype(y_pred.dtype, np.floating):
                nilai_MSE = mean_squared_error(y_test, y_pred)
                nilai_R2 = r2_score(y_test, y_pred)
        # REGRESSION
                st.write(nilai_MSE)
                st.write(nilai_R2)
                coef_df = pd.DataFrame({
                "feature": X_cols,
                "coefficient": ml_model.coef_
                })
                st.dataframe(coef_df)
                st.session_state["nilai_MSE"] = nilai_MSE
                st.session_state["nilai_R2"] = nilai_R2
                st.session_state["coef_df"] = coef_df
            else:
        # CLASSIFICATION
                st.write("Accuracy:", accuracy_score(y_test, y_pred))
                st.dataframe(classification_report(y_test, y_pred, output_dict=True))
            # Save model
            st.session_state["model"] = ml_model
            st.session_state["features"] = X_cols


    # -------------------- TAB 5 --------------------
    with tab4:
        st.header("Prediction")

        if "model" not in st.session_state:
            st.warning("Train a model first")
            st.stop()

        ml_model = st.session_state["model"]
        features = st.session_state["features"]

        user_input = []

        for col in features:
            val = st.number_input(f"{col}")
            user_input.append(val)

        if st.button("Predict"):
            pred = ml_model.predict([user_input])
            st.success(f"Prediction: {pred}")

    with tab5:
        if st.button("Generate Jawaban AI"):

            if "nilai_MSE" not in st.session_state:
                st.error("❌ Model belum dilatih")
                st.stop()

            nilai_MSE = st.session_state.get("nilai_MSE")
            nilai_R2 = st.session_state.get("nilai_R2")
            coef_df = st.session_state.get("coef_df")
            pred = st.session_state.get("pred")

            context_dashboard = ""
            if "coef_df" in st.session_state:
                context_dashboard = st.session_state["coef_df"].to_string()
            elif "pred" in st.session_state:
                context_dashboard = st.session_state["pred"].to_string()
            elif "nilai_MSE" in st.session_state:
                context_dashboard = st.session_state["nilai_MSE"].to_string()
            elif "nilai_R2" in st.session_state:
                context_dashboard = st.session_state["nilai_R2"].to_string()
            prompt = f"""
            koefisien:
            {context_dashboard}

            hasil prediksi:
            {context_dashboard}

            MSE
            {nilai_MSE}

            R2
            {nilai_R2}

            Anda adalah AI Data Analyst profesional di bidang peternakan ayam petelur.
            ======================
            TUGAS ANDA
            ======================

            1. Jelaskan arti nilai R2 dan MSE dengan bahasa sederhana:
            - Apa arti R2 terhadap kualitas model?
            - Apakah nilai tersebut tergolong baik atau tidak?
            - Apa arti MSE terhadap error model?
            - Apakah model cukup layak digunakan?

            2. Analisis tiap variabel (koefisien):
            Untuk setiap variabel:
            - Jelaskan apakah pengaruhnya positif atau negatif
            - Jelaskan arti besarnya koefisien
            - Apakah masuk akal secara bisnis
            - Jika tidak masuk akal, jelaskan kemungkinan penyebab (misalnya korelasi, data bias, dll)

            3. Analisis kondisi bisnis saat ini:
            - Apakah performa produksi efisien atau tidak
            - Hubungkan FCR, HDP, dan profit

            4. Analisis hasil prediksi:
            - Jelaskan arti nilai prediksi
            - Jelaskan bagaimana setiap input (fitur X) mempengaruhi hasil prediksi
            - Variabel mana yang paling dominan dalam mempengaruhi hasil

            5. Rekomendasi tindakan:
            - Apa yang harus ditingkatkan?
            - Apa yang harus dikurangi?
            - Berikan saran konkret (operasional, bukan teori)

            Gunakan bahasa Indonesia yang jelas, terstruktur, dan mudah dipahami.
            Gunakan bullet point jika perlu.
            Fokus pada insight praktis, bukan hanya teori statistik.

            Pertanyaan:
            {user_input}

            Berikan analisis dan solusi profesional.
            """
            response = model.generate_content(prompt)
            st.write(response.text)

        else:
            st.warning("Database kosong.")
        


elif menu == "Nutrisi":

    import numpy as np
    from scipy.optimize import linprog

    st.header("🐔 Optimasi Ransum Ayam Petelur (Biaya + Nutrisi)")

    # ==============================
    # DATA
    # ==============================
    df_pakan = pd.DataFrame({
        "Nama Pakan": [
            "Jagung Kuning", "Dedak Padi", "Bungkil Kedelai", "Tepung Ikan",
            "Pollard Gandum", "Minyak Sawit", "DDGS",
            "Tepung Daging & Tulang", "Kapur", "DCP"
        ],
        "Protein (%)": [8.5,12,44,60,16,0,27,50,0,0],
        "Energi (kkal)": [3300,1800,2500,2800,2000,8800,2800,2200,0,0],
        "Ca (%)": [0.02,0.1,0.3,5.5,0.15,0,0.1,10,38,23],
        "P (%)": [0.3,1.5,0.6,3.0,1.0,0,0.8,5.0,0,18],
        "Lisin (%)": [0.26,0.5,2.8,4.5,0.6,0,0.9,2.5,0,0],
        "Metionin (%)": [0.18,0.3,0.6,1.5,0.25,0,0.5,0.7,0,0],
        "Harga (Rp/satuan)": [5000,4000,9000,12000,4500,14000,7000,8000,1500,10000]
    })

    # ==============================
    # PILIH FASE
    # ==============================
    fase = st.selectbox("Fase Produksi", [
        "Layer Awal (18-32 minggu)",
        "Peak Production (32-50 minggu)",
        "Layer Tua (>50 minggu)"
    ])

    kebutuhan = {
        "Layer Awal (18-32 minggu)": {"protein":19.0, "energi":2800, "ca":4.0, "p":0.45, "lisin":1.00, "metionin":0.50},
        "Peak Production (32-50 minggu)": {"protein":17.5, "energi":2800, "ca":4.2, "p":0.40, "lisin":0.93, "metionin":0.46},
        "Layer Tua (>50 minggu)": {"protein":16.0, "energi":2700, "ca":4.5, "p":0.35, "lisin":0.85, "metionin":0.40},
    }

    target = kebutuhan[fase]

    # ==============================
    # WEIGHT MULTI OBJECTIVE
    # ==============================
    alpha = st.slider("Fokus Biaya (0.5=seimbang, 1=hemat)", 0.1, 1.0, 0.7)

    st.write("Semakin tinggi → lebih hemat, semakin rendah → nutrisi maksimal")

    # ==============================
    # DATA ARRAY
    # ==============================
    harga = df_pakan["Harga (Rp/satuan)"].values
    protein = df_pakan["Protein (%)"].values
    energi = df_pakan["Energi (kkal)"].values
    ca = df_pakan["Ca (%)"].values
    p = df_pakan["P (%)"].values
    lisin = df_pakan["Lisin (%)"].values
    metionin = df_pakan["Metionin (%)"].values

    # ==============================
    # NORMALISASI (AGAR SEIMBANG)
    # ==============================
    nutrisi_score = (
        protein/target["protein"] +
        energi/target["energi"] +
        ca/target["ca"] +
        p/target["p"] +
        lisin/target["lisin"] +
        metionin/target["metionin"]
    )

    # ==============================
    # OBJECTIVE (BIAYA - NUTRISI)
    # ==============================
    c = alpha * harga - (1 - alpha) * nutrisi_score * 1000

    # ==============================
    # CONSTRAINT
    # ==============================
    
    A = []
    b = []

    A.append(-protein / 100)
    b.append(-target["protein"])

    A.append(-energi / 100)
    b.append(-target["energi"])

    A.append(-ca / 100)
    b.append(-target["ca"])

    A.append(-p / 100)
    b.append(-target["p"])

    A.append(-lisin / 100)
    b.append(-target["lisin"])

    A.append(-metionin / 100)
    b.append(-target["metionin"])

    # total = 100%
    A_eq = [np.ones(len(harga))]
    b_eq = [100]


    # ==============================
    # SOLVE
    # ==============================
    result = linprog(
        c,
        A_ub=A,
        b_ub=b,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, None) for _ in harga],
        method='highs'
    )

    # ==============================
    # OUTPUT
    # ==============================
    if result.success:

        df_pakan["Komposisi (%)"] = np.round(result.x, 2)

        st.subheader("📊 Komposisi Ransum")
        st.dataframe(df_pakan)

        def hitung(col):
            return np.sum(result.x * df_pakan[col]) / 100

        st.subheader("📈 Nutrisi")

        st.metric("Protein", round(hitung("Protein (%)"),2))
        st.metric("Energi", round(hitung("Energi (kkal)"),0))
        st.metric("Ca", round(hitung("Ca (%)"),2))

        biaya = np.sum(result.x * harga) / 100
        st.success(f"💰 Biaya: Rp {round(biaya,0)}/kg")

        st.bar_chart(df_pakan.set_index("Nama Pakan")["Komposisi (%)"])

    else:
        st.error("❌ Tidak ditemukan solusi")
        
elif menu == "Kesehatan":

    st.header("🩺 Sistem Kesehatan Ayam (AI + Expert System)")

    tab1, tab2, tab3 = st.tabs([
        "CF Method",
        "Eggrow Vision",
        "Combine"
    ])

    # ==============================
    # LOAD DATA (GLOBAL)
    # ==============================
    @st.cache_data
    def load_data():
        import os
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(BASE_DIR, "..", "Data")
    
        df_gejala = pd.read_csv(os.path.join(data_path, "gejala_cf.csv"))
        df_tanya = pd.read_csv(os.path.join(data_path, "pertanyaan.csv"))
        df_penyakit = pd.read_csv(os.path.join(data_path, "relasi_penyakit.csv"))
    
        return df_gejala, df_tanya, df_penyakit

    df_gejala, df_tanya, df_penyakit = load_data()
    
    df = df_gejala.merge(df_tanya, on="kode_pertanyaan")

    # =========================
    # CF MAP
    # =========================
    cf_map = {
        "Tidak Ada": -0.8,
        "Kemungkinan Kecil": 0.3,
        "Kemungkinan Besar": 0.6,
        "Ada": 0.9
    }

    # =========================
    # FUNCTION CF
    # =========================
    def combine_cf(cf_list):
        if not cf_list:
            return 0

        result = cf_list[0]
        for cf in cf_list[1:]:
            result = result + cf * (1 - result)

        return result

    # =========================
    # TAB 1 → CF METHOD
    # =========================
    with tab1:
        st.header("CF Method")

        st.subheader("📝 Jawab Pertanyaan")

        jawaban_user = {}

        for i, row in df.iterrows():
            jawaban = st.selectbox(
                row["pertanyaan"],
                list(cf_map.keys()),
                key=f"cf_{i}"
            )
            jawaban_user[row["kode_pertanyaan"]] = cf_map[jawaban]

        # STEP 1
        df["cf_evidence"] = df["kode_pertanyaan"].map(jawaban_user)
        df["cf_gejala"] = df["cf_evidence"] * df["cf_rule"]

        hasil = []

        if st.button("🔍 Diagnosa Sekarang"):

            hasil = []  # pastikan ini ada

            for kode in df_penyakit["kode_penyakit"].unique():

                subset = df_penyakit[df_penyakit["kode_penyakit"] == kode]

                cf_list = []

                for _, row in subset.iterrows():
                    kode_gejala = row["kode_gejala"]
                    cf_pakar = row["cf_pakar"]

                    cf_gejala = df[df["kode_gejala"] == kode_gejala]["cf_gejala"]

                    if not cf_gejala.empty:
                        cf_penyakit = cf_gejala.values[0] * cf_pakar
                        cf_list.append(cf_penyakit)

                # =========================
                # KOMBINASI CF
                # =========================
                cf_final = combine_cf(cf_list)

                # 🔥 PERBAIKAN: NEGATIF → 0
                cf_final = max(0, cf_final)

                nama = subset["nama_penyakit"].values[0]

                hasil.append({
                    "penyakit": nama,
                    "cf": cf_final
                })

            # =========================
            # SORTING
            # =========================
            hasil = sorted(hasil, key=lambda x: x["cf"], reverse=True)

            # =========================
            # OUTPUT
            # =========================
            st.subheader("📊 Hasil Diagnosa")

            for h in hasil:
                persen = max(0, h["cf"] * 100)  # 🔥 aman dari negatif
                st.write(f"**{h['penyakit']}** → {persen:.2f}%")

            terbaik = hasil[0]

            persen_terbaik = max(0, terbaik["cf"] * 100)

            st.success(f"""
            🐔 Kemungkinan terbesar:
            **{terbaik['penyakit']}**
            ({persen_terbaik:.2f}%)
            """)

            # =========================
            # PROGRESS BAR AMAN
            # =========================
            progress_val = int(max(0, min(100, persen_terbaik)))
            st.progress(progress_val)
            # =========================
            # AI EXPLANATION
            # =========================
            prompt = f"""
            Penyakit: {terbaik['penyakit']}
            Tingkat keyakinan: {terbaik['cf']*100:.2f}%

            Jelaskan:
            - Penyebab
            - Dampak produksi
            - Tindakan cepat
            - Pencegahan
            """

            try:
                res = model.generate_content(prompt)
                st.subheader("🤖 Insight AI")
                st.write(res.text)
            except:
                st.error("AI tidak tersedia")

       @st.cache_resource
        def load_data_dl():
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
            model_path = os.path.join(BASE_DIR, "..", "model", "eggrow_vision_model.h5")
            class_path = os.path.join(BASE_DIR, "..", "model", "labels.npy")
        
            # DEBUG (biar tahu errornya dimana kalau gagal)
            st.write("📂 MODEL PATH:", model_path)
            st.write("📂 EXISTS:", os.path.exists(model_path))
            st.write("📂 FILES:", os.listdir(os.path.join(BASE_DIR, "..", "model")))
        
            model = tf.keras.models.load_model(model_path, compile=False)
            classes = np.load(class_path)
        
            return model, classes
        
        
        model_dl, class_dl = load_data_dl()          
        # =========================
        # UI
        # =========================
        with tab2:
            st.header("📷 Eggrow Vision (Deep Learning)")
    
            uploaded_img = st.file_uploader("Upload gambar ayam", type=["jpg","png"])
            
        
            if uploaded_img:
                file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
    
                img_resized = cv2.resize(img, (128, 128))
                img_norm = img_resized / 255.0
                img_input = np.expand_dims(img_norm, axis=0)
    
                st.image(img_resized, channels="BGR")
    
                if st.button("🔍 Analisis AI Vision"):
            
                    pred = model_dl.predict(img_input)
                    idx = np.argmax(pred)
                    confidence = float(np.max(pred))
    
                    penyakit = class_names[idx]
    
                    st.success(f"""
                    🐔 Prediksi:
                    **{penyakit}**
                    Confidence: {confidence*100:.2f}%
                    """)


    # =========================
    # TAB 3 → COMBINE
    # =========================
    with tab3:
        st.header("Combine Analysis")

        if st.button("🚀 Analisis Menyeluruh"):
            st.info("Gunakan hasil CF + gambar untuk analisis gabungan (custom logic bisa ditambahkan)")

elif menu == "Summary":

    st.header("📋 Executive AI Summary")

    # =========================
    # AMBIL DATA DARI DASHBOARD
    # =========================

    df = st.session_state.get("df_filtered")

    if df is not None:
        st.write("Data terbaru:")
        st.dataframe(df)
    else:
        st.info("Belum ada data hasil analisis")
        st.stop()

    # =========================
    # HITUNG KPI
    # =========================
    avg_fcr = df["fcr"].mean()
    avg_hdp = df["hdp"].mean()
    total_profit = df["profit"].sum()
    avg_profit = df["profit"].mean()

    # =========================
    # TAMPILKAN METRIC
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("FCR", round(avg_fcr, 3))
    col2.metric("HDP (%)", round(avg_hdp, 2))
    col3.metric("Total Profit", format_rupiah(total_profit))
    col4.metric("Avg Profit", format_rupiah(avg_profit))

    # =========================
    # ANALISIS LOGIKA (IMPORTANT)
    # =========================
    analisis = []

    if avg_hdp < HDP_ALERT:
        analisis.append("Produksi telur rendah (HDP di bawah standar).")

    if avg_fcr > FCR_ALERT:
        analisis.append("Efisiensi pakan buruk (FCR tinggi).")

    if total_profit < 0:
        analisis.append("Usaha mengalami kerugian.")

    if not analisis:
        analisis.append("Performa produksi relatif stabil.")

    st.subheader("🧠 Analisis Sistem")
    for a in analisis:
        st.write(f"- {a}")

    # =========================
    # HUBUNGAN DENGAN PAKAN
    # =========================
    st.subheader("🔗 Insight Nutrisi & Produksi")

    insight = []

    if avg_fcr > 2.2:
        insight.append("Kemungkinan pakan kurang efisien (energi/protein tidak optimal).")

    if avg_hdp < 90:
        insight.append("Produksi telur bisa dipengaruhi oleh defisiensi nutrisi atau kesehatan ayam.")

    if total_profit < 0:
        insight.append("Biaya pakan terlalu tinggi dibanding hasil produksi.")

    for i in insight:
        st.write(f"- {i}")

    # =========================
    # AI REPORT
    # =========================
    if st.button("🤖 Generate Executive Report"):

        prompt = f"""
        Anda adalah AI Expert di bidang peternakan ayam petelur.

        DATA:
        - FCR: {avg_fcr}
        - HDP: {avg_hdp}
        - Total Profit: {format_rupiah(total_profit)}

        STANDAR:
        - HDP optimal: {HDP_OPTIMAL}
        - FCR optimal: {FCR_OPTIMAL}

        ANALISIS AWAL:
        {analisis}

        TUGAS:
        1. Evaluasi performa produksi
        2. Jelaskan penyebab utama
        3. Hubungkan dengan kualitas pakan
        4. Berikan rekomendasi konkret:
           - perbaikan ransum
           - efisiensi biaya
           - peningkatan produksi
        5. Gunakan bahasa profesional & terstruktur

        Tanpa identitas.
        """

        response = model.generate_content(prompt)
        report = response.text

        st.subheader("📄 Executive Report")
        st.write(report)

        def markdown_to_html(text):
                # Convert **bold** menjadi <b>bold</b>
                text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
                return text
            
        def add_footer(canvas, doc):
                canvas.saveState()
                footer_text = f"EggRow AI System | Generated: {datetime.now().strftime('%d-%m-%Y')} | Page {doc.page}"
                canvas.setFont("Helvetica", 8)
                canvas.setFillColor(colors.grey)
                canvas.drawCentredString(A4[0] / 2, 0.5 * inch, footer_text)
                canvas.restoreState()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=40,
                leftMargin=40,
                topMargin=60,
                bottomMargin=60
            )

        elements = []
        styles = getSampleStyleSheet()

            # ===== CUSTOM STYLES =====

        title_style = ParagraphStyle(
                name="TitleStyle",
                parent=styles["Title"],
                alignment=TA_CENTER,
                fontSize=18,
                spaceAfter=20
            )

        heading_style = ParagraphStyle(
                name="HeadingStyle",
                parent=styles["Heading2"],
                fontSize=14,
                spaceBefore=12,
                spaceAfter=6
            )

        subheading_style = ParagraphStyle(
                name="SubHeadingStyle",
                parent=styles["Heading3"],
                fontSize=12,
                spaceBefore=8,
                spaceAfter=4
            )

        body_style = ParagraphStyle(
                name="BodyStyle",
                parent=styles["Normal"],
                alignment=TA_JUSTIFY,
                fontSize=11,
                leading=16,
                spaceAfter=8
            )

            # ===== TITLE =====

        elements.append(Paragraph("LAPORAN EXECUTIVE AYAM LAYER", title_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        elements.append(Spacer(1, 0.3 * inch))

            # ===== FORMAT TEKS OTOMATIS =====

        lines = report.split("\n")

        for line in lines:
                line = line.strip()

                if not line:
                    continue

                # 1️⃣ Convert semua markdown bold terlebih dahulu
                line = markdown_to_html(line)

                # =============================
                # BULLET LIST
                # =============================
                if line.startswith("•") or line.startswith("*"):

                    # Hapus simbol bullet asli di depan
                    clean_line = line.lstrip("*• ").strip()

                    elements.append(Paragraph(f"• {clean_line}", body_style))

                # =============================
                # NUMBERED LIST (1. 2. 3.)
                # =============================
                elif line[0].isdigit() and "." in line:
                    elements.append(Paragraph(f"<b>{line}</b>", subheading_style))

                # =============================
                # HEADING (jika satu baris full bold)
                # =============================
                elif line.startswith("<b>") and line.endswith("</b>"):
                    elements.append(Paragraph(line, heading_style))

                # =============================
                # NORMAL TEXT
                # =============================
                else:
                    elements.append(Paragraph(line, body_style))

                elements.append(Spacer(1, 0.1 * inch))

            # ===== BUILD DOCUMENT =====

        doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)

        st.download_button(
                "📥 Download PDF",
                data=buffer.getvalue(),
                file_name="laporan_ayam_layer_profesional.pdf",
                mime="application/pdf"
            )

    else:
        st.warning("Database kosong.")
