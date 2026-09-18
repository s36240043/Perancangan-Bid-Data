import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats

st.set_page_config(page_title="Medical Cost - Linear Regression", layout="wide")
st.title("🏥 Prediksi Biaya Asuransi Kesehatan")
st.write(
    "Replikasi notebook Kaggle *Linear Regression Tutorial* (dataset Medical Cost "
    "Personal / insurance.csv), divisualisasikan interaktif pakai Streamlit."
)

uploaded_file = st.file_uploader("Upload file insurance.csv", type=["csv"])

if uploaded_file is None:
    st.info(
        "Belum ada file. Download dataset **'Medical Cost Personal Datasets'** "
        "(insurance.csv) dari Kaggle, lalu upload di sini buat mulai."
    )
    st.stop()

df = pd.read_csv(uploaded_file)

tab1, tab2, tab3 = st.tabs(
    ["Exploratory Data Analysis", "🔧 Preprocessing & Model", "📈 Evaluasi & Validasi"]
)

# ---------------------- TAB 1: EDA ----------------------
with tab1:
    st.subheader("Preview Dataset")
    st.write(df.head())
    st.caption(f"Jumlah baris & kolom: {df.shape}")

    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

    st.subheader("Cek Missing Value")
    st.write(df.isnull().sum())

    st.subheader("Korelasi Antar Variabel Numerik")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), cmap="Wistia", annot=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribusi Charges")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df["charges"], bins=50, color="r", kde=True, ax=ax)
        ax.set_title("Distribusi Charges (Asli, right-skewed)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(np.log10(df["charges"]), bins=40, color="b", kde=True, ax=ax)
        ax.set_title("Distribusi Charges (Log Scale, mendekati normal)")
        st.pyplot(fig)

    st.subheader("Charges vs Sex & Smoker")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.violinplot(x="sex", y="charges", data=df, palette="Wistia", ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.violinplot(x="smoker", y="charges", data=df, palette="magma", ax=ax)
        st.pyplot(fig)

    st.subheader("Charges vs Age & BMI (diwarnai status smoker)")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.scatterplot(x="age", y="charges", hue="smoker", data=df, ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df, ax=ax)
        st.pyplot(fig)

# ---------------------- TAB 2: Preprocessing & Model ----------------------
with tab2:
    st.subheader("1. Encoding Variabel Kategorikal")
    st.write(
        "Kolom `sex`, `children`, `smoker`, `region` diubah jadi dummy variable "
        "(One-Hot Encoding) pakai `pd.get_dummies`, dengan `drop_first=True` "
        "buat menghindari dummy variable trap."
    )
    categorical_columns = ["sex", "children", "smoker", "region"]
    df_encode = pd.get_dummies(
        df, prefix="OHE", prefix_sep="_", columns=categorical_columns, drop_first=True
    )
    st.write(df_encode.head())

    st.subheader("2. Log Transform Target (charges)")
    st.write(
        "Distribusi `charges` asli right-skewed, jadi ditransformasi pakai "
        "natural log (`np.log`) biar mendekati distribusi normal."
    )
    df_encode["charges"] = np.log(df_encode["charges"])

    st.subheader("3. Train-Test Split")
    test_size = st.slider("Persentase data testing", 0.1, 0.5, 0.3)
    X = df_encode.drop("charges", axis=1)
    y = df_encode["charges"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=23
    )
    st.caption(f"Data training: {X_train.shape[0]} baris · Data testing: {X_test.shape[0]} baris")

    st.subheader("4. Training Model Linear Regression")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    st.success("Model berhasil dilatih!")

    st.subheader("Parameter (Koefisien) Model")
    param_df = pd.DataFrame(
        {
            "Fitur": ["intercept"] + list(X.columns),
            "Koefisien": [lin_reg.intercept_] + list(lin_reg.coef_),
        }
    )
    st.write(param_df)

    with st.expander("🔍 Bonus: Normal Equation (verifikasi manual)"):
        st.write(
            "Notebook aslinya juga ngitung parameter model secara manual pakai "
            "Normal Equation θ = (XᵀX)⁻¹Xᵀy, buat verifikasi hasil sklearn."
        )
        X_train_0 = np.c_[np.ones((X_train.shape[0], 1)), X_train.to_numpy()]
        theta = np.linalg.pinv(X_train_0) @ y_train.to_numpy()

        compare_df = pd.DataFrame(
            {
                "Fitur": ["intercept"] + list(X.columns),
                "Theta (Normal Equation)": theta,
                "Sklearn Coef": [lin_reg.intercept_] + list(lin_reg.coef_),
            }
        )
        st.write(compare_df)
        st.caption("Kalau kedua kolom nilainya mirip, implementasi manual & sklearn konsisten ✅")

    # simpan ke session_state biar bisa dipakai di tab 3
    st.session_state["model"] = lin_reg
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

# ---------------------- TAB 3: Evaluasi & Validasi ----------------------
with tab3:
    if "model" not in st.session_state:
        st.warning("Latih model dulu di tab 'Preprocessing & Model'.")
    else:
        lin_reg = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        y_pred = lin_reg.predict(X_test)

        st.subheader("Metrik Evaluasi")
        mse = mean_squared_error(y_test, y_pred)
        r2 = lin_reg.score(X_test, y_test)
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
        col2.metric("R² Score", f"{r2 * 100:.2f}%")

        st.subheader("Validasi Asumsi Model")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax, color="r")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("1. Linearity: Actual vs Predicted")
            st.pyplot(fig)

        residual = np.asarray(y_test - y_pred, dtype=float)
        y_pred_float = np.asarray(y_pred, dtype=float)
        y_test_float = np.asarray(y_test, dtype=float)

        with col2:
            fig, ax = plt.subplots()
            sns.histplot(residual, kde=True, ax=ax, color="b")
            ax.axvline(residual.mean(), color="k", linestyle="--")
            ax.set_title(f"2. Distribusi Residual (mean={residual.mean():.4f})")
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            stats.probplot(residual, plot=ax)
            ax.set_title("3. Q-Q Plot (normalitas residual)")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_pred_float, y=residual, ax=ax, color="r")
            ax.axhline(0, color="k", linestyle="--")
            ax.set_title("4. Homoscedasticity: Residual vs Predicted")
            st.pyplot(fig)

        vif = 1 / (1 - r2)
        st.metric("Variance Inflation Factor (VIF)", f"{vif:.3f}")
        st.caption("VIF < 5 → aman dari multicollinearity yang parah.")