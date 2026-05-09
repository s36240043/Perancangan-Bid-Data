import polars as pl
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import numpy as np

def train_isolation_forest(df: pl.DataFrame, features: list, contamination=0.05) -> pl.DataFrame:
    """
    Melatih model Isolation Forest pada data yang lolos dari Hard Block.
    """
    print("\n" + "="*50)
    print("🌲 MEMULAI TRAINING ISOLATION FOREST")
    print("="*50)

    # 1. DATA SPLITTING (Mencegah Overfitting pada Anomali Ekstrem)
    # Kita HANYA melatih data yang statusnya FLAG_FOR_ML atau PASS_ORGANIC
    df_train = df.filter(pl.col("business_action") != "HARD_BLOCK_REJECT_PAYOUT")
    
    # Simpan data sampah untuk digabungkan lagi nanti
    df_trash = df.filter(pl.col("business_action") == "HARD_BLOCK_REJECT_PAYOUT")
    print(f"Data valid untuk ML: {df_train.height:,} baris")
    print(f"Data sampah (dibuang dari ML): {df_trash.height:,} baris")

    # 2. IMPUTASI CERDAS (Smart Imputation)
    # Jika ritme (std) null, artinya klik kurang dari syarat. 
    # Kita isi 999.0 agar ML menganggap ini variansi tinggi (sangat acak = manusia)
    # df_train = df_train.with_columns([
    #     pl.col("ip_rhythm_std_1h").fill_null(999.0)
    # ])

    # 3. KONVERSI MEMORI EFISIEN (Polars -> NumPy)
    # Kita tidak menggunakan .to_pandas() untuk menghindari RAM meledak
    X = df_train.select(features).to_numpy()

    # 4. TRAINING MODEL
    # contamination = estimasi persentase anomali di dataset (misal 5%)
    # n_jobs = -1 (gunakan semua core CPU)
    print("\nMelatih model...")
    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination=contamination, 
        random_state=42, 
        n_jobs=-1
    )
    
    # Fit dan Prediksi sekaligus
    predictions = iso_forest.fit_predict(X)
    
    # Ambil skor anomali mentah (semakin negatif = semakin anomali/bot)
    anomaly_scores = iso_forest.decision_function(X)

    # 5. INTEGRASI HASIL KE POLARS
    # Scikit-Learn output: 1 = Inlier (Normal), -1 = Outlier (Anomali/Bot)
    df_train = df_train.with_columns([
        pl.Series("if_prediction", predictions).cast(pl.Int32),
        pl.Series("if_anomaly_score", anomaly_scores).cast(pl.Float64)
    ])

    # 6. TRANSLASI KE KEPUTUSAN BISNIS (FINAL STATUS)
    df_train = df_train.with_columns([
        # Kondisi 1: ML memvonis ini Bot (-1)
        pl.when(pl.col("if_prediction") == -1)
          .then(pl.lit("ML_DETECTED_BOT"))
          
        # Kondisi 2: ML memvonis Normal (1), DAN status awalnya adalah Flag/Curiga
        .when(pl.col("business_action") == "FLAG_FOR_ML")
          .then(pl.lit("CLEARED_BY_ML"))
          
        # Kondisi 3: Sisanya (yang dari awal memang PASS_ORGANIC dan tidak dicurigai)
        .otherwise(pl.col("business_action"))
        .alias("final_business_action")
    ])
    # Tambahkan kolom final_business_action ke data sampah juga agar skema sama saat digabung
    df_trash = df_trash.with_columns([
        pl.lit(-1).alias("if_prediction").cast(pl.Int32),
        pl.lit(-9.99).alias("if_anomaly_score").cast(pl.Float64), # Dummy score untuk sampah
        pl.col("business_action").alias("final_business_action")
    ])

    # 7. GABUNGKAN KEMBALI KESELURUHAN DATA
    df_final = pl.concat([df_train, df_trash])
    
    print("✅ Training Selesai!")
    return df_final

def train_and_infer_xgboost(df: pl.DataFrame, rule_features: list, ml_features: list) -> pl.DataFrame:
    """
    Melatih model XGBoost pada data Ground Truth ekstrem, 
    lalu memprediksi probabilitas bot pada data yang mencurigakan (FLAG_FOR_ML).
    """
    print("\n" + "="*60)
    print("🚀 MEMULAI SEMI-SUPERVISED XGBOOST PIPELINE")
    print("="*60)

    

    # 1. PERSIAPAN DATA PELATIHAN (GROUND TRUTH)
    print("1. Mengekstraksi Data Ground Truth...")
    df_train = df.filter(
        pl.col("business_action").is_in(["HARD_BLOCK_REJECT_PAYOUT", "PASS_ORGANIC"])
    )
    
    # Membuat Pseudo-Label (Target Variable Y)
    df_train = df_train.with_columns([
        pl.when(pl.col("business_action") == "HARD_BLOCK_REJECT_PAYOUT")
          .then(pl.lit(1))
          .otherwise(pl.lit(0))
          .alias("is_bot_target")
    ])

    # Imputasi nilai kosong pada fitur ML (contoh: ip_rhythm_std_1h)
    # Kita menggunakan nilai median agar tidak terdeteksi sebagai outlier
    # for feat in ml_features:
    #     if df_train.schema[feat] in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
    #         df_train = df_train.with_columns(pl.col(feat).fill_null(999.0))

    # Konversi ke NumPy (Menghindari Pandas untuk efisiensi RAM)
    X_train = df_train.select(ml_features).to_numpy()
    y_train = df_train.select("is_bot_target").to_numpy().flatten()
    print(f"   Distribusi Pelatihan: {np.sum(y_train==1):,} Bot | {np.sum(y_train==0):,} Manusia")

    # 2. PELATIHAN MODEL XGBOOST
    print("\n2. Melatih Model XGBoost (Orthogonal Feature Space)...")
    # Parameter dioptimalkan untuk kecepatan dan pencegahan overfitting (regularisasi L2)
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 5,          # Pohon tidak terlalu dalam untuk mencegah hafalan
        'learning_rate': 0.1,
        'n_estimators': 100,
        'reg_lambda': 1.0,       # L2 Regularization (Penalti fitur dominan)
        'n_jobs': -1,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    # 3. PERSIAPAN DATA INFERENSI (ZONA ABU-ABU)
    print("\n3. Mengeksekusi Inferensi pada Zona Abu-abu (FLAG_FOR_ML)...")
    df_grey = df.filter(pl.col("business_action") == "FLAG_FOR_ML")
    
    # # Imputasi fitur ML pada data inferensi menggunakan logika yang sama
    # for feat in ml_features:
    #     if df_grey.schema[feat] in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
    #         df_grey = df_grey.with_columns(pl.col(feat).fill_null(999.0))

    X_grey = df_grey.select(ml_features).to_numpy()

    # 4. PREDIKSI PROBABILITAS
    # predict_proba menghasilkan matriks [Probabilitas 0, Probabilitas 1]
    grey_probabilities = model.predict_proba(X_grey)[:, 1]

    # Menggabungkan hasil probabilitas kembali ke dataframe Polars
    df_grey_scored = df_grey.with_columns([
        pl.Series("bot_probability", grey_probabilities).cast(pl.Float32)
    ])

    # 5. INTEGRASI KEPUTUSAN BISNIS (THRESHOLDING)
    # Asumsi Threshold: Jika probabilitas > 0.85, anggap Bot. Sisanya bersihkan.
    df_grey_scored = df_grey_scored.with_columns([
        pl.when(pl.col("bot_probability") > 0.85)
          .then(pl.lit("ML_DETECTED_BOT"))
          .otherwise(pl.lit("CLEARED_BY_ML"))
          .alias("final_business_action")
    ])

    # Untuk data pelatihan (Ground Truth), berikan probabilitas absolut dan pertahankan status aslinya
    df_train_scored = df_train.with_columns([
        pl.when(pl.col("is_bot_target") == 1)
          .then(pl.lit(1.0).cast(pl.Float32))
          .otherwise(pl.lit(0.0).cast(pl.Float32))
          .alias("bot_probability"),
        pl.col("business_action").alias("final_business_action")
    ]).drop("is_bot_target") # Buang kolom pseudo-label agar skema sama

    # 6. GABUNGKAN SELURUH DATA
    df_final = pl.concat([df_train_scored, df_grey_scored])
    
    print("\n✅ Pipeline Selesai! Dataframe akhir telah disatukan.")
    return model, df_final

# --- CARA PENGGUNAAN ---
# rule_features = ["ip_clicks_last_10m", "seconds_since_prev_click", "fingerprint_clicks_last_1h"]
# ml_features = ["ip_rhythm_std_1h", "ip_unique_channels_per_hour", "device_os_entropy", "is_first_click"] # Fitur Ortogonal

# df_scored = train_and_infer_xgboost(df_strategy, rule_features, ml_features)