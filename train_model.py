import polars as pl
from sklearn.ensemble import IsolationForest
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
    df_train = df_train.with_columns([
        pl.col("ip_rhythm_std_1h").fill_null(999.0)
    ])

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

