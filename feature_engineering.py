import polars as pl
def generate_fraud_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Mengekstrak fitur anomali bot dari log klik menggunakan Polars.
    Versi ini dioptimalkan untuk XGBoost (Native NaN Handling) dan menghindari data leakage.
    """
    
    # 1. Konversi format waktu
    if df.schema.get("click_time") == pl.String:
        df = df.with_columns(
            pl.col("click_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )

    # 2. Pengurutan waktu terpusat (Krusial untuk fungsi shift)
    df = df.sort("click_time")

    # 3. Pipeline Eksekusi Berantai
    df_features = (
        df.with_columns([
            pl.col("click_time").dt.hour().alias("click_hour"),
            pl.col("click_time").dt.minute().alias("click_minute"),
            # Mencari waktu klik sebelumnya spesifik untuk setiap IP
            pl.col("click_time").shift(1).over("ip").alias("prev_click_time")
        ])
        .with_columns([
            # -- FITUR SEKUENSIAL --
            # Dibiarkan bernilai Null untuk klik pertama (tanpa fill_null)
            (pl.col("click_time") - pl.col("prev_click_time"))
                .dt.total_seconds()
                .alias("seconds_since_prev_click"),

            # -- FITUR DENSITAS WAKTU --
            pl.col("click_time").rank("dense")
                .over([pl.col("ip"), pl.col("click_time").dt.truncate("10m")])
                .alias("ip_clicks_last_10m"),

            pl.col("click_time").rank("dense")
                .over(["ip", "device", "os", pl.col("click_time").dt.truncate("1h")])
                .alias("fingerprint_clicks_last_1h"),

            # -- FITUR DIVERSITAS (EXPANDING / NO FUTURE LEAK) --
            # tandai kemunculan pertama channel dalam (ip, hour) lalu akumulasi => distinct so far
            (
                (pl.col("channel")
                   .cum_count()
                   .over([pl.col("ip"), pl.col("channel"), pl.col("click_time").dt.truncate("1h")]) == 0)
                .cast(pl.Int32)
                .cum_sum()
                .over([pl.col("ip"), pl.col("click_time").dt.truncate("1h")])
            ).alias("ip_unique_channels_per_hour"),

            (
                (pl.col("os")
                   .cum_count()
                   .over([pl.col("ip"), pl.col("os"), pl.col("click_time").dt.truncate("1h")]) == 0)
                .cast(pl.Int32)
                .cum_sum()
                .over([pl.col("ip"), pl.col("click_time").dt.truncate("1h")])
            ).alias("ip_unique_os_per_hour"),

            (
                (pl.col("app")
                   .cum_count()
                   .over([pl.col("ip"), pl.col("app"), pl.col("click_time").dt.truncate("1h")]) == 0)
                .cast(pl.Int32)
                .cum_sum()
                .over([pl.col("ip"), pl.col("click_time").dt.truncate("1h")])
            ).alias("ip_unique_apps_per_hour")
        ])
        .with_columns([
            # -- FITUR RITME (ORTOGONAL L2) --
            # Fungsi std() di Polars secara otomatis akan mengabaikan nilai Null
            pl.col("seconds_since_prev_click").std()
                .over([pl.col("ip"), pl.col("click_time").dt.truncate("1h")])
                .alias("ip_rhythm_std_1h"),
                
            # -- FITUR INDIKATOR (PERBAIKAN LOGIKA) --
            # Klik pertama dikonfirmasi jika prev_click_time kosong
            pl.col("prev_click_time").is_null().cast(pl.Int32).alias("is_first_click")
        ])
        .drop("prev_click_time") # Membersihkan memori dari kolom sementara
    )

    return df_features

def apply_heuristic_rules(df: pl.DataFrame) -> pl.DataFrame:
    """
    Melabeli baris sebagai 'obvious bot' berdasarkan ambang batas heuristik.
    """
    # Pastikan attributed_time sudah datetime jika ada
    if "attributed_time" in df.columns and df.schema.get("attributed_time") == pl.String:
        df = df.with_columns(
            pl.col("attributed_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        )

    # Kalkulasi CTIT (Click-To-Install Time)
    if "attributed_time" in df.columns:
        df = df.with_columns(
            (pl.col("attributed_time") - pl.col("click_time"))
                .dt.total_seconds()
                .alias("ctit_seconds")
        )
    else:
        df = df.with_columns(pl.lit(None).alias("ctit_seconds"))

    # Definisi Rules (Predicate)
    rule_speed     = (pl.col("seconds_since_prev_click") >= 0) & (pl.col("seconds_since_prev_click") < 1.0)
    rule_burst     = pl.col("ip_clicks_last_10m") > 300
    rule_emulator  = pl.col("fingerprint_clicks_last_1h") > 150
    rule_spraying  = pl.col("ip_unique_channels_per_hour") > 20
    rule_injection = (pl.col("ctit_seconds").is_not_null()) & (pl.col("ctit_seconds") < 3.0)

    # Aplikasi Label dan Alasan
    df = df.with_columns([
        pl.when(rule_speed | rule_burst | rule_emulator | rule_spraying | rule_injection)
          .then(True)
          .otherwise(False)
          .alias("is_obvious_bot"),

        pl.when(rule_injection).then(pl.lit("Click Injection"))
          .when(rule_speed).then(pl.lit("Super Human Speed"))
          .when(rule_burst).then(pl.lit("Burst Clicks"))
          .when(rule_emulator).then(pl.lit("Persistent Emulator"))
          .when(rule_spraying).then(pl.lit("Channel Spraying"))
          .otherwise(pl.lit("Normal"))
          .alias("rule_based_reason")
    ])

    return df