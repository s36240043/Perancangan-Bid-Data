import polars as pl

def generate_fraud_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Mengekstrak fitur anomali bot dari log klik menggunakan Polars.
    """

    # 1. Konversi format waktu
    if df.schema.get("click_time") == pl.String:
        df = df.with_columns(
            pl.col("click_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )

    # 2. Pengurutan waktu
    df = df.sort("click_time")

    # 3. Pipeline Eksekusi Berantai
    df_features = (
        df.with_columns([
            pl.col("click_time").dt.hour().alias("click_hour"),
            pl.col("click_time").dt.minute().alias("click_minute"),
            pl.col("click_time").shift(1).over(["ip", "device", "os"]).alias("prev_click_time")
        ])
        .with_columns([
            # -- FITUR SEKUENSIAL --
            (pl.col("click_time") - pl.col("prev_click_time"))
                .dt.total_seconds()
                .fill_null(-1)
                .alias("seconds_since_prev_click"),

            # -- FITUR DENSITAS WAKTU (KOREKSI DENGAN TRUNCATE) --
            # Total klik IP dalam blok 10 menit
            pl.col("app")
                .count()
                .over([pl.col("ip"), pl.col("click_time").dt.truncate("10m")])
                .alias("ip_clicks_last_10m"),

            # Total klik perangkat (fingerprint) dalam blok 1 jam
            pl.col("app")
                .count()
                .over(["ip", "device", "os", pl.col("click_time").dt.truncate("1h")])
                .alias("fingerprint_clicks_last_1h"),

            # -- FITUR DIVERSITAS --
            pl.col("channel")
                .n_unique()
                .over([pl.col("ip"), pl.col("click_time").dt.truncate("1h")])
                .alias("ip_unique_channels_per_hour")
        ])
        .drop("prev_click_time")
    )

    df_features = df_features.with_columns([
    # Menghitung Standar Deviasi selisih waktu dalam blok 1 Jam untuk setiap IP
    # Semakin kecil nilainya, semakin "mesin" perilakunya
    pl.col("seconds_since_prev_click")
        .std()
        .over([pl.col("ip"), pl.col("click_time").dt.truncate("1h")])
        .alias("ip_rhythm_std_1h"),
        ])

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
    rule_speed     = (pl.col("seconds_since_prev_click") >= 0) & (pl.col("seconds_since_prev_click") < 0.5)
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