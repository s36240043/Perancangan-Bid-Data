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

    return df_features