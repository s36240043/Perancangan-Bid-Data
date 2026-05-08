import polars as pl

def apply_business_blocking_strategy(df: pl.DataFrame) -> pl.DataFrame:
    """
    Menerapkan strategi pemblokiran final berdasarkan logika finansial.
    Menghasilkan kolom 'business_action' untuk memandu pipeline selanjutnya.
    """
    
    # 1. Definisi Kategori Aturan
    # HARUS DIBLOKIR: Pencurian jelas dan serangan brute-force
    hard_block_rules = [
        "Click Injection", 
        "Super Human Speed", 
        "Burst Clicks", 
        "Persistent Emulator"
    ]
    
    # SUSPICIOUS (MENCURIGAKAN): Butuh evaluasi Machine Learning
    ml_flag_rules = [
        "Channel Spraying"
    ]
    
    # 2. Implementasi Logika ke Kolom Baru
    df_strategy = df.with_columns([
        pl.when(pl.col("rule_based_reason").is_in(hard_block_rules))
          .then(pl.lit("HARD_BLOCK_REJECT_PAYOUT"))
          
          .when(pl.col("rule_based_reason").is_in(ml_flag_rules))
          .then(pl.lit("FLAG_FOR_ML"))
          
          .otherwise(pl.lit("PASS_ORGANIC"))
          .alias("business_action")
    ])
    
    return df_strategy
