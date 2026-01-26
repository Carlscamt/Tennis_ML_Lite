import polars as pl

def debug_h2h():
    df = pl.DataFrame({
        "player_id": [1, 1],
        "opponent_id": [2, 2],
        "player_won": [True, False],
        "start_timestamp": [1000, 2000]
    }).lazy().sort("start_timestamp")
    
    # Logic from add_h2h_features
    # 1. Matchup key
    df = df.with_columns([
        pl.when(pl.col("player_id") < pl.col("opponent_id"))
        .then(pl.concat_str([pl.col("player_id"), pl.lit("_"), pl.col("opponent_id")]))
        .otherwise(pl.concat_str([pl.col("opponent_id"), pl.lit("_"), pl.col("player_id")]))
        .alias("matchup_key")
    ])
    
    # 2. H2H stats
    df = df.with_columns([
        pl.col("player_won")
        .cast(pl.Float64)
        .shift(1)
        .fill_null(0)
        .cum_sum()
        .over(["player_id", "matchup_key"])
        .alias("h2h_wins"),
        
        pl.col("player_id")
        .is_not_null()
        .cast(pl.Int64)
        .shift(1)
        .fill_null(0)
        .cum_sum()
        .over(["player_id", "matchup_key"])
        .alias("h2h_matches")
    ])
    
    res = df.collect()
    print("Result:")
    # print(res)
    print("\nh2h_matches column:", res["h2h_matches"].to_list())
    print("\nh2h_wins column:", res["h2h_wins"].to_list())

if __name__ == "__main__":
    debug_h2h()
