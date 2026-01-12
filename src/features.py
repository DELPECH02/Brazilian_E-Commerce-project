import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering focused on efficiency and style of play.

    Notes:
    - Some features (e.g. pts_per_play) are useful for descriptive analytics,
      but should NOT be used to predict the same target (Pts) to avoid leakage.
    """
    out = df.copy()

    # Basic ratios
    out["pass_rate"] = out["Att"] / (out["Att"] + out["RushAtt"])
    out["rush_rate"] = 1 - out["pass_rate"]

    out["pass_yds_share"] = out["PassYds"] / out["TotalYds"]
    out["rush_yds_share"] = out["RushYds"] / out["TotalYds"]

    # First down composition
    out["fd_share_pass"] = out["PassFirstDowns"] / out["FirstDowns"]
    out["fd_share_rush"] = out["RushFirstDowns"] / out["FirstDowns"]
    out["fd_share_pen"] = out["FirstDownByPen"] / out["FirstDowns"]

    # Ball security + discipline
    out["turnovers_per_play"] = out["TO"] / out["Ply"]
    out["flags_per_game"] = out["Pen"] / out["G"]
    out["pen_yds_per_flag"] = out["PenYds"] / out["Pen"].replace(0, pd.NA)
    out["pen_yds_per_game"] = out["PenYds"] / out["G"]

    # Descriptive scoring efficiency proxies (OK for EDA, watch for leakage in modeling)
    out["pts_per_play"] = out["Pts"] / out["Ply"]
    out["yds_per_point"] = out["TotalYds"] / out["Pts"].replace(0, pd.NA)

    # Safe fills for divide-by-zero cases
    out = out.fillna(0)

    return out

def model_features(df: pd.DataFrame, target: str = "Pts") -> tuple[pd.DataFrame, pd.Series]:
    """Return X, y with a curated set of features and leakage protection."""
    engineered = add_features(df)

    y = engineered[target]

    # Identifiers
    drop_cols = {"team", "team_key", target}

    # Leakage protection: remove features derived from the target
    if target == "Pts":
        drop_cols |= {"pts_per_play", "yds_per_point"}

    X = engineered.drop(columns=[c for c in drop_cols if c in engineered.columns])
    X = X.select_dtypes(include=["number"]).copy()

    return X, y
