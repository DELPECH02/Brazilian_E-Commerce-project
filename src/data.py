import pandas as pd

TEAM_NORMALIZATION = {
    # Franchises that changed city/name over time
    "St. Louis Rams": "Los Angeles Rams",
    "San Diego Chargers": "Los Angeles Chargers",
    "Oakland Raiders": "Las Vegas Raiders",
    "Washington Redskins": "Washington Commanders",
    "Washington Football Team": "Washington Commanders",
}

NUMERIC_COLS = [
    "G","Pts","TotalYds","Ply","YdsPerPlay","TO","FL","FirstDowns",
    "Cmp","Att","PassYds","TD","Int","NY/A","PassFirstDowns",
    "RushAtt","RushYds","RushTD","Y/A","RushFirstDowns","Pen",
    "PenYds","FirstDownByPen","ScorePct","TurnoverPct","EXP","year"
]

def load_raw_csv(path: str | None = None) -> pd.DataFrame:
    """Load raw scraped dataset."""
    if path is None:
        raise ValueError("path is required (e.g., data/raw/nfl_offense.csv)")
    return pd.read_csv(path)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning + schema normalization.

    - Normalizes team names for franchises that rebranded/relocated.
    - Ensures numeric columns are numeric.
    - Adds a stable team_key column for joins/filters.
    """
    out = df.copy()

    if "team" not in out.columns or "year" not in out.columns:
        raise ValueError("Expected columns 'team' and 'year'")

    out["team"] = out["team"].astype(str).str.strip()
    out["team"] = out["team"].replace(TEAM_NORMALIZATION)

    for c in NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if out[NUMERIC_COLS].isna().any().any():
        bad = out[NUMERIC_COLS].isna().sum().sort_values(ascending=False)
        raise ValueError(
            "Found missing/invalid numeric values after coercion. "
            f"Top offenders:\n{bad.head(10)}"
        )

    out["year"] = out["year"].astype(int)

    out["team_key"] = (
        out["team"]
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "-", regex=True)
        .str.strip("-")
    )

    out = out.drop_duplicates(subset=["team_key", "year"]).reset_index(drop=True)
    return out

def save_processed_csv(df: pd.DataFrame, path: str) -> None:
    """Save processed dataset as CSV (no extra dependencies)."""
    df.to_csv(path, index=False)
