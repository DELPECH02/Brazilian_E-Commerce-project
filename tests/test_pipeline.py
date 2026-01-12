import pandas as pd
from src.data import clean
from src.features import add_features

def test_clean_adds_team_key_and_no_nans():
    df = pd.DataFrame({
        "team": ["San Diego Chargers"],
        "G":[16],
        "Pts":[26.1],
        "TotalYds":[347.9],
        "Ply":[63.9],
        "YdsPerPlay":[5.4],
        "TO":[1.75],
        "FL":[0.75],
        "FirstDowns":[21.1],
        "Cmp":[21.1],
        "Att":[33.2],
        "PassYds":[246.2],
        "TD":[1.81],
        "Int":[1.00],
        "NY/A":[6.8],
        "PassFirstDowns":[12.6],
        "RushAtt":[27.6],
        "RushYds":[101.7],
        "RushTD":[1.38],
        "Y/A":[4.5],
        "RushFirstDowns":[7.25],
        "Pen":[6.88],
        "PenYds":[55.6],
        "FirstDownByPen":[0.25],
        "ScorePct":[38.4],
        "TurnoverPct":[12.9],
        "EXP":[10.0],
        "year":[2010],
    })
    out = clean(df)
    assert "team_key" in out.columns
    assert out.loc[0,"team"] == "Los Angeles Chargers"  # normalization
    assert out.isna().sum().sum() == 0

def test_add_features_creates_expected_columns():
    df = pd.DataFrame({
        "team":["X"], "year":[2020],
        "Att":[30], "RushAtt":[30],
        "PassYds":[240], "RushYds":[120], "TotalYds":[360],
        "PassFirstDowns":[12], "RushFirstDowns":[8], "FirstDownByPen":[2], "FirstDowns":[22],
        "TO":[1], "Ply":[60], "Pen":[6], "PenYds":[50], "G":[16], "Pts":[24],
        "YdsPerPlay":[6], "FL":[0.5], "Cmp":[20], "TD":[2], "Int":[0.5], "NY/A":[7],
        "RushTD":[1], "Y/A":[4], "ScorePct":[40], "TurnoverPct":[10], "EXP":[5],
    })
    out = add_features(df)
    for col in ["pass_rate","pts_per_play","yds_per_point","flags_per_game"]:
        assert col in out.columns
