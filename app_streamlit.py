# app_streamlit.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import requests
import streamlit.components.v1 as components

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

try:
    import joblib
except Exception:
    joblib = None


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="NFL Offense Analytics",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "raw" / "nfl_offense.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

# =========================================================
# CUSTOM CSS — Dark Pro Theme
# =========================================================
st.markdown("""
<style>
/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #111d33 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #c8d6e5 !important;
    font-weight: 500;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
[data-testid="stMetric"] label {
    color: #8ab4f8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e8eaed !important;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1557b0 0%, #1a73e8 100%);
    box-shadow: 0 4px 12px rgba(26,115,232,0.4);
    transform: translateY(-1px);
}

/* Expander */
.streamlit-expanderHeader {
    background: #0d1b2a !important;
    border-radius: 8px;
    font-weight: 600;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* Cards */
div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div > div[data-testid="stContainer"] {
    background: linear-gradient(135deg, #0d1b2a 0%, #162033 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    transition: all 0.2s ease;
}
div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div > div[data-testid="stContainer"]:hover {
    border-color: #4285f4;
    box-shadow: 0 4px 20px rgba(66,133,244,0.15);
}

/* Headings */
h1, h2, h3 {
    color: #e8eaed !important;
}

/* Page title bar */
.title-bar {
    background: linear-gradient(135deg, #0a1628 0%, #1a2742 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.title-bar h1 {
    margin: 0;
    font-size: 1.8rem;
    background: linear-gradient(135deg, #8ab4f8, #4285f4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.title-bar p {
    color: #8899aa;
    margin: 4px 0 0 0;
    font-size: 0.95rem;
}

/* Stat badge */
.stat-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #8ab4f8;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# PLOTLY THEME
# =========================================================
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,42,0.8)",
    font=dict(family="Inter", color="#c8d6e5"),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=["#4285f4", "#34a853", "#fbbc04", "#ea4335", "#8ab4f8",
              "#81c995", "#fdd663", "#f28b82", "#aecbfa", "#a8dab5"],
)


def plotly_fig(**kwargs) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, **kwargs)
    return fig


# =========================================================
# HELPERS
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def call_api_prediction(features_dict: dict) -> Optional[float]:
    url = "http://127.0.0.1:8000/predict"
    try:
        clean_features = {}
        for k, v in features_dict.items():
            if pd.isna(v):
                clean_features[k] = 0.0
            else:
                clean_features[k] = float(v) if hasattr(v, "item") else v
        response = requests.post(url, json={"features": clean_features}, timeout=5)
        if response.status_code == 200:
            return response.json()["prediction"]
        else:
            st.error(f"Erreur Backend ({response.status_code}) : {response.text}")
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
    return None


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def infer_time_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    team_col = None
    year_col = None
    for c in df.columns:
        cl = c.lower()
        if team_col is None and any(k in cl for k in ["team", "franchise", "club"]):
            team_col = c
        if year_col is None and any(k in cl for k in ["season", "year"]):
            year_col = c
    return team_col, year_col


def guess_target_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols_low = [c.lower() for c in df.columns]
    cols = list(df.columns)

    def pick(keys: List[str]) -> Optional[str]:
        candidates = []
        for i, cl in enumerate(cols_low):
            if any(k in cl for k in keys):
                candidates.append(cols[i])
        bad = ["rank", "name", "team", "season", "year", "id"]
        candidates = [c for c in candidates if not any(b in c.lower() for b in bad)]
        return candidates[0] if candidates else None

    points = pick(["pts", "points", "point", "score", "scoring"])
    yards = pick(["yds", "yards", "yard"])
    return points, yards


def numeric_feature_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]


def nice_metric_row(metrics: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{metrics['rmse']:.3f}")
    c2.metric("MAE", f"{metrics['mae']:.3f}")
    c3.metric("R²", f"{metrics['r2']:.3f}")


def train_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    test_years: int = 3,
    year_col: Optional[str] = None,
    random_state: int = 42,
    n_estimators: int = 600,
    max_depth: Optional[int] = None,
) -> Tuple[RandomForestRegressor, dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = df.dropna(subset=[target_col]).copy()
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    X = X.fillna(X.median(numeric_only=True))

    if year_col and year_col in data.columns and pd.api.types.is_numeric_dtype(data[year_col]):
        years_sorted = sorted(data[year_col].dropna().unique())
        if len(years_sorted) > test_years:
            cutoff = years_sorted[-test_years]
            train_idx = data[year_col] < cutoff
            test_idx = data[year_col] >= cutoff
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1, max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "rmse": rmse(y_test, pred),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "target": target_col,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return model, metrics, X_train, X_test, y_train, y_test


def save_model(model, path: Path):
    if joblib is None:
        return
    joblib.dump(model, path)


def load_model(path: Path):
    if joblib is None or not path.exists():
        return None
    return joblib.load(path)


# =========================================================
# PLOTLY CHART HELPERS
# =========================================================
def plot_scatter_plotly(y_true: pd.Series, y_pred: np.ndarray, title: str):
    fig = px.scatter(
        x=y_true, y=y_pred, labels={"x": "Valeur réelle", "y": "Prédiction"},
        title=title, opacity=0.7,
    )
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="#ea4335", dash="dash", width=2),
                             name="Parfait", showlegend=True))
    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    st.plotly_chart(fig, use_container_width=True)


def plot_importance_bar(series: pd.Series, title: str, top_n: int = 20):
    s = series.sort_values(ascending=False).head(top_n)
    fig = px.bar(
        x=s.values, y=s.index, orientation="h",
        title=title, labels={"x": "Importance", "y": "Variable"},
        color=s.values, color_continuous_scale="Blues",
    )
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), showlegend=False,
                      coloraxis_showscale=False, height=max(400, top_n * 25))
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(df: pd.DataFrame, cols: List[str], title: str):
    corr = df[cols].corr(numeric_only=True)
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        title=title, aspect="auto", zmin=-1, zmax=1,
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=max(500, len(cols) * 30))
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TEAM LOCATIONS & ESPN
# =========================================================
ESPN_TEAM_SLUG = {
    "ARI": "ari", "ATL": "atl", "BAL": "bal", "BUF": "buf", "CAR": "car",
    "CHI": "chi", "CIN": "cin", "CLE": "cle", "DAL": "dal", "DEN": "den",
    "DET": "det", "GB": "gb", "HOU": "hou", "IND": "ind", "JAX": "jax",
    "KC": "kc", "LV": "lv", "LAC": "lac", "LAR": "lar", "MIA": "mia",
    "MIN": "min", "NE": "ne", "NO": "no", "NYG": "nyg", "NYJ": "nyj",
    "PHI": "phi", "PIT": "pit", "SF": "sf", "SEA": "sea", "TB": "tb",
    "TEN": "ten", "WAS": "wsh"
}

# =========================================================
# NFL TEAMS DATABASE — Colors & 2024 Offensive Starters
# =========================================================
NFL_TEAMS_DATA = {
    "Arizona Cardinals": {
        "abbr": "ARI", "primary": "#97233F", "secondary": "#000000", "accent": "#FFB612",
        "roster": {"QB": "Kyler Murray", "RB": "James Conner", "WR1": "Marvin Harrison Jr.", "WR2": "Michael Wilson", "WR3": "Greg Dortch", "TE": "Trey McBride"},
    },
    "Atlanta Falcons": {
        "abbr": "ATL", "primary": "#A71930", "secondary": "#000000", "accent": "#A5ACAF",
        "roster": {"QB": "Kirk Cousins", "RB": "Bijan Robinson", "WR1": "Drake London", "WR2": "Darnell Mooney", "WR3": "Ray-Ray McCloud", "TE": "Kyle Pitts"},
    },
    "Baltimore Ravens": {
        "abbr": "BAL", "primary": "#241773", "secondary": "#000000", "accent": "#9E7C0C",
        "roster": {"QB": "Lamar Jackson", "RB": "Derrick Henry", "WR1": "Zay Flowers", "WR2": "Rashod Bateman", "WR3": "Nelson Agholor", "TE": "Mark Andrews"},
    },
    "Buffalo Bills": {
        "abbr": "BUF", "primary": "#00338D", "secondary": "#C60C30", "accent": "#FFFFFF",
        "roster": {"QB": "Josh Allen", "RB": "James Cook", "WR1": "Keon Coleman", "WR2": "Khalil Shakir", "WR3": "Curtis Samuel", "TE": "Dalton Kincaid"},
    },
    "Carolina Panthers": {
        "abbr": "CAR", "primary": "#0085CA", "secondary": "#101820", "accent": "#BFC0BF",
        "roster": {"QB": "Bryce Young", "RB": "Chuba Hubbard", "WR1": "Diontae Johnson", "WR2": "Adam Thielen", "WR3": "Jonathan Mingo", "TE": "Tommy Tremble"},
    },
    "Chicago Bears": {
        "abbr": "CHI", "primary": "#0B162A", "secondary": "#C83200", "accent": "#FFFFFF",
        "roster": {"QB": "Caleb Williams", "RB": "D'Andre Swift", "WR1": "DJ Moore", "WR2": "Keenan Allen", "WR3": "Rome Odunze", "TE": "Cole Kmet"},
    },
    "Cincinnati Bengals": {
        "abbr": "CIN", "primary": "#FB4F14", "secondary": "#000000", "accent": "#FFFFFF",
        "roster": {"QB": "Joe Burrow", "RB": "Zack Moss", "WR1": "Ja'Marr Chase", "WR2": "Tee Higgins", "WR3": "Andrei Iosivas", "TE": "Mike Gesicki"},
    },
    "Cleveland Browns": {
        "abbr": "CLE", "primary": "#311D00", "secondary": "#FF3C00", "accent": "#FFFFFF",
        "roster": {"QB": "Deshaun Watson", "RB": "Jerome Ford", "WR1": "Amari Cooper", "WR2": "Jerry Jeudy", "WR3": "Elijah Moore", "TE": "David Njoku"},
    },
    "Dallas Cowboys": {
        "abbr": "DAL", "primary": "#003594", "secondary": "#041E42", "accent": "#869397",
        "roster": {"QB": "Dak Prescott", "RB": "Rico Dowdle", "WR1": "CeeDee Lamb", "WR2": "Brandin Cooks", "WR3": "Jalen Tolbert", "TE": "Jake Ferguson"},
    },
    "Denver Broncos": {
        "abbr": "DEN", "primary": "#002244", "secondary": "#FB4F14", "accent": "#FFFFFF",
        "roster": {"QB": "Bo Nix", "RB": "Javonte Williams", "WR1": "Courtland Sutton", "WR2": "Josh Reynolds", "WR3": "Marvin Mims Jr.", "TE": "Adam Trautman"},
    },
    "Detroit Lions": {
        "abbr": "DET", "primary": "#0076B6", "secondary": "#B0B7BC", "accent": "#000000",
        "roster": {"QB": "Jared Goff", "RB": "Jahmyr Gibbs", "WR1": "Amon-Ra St. Brown", "WR2": "Jameson Williams", "WR3": "Kalif Raymond", "TE": "Sam LaPorta"},
    },
    "Green Bay Packers": {
        "abbr": "GB", "primary": "#203731", "secondary": "#FFB612", "accent": "#FFFFFF",
        "roster": {"QB": "Jordan Love", "RB": "Josh Jacobs", "WR1": "Jayden Reed", "WR2": "Romeo Doubs", "WR3": "Dontayvion Wicks", "TE": "Tucker Kraft"},
    },
    "Houston Texans": {
        "abbr": "HOU", "primary": "#03202F", "secondary": "#A71930", "accent": "#FFFFFF",
        "roster": {"QB": "C.J. Stroud", "RB": "Joe Mixon", "WR1": "Nico Collins", "WR2": "Stefon Diggs", "WR3": "Tank Dell", "TE": "Dalton Schultz"},
    },
    "Indianapolis Colts": {
        "abbr": "IND", "primary": "#002C5F", "secondary": "#A2AAAD", "accent": "#FFFFFF",
        "roster": {"QB": "Anthony Richardson", "RB": "Jonathan Taylor", "WR1": "Michael Pittman Jr.", "WR2": "Josh Downs", "WR3": "Alec Pierce", "TE": "Mo Alie-Cox"},
    },
    "Jacksonville Jaguars": {
        "abbr": "JAX", "primary": "#006778", "secondary": "#101820", "accent": "#9F792C",
        "roster": {"QB": "Trevor Lawrence", "RB": "Travis Etienne Jr.", "WR1": "Christian Kirk", "WR2": "Gabe Davis", "WR3": "Brian Thomas Jr.", "TE": "Evan Engram"},
    },
    "Kansas City Chiefs": {
        "abbr": "KC", "primary": "#E31837", "secondary": "#FFB81C", "accent": "#FFFFFF",
        "roster": {"QB": "Patrick Mahomes", "RB": "Isiah Pacheco", "WR1": "Rashee Rice", "WR2": "Xavier Worthy", "WR3": "Hollywood Brown", "TE": "Travis Kelce"},
    },
    "Las Vegas Raiders": {
        "abbr": "LV", "primary": "#000000", "secondary": "#A5ACAF", "accent": "#FFFFFF",
        "roster": {"QB": "Gardner Minshew", "RB": "Zamir White", "WR1": "Davante Adams", "WR2": "Jakobi Meyers", "WR3": "Tre Tucker", "TE": "Brock Bowers"},
    },
    "Los Angeles Chargers": {
        "abbr": "LAC", "primary": "#0080C6", "secondary": "#FFC20E", "accent": "#FFFFFF",
        "roster": {"QB": "Justin Herbert", "RB": "J.K. Dobbins", "WR1": "Quentin Johnston", "WR2": "Ladd McConkey", "WR3": "Joshua Palmer", "TE": "Will Dissly"},
    },
    "Los Angeles Rams": {
        "abbr": "LAR", "primary": "#003594", "secondary": "#FFA300", "accent": "#FFFFFF",
        "roster": {"QB": "Matthew Stafford", "RB": "Kyren Williams", "WR1": "Puka Nacua", "WR2": "Cooper Kupp", "WR3": "Demarcus Robinson", "TE": "Tyler Higbee"},
    },
    "Miami Dolphins": {
        "abbr": "MIA", "primary": "#008E97", "secondary": "#FC4C02", "accent": "#FFFFFF",
        "roster": {"QB": "Tua Tagovailoa", "RB": "De'Von Achane", "WR1": "Tyreek Hill", "WR2": "Jaylen Waddle", "WR3": "River Cracraft", "TE": "Jonnu Smith"},
    },
    "Minnesota Vikings": {
        "abbr": "MIN", "primary": "#4F2683", "secondary": "#FFC62F", "accent": "#FFFFFF",
        "roster": {"QB": "Sam Darnold", "RB": "Aaron Jones", "WR1": "Justin Jefferson", "WR2": "Jordan Addison", "WR3": "Jalen Nailor", "TE": "T.J. Hockenson"},
    },
    "New England Patriots": {
        "abbr": "NE", "primary": "#002244", "secondary": "#C60C30", "accent": "#B0B7BC",
        "roster": {"QB": "Jacoby Brissett", "RB": "Rhamondre Stevenson", "WR1": "DeMario Douglas", "WR2": "Ja'Lynn Polk", "WR3": "Kendrick Bourne", "TE": "Hunter Henry"},
    },
    "New Orleans Saints": {
        "abbr": "NO", "primary": "#101820", "secondary": "#D3BC8D", "accent": "#FFFFFF",
        "roster": {"QB": "Derek Carr", "RB": "Alvin Kamara", "WR1": "Chris Olave", "WR2": "Rashid Shaheed", "WR3": "A.T. Perry", "TE": "Juwan Johnson"},
    },
    "New York Giants": {
        "abbr": "NYG", "primary": "#0B2265", "secondary": "#A71930", "accent": "#A5ACAF",
        "roster": {"QB": "Daniel Jones", "RB": "Devin Singletary", "WR1": "Malik Nabers", "WR2": "Darius Slayton", "WR3": "Wan'Dale Robinson", "TE": "Darren Waller"},
    },
    "New York Jets": {
        "abbr": "NYJ", "primary": "#125740", "secondary": "#000000", "accent": "#FFFFFF",
        "roster": {"QB": "Aaron Rodgers", "RB": "Breece Hall", "WR1": "Garrett Wilson", "WR2": "Mike Williams", "WR3": "Allen Lazard", "TE": "Tyler Conklin"},
    },
    "Philadelphia Eagles": {
        "abbr": "PHI", "primary": "#004C54", "secondary": "#A5ACAF", "accent": "#ACC0C6",
        "roster": {"QB": "Jalen Hurts", "RB": "Saquon Barkley", "WR1": "A.J. Brown", "WR2": "DeVonta Smith", "WR3": "Jahan Dotson", "TE": "Dallas Goedert"},
    },
    "Pittsburgh Steelers": {
        "abbr": "PIT", "primary": "#101820", "secondary": "#FFB612", "accent": "#C60C30",
        "roster": {"QB": "Russell Wilson", "RB": "Najee Harris", "WR1": "George Pickens", "WR2": "Van Jefferson", "WR3": "Roman Wilson", "TE": "Pat Freiermuth"},
    },
    "San Francisco 49ers": {
        "abbr": "SF", "primary": "#AA0000", "secondary": "#B3995D", "accent": "#FFFFFF",
        "roster": {"QB": "Brock Purdy", "RB": "Christian McCaffrey", "WR1": "Deebo Samuel", "WR2": "Brandon Aiyuk", "WR3": "Jauan Jennings", "TE": "George Kittle"},
    },
    "Seattle Seahawks": {
        "abbr": "SEA", "primary": "#002244", "secondary": "#69BE28", "accent": "#A5ACAF",
        "roster": {"QB": "Geno Smith", "RB": "Kenneth Walker III", "WR1": "DK Metcalf", "WR2": "Tyler Lockett", "WR3": "Jaxon Smith-Njigba", "TE": "Noah Fant"},
    },
    "Tampa Bay Buccaneers": {
        "abbr": "TB", "primary": "#D50A0A", "secondary": "#34302B", "accent": "#FF7900",
        "roster": {"QB": "Baker Mayfield", "RB": "Rachaad White", "WR1": "Mike Evans", "WR2": "Chris Godwin", "WR3": "Sterling Shepard", "TE": "Cade Otton"},
    },
    "Tennessee Titans": {
        "abbr": "TEN", "primary": "#0C2340", "secondary": "#4B92DB", "accent": "#C8102E",
        "roster": {"QB": "Will Levis", "RB": "Tony Pollard", "WR1": "DeAndre Hopkins", "WR2": "Calvin Ridley", "WR3": "Tyler Boyd", "TE": "Chig Okonkwo"},
    },
    "Washington Commanders": {
        "abbr": "WAS", "primary": "#5A1414", "secondary": "#FFB612", "accent": "#FFFFFF",
        "roster": {"QB": "Jayden Daniels", "RB": "Brian Robinson Jr.", "WR1": "Terry McLaurin", "WR2": "Jahan Dotson", "WR3": "Luke McCaffrey", "TE": "Zach Ertz"},
    },
}

# Reverse lookup: abbreviation -> full name
_ABBR_TO_FULLNAME = {v["abbr"]: k for k, v in NFL_TEAMS_DATA.items()}


def match_team_to_nfl(team_value: str) -> Optional[dict]:
    """Match a CSV team value to its NFL_TEAMS_DATA entry."""
    if not team_value:
        return None
    t = str(team_value).strip()
    # Direct full-name match
    if t in NFL_TEAMS_DATA:
        return NFL_TEAMS_DATA[t]
    # Abbreviation match
    t_upper = t.upper()
    if t_upper in _ABBR_TO_FULLNAME:
        return NFL_TEAMS_DATA[_ABBR_TO_FULLNAME[t_upper]]
    # Fuzzy: check if CSV value is contained in any team name
    for name, data in NFL_TEAMS_DATA.items():
        if t_upper in name.upper() or name.upper() in t_upper:
            return data
    return None


def render_team_header_html(team_name: str, team_data: dict, subtitle: str = "") -> str:
    """Generate a branded team header HTML block with ESPN team logo."""
    p = team_data["primary"]
    s = team_data["secondary"]
    a = team_data["accent"]
    abbr = team_data["abbr"]
    slug = ESPN_TEAM_SLUG.get(abbr, abbr.lower())
    logo_url = f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"
    return f'''
    <div style="background:linear-gradient(135deg, {p} 0%, {s} 100%);
                border-radius:16px; padding:40px 40px 32px; margin-bottom:28px;
                position:relative; overflow:hidden; border:1px solid rgba(255,255,255,0.1);">
        <div style="position:absolute; top:50%; right:30px; transform:translateY(-50%);
                    width:140px; height:140px; opacity:0.15;
                    background:url('{logo_url}') center/contain no-repeat;"></div>
        <div style="position:absolute; bottom:-20px; left:50%; width:300px; height:300px;
                    border-radius:50%; background:{a}; opacity:0.05; transform:translateX(-50%);"></div>
        <div style="position:relative; z-index:1;">
            <span style="display:inline-block; background:rgba(0,0,0,0.3); padding:4px 14px;
                         border-radius:20px; font-size:0.75rem; color:{a}; font-weight:600;
                         letter-spacing:0.1em; text-transform:uppercase; margin-bottom:12px;">
                {abbr} &middot; NFL Offense
            </span>
            <h1 style="margin:8px 0 0; font-size:2.6rem; color:white; font-weight:800;
                       text-transform:uppercase; letter-spacing:0.02em; line-height:1.1;
                       text-shadow:0 2px 20px rgba(0,0,0,0.3);">{team_name}</h1>
            <p style="color:rgba(255,255,255,0.6); margin:6px 0 0; font-size:1rem;">{subtitle}</p>
        </div>
    </div>
    '''


def render_formation_html(team_name: str, team_data: dict) -> str:
    """Generate a full HTML page for NFL offensive formation (rendered via components.html)."""
    p = team_data["primary"]
    s = team_data["secondary"]
    a = team_data["accent"]
    r = team_data["roster"]
    abbr = team_data["abbr"]
    slug = ESPN_TEAM_SLUG.get(abbr, abbr.lower())
    logo_url = f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"

    players = [
        (8,  12, "WR", r["WR1"], False),
        (8,  88, "WR", r["WR2"], False),
        (22, 68, "SLOT", r["WR3"], False),
        (33, 24, "TE", r["TE"], False),
        (56, 50, "QB", r["QB"], True),
        (75, 50, "RB", r["RB"], False),
    ]

    player_divs = ""
    for top, left, pos, name, is_qb in players:
        bg = "rgba(255,255,255,0.15)" if is_qb else "rgba(0,0,0,0.50)"
        border = f"2px solid {a}" if is_qb else "1px solid rgba(255,255,255,0.2)"
        shadow = f"0 0 30px {a}55" if is_qb else "0 4px 15px rgba(0,0,0,0.3)"
        player_divs += f'''<div class="player" style="top:{top}%;left:{left}%;
            background:{bg};border:{border};box-shadow:{shadow};">
            <div class="pos">{pos}</div>
            <div class="name">{name}</div>
        </div>'''

    ol_circles = ""
    for pos in ["LT", "LG", "C", "RG", "RT"]:
        sz = "34px" if pos == "C" else "28px"
        bg = "rgba(255,255,255,0.15)" if pos == "C" else "rgba(255,255,255,0.08)"
        bw = "2px" if pos == "C" else "1px"
        ol_circles += f'''<div style="width:{sz};height:{sz};border-radius:50%;background:{bg};
            border:{bw} solid rgba(255,255,255,0.25);display:flex;align-items:center;
            justify-content:center;font-size:0.5rem;color:rgba(255,255,255,0.5);
            font-weight:600;">{pos}</div>'''

    return f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;overflow:hidden;}}
.field{{
    position:relative;width:100%;height:620px;border-radius:20px;overflow:hidden;
    background:
        linear-gradient(180deg, {p}dd 0%, {s}bb 50%, {p}cc 100%),
        repeating-linear-gradient(0deg,
            rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 2px,
            transparent 2px, transparent 62px),
        linear-gradient(180deg, #1a5c2a 0%, #237a35 25%, #1a5c2a 50%, #237a35 75%, #1a5c2a 100%);
    background-blend-mode: normal, overlay, normal;
}}
.endzone{{
    position:absolute;left:0;right:0;height:7%;
    display:flex;align-items:center;justify-content:center;
}}
.endzone-top{{top:0;background:{p}99;border-bottom:2px solid rgba(255,255,255,0.15);}}
.endzone-bottom{{bottom:0;background:{s}66;border-top:1px solid rgba(255,255,255,0.08);}}
.endzone-text{{
    color:rgba(255,255,255,0.18);font-size:1.6rem;font-weight:900;
    text-transform:uppercase;letter-spacing:0.4em;
}}
.logo-watermark{{
    position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    width:220px;height:220px;opacity:0.07;
    background:url('{logo_url}') center/contain no-repeat;
    pointer-events:none;
}}
.yard-line{{position:absolute;left:6%;right:6%;height:1px;background:rgba(255,255,255,0.10);}}
.los{{position:absolute;left:4%;right:4%;height:2px;top:44%;background:rgba(255,255,255,0.22);}}
.los-label{{
    position:absolute;top:44%;left:50%;transform:translate(-50%,-140%);
    color:rgba(255,255,255,0.3);font-size:0.6rem;text-transform:uppercase;
    letter-spacing:0.2em;background:{p}dd;padding:2px 14px;border-radius:4px;
}}
.hash{{position:absolute;top:0;bottom:0;width:1px;background:rgba(255,255,255,0.04);}}
.ol-zone{{
    position:absolute;top:41%;left:30%;right:30%;
    display:flex;justify-content:center;gap:6px;z-index:1;
}}
.player{{
    position:absolute;transform:translateX(-50%);text-align:center;z-index:2;
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
    padding:10px 20px;border-radius:12px;min-width:120px;
    transition:transform 0.25s ease, box-shadow 0.25s ease;cursor:default;
}}
.player:hover{{transform:translateX(-50%) scale(1.08);}}
.player .pos{{
    color:rgba(255,255,255,0.50);font-size:0.6rem;text-transform:uppercase;
    letter-spacing:0.15em;margin-bottom:3px;
}}
.player .name{{
    color:#fff;font-weight:700;font-size:0.9rem;text-transform:uppercase;
    letter-spacing:0.04em;white-space:nowrap;
}}
.label-box{{
    position:absolute;bottom:14px;z-index:3;
    background:rgba(0,0,0,0.5);backdrop-filter:blur(8px);
    -webkit-backdrop-filter:blur(8px);
    padding:6px 18px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);
}}
</style></head><body>
<div class="field">
    <div class="endzone endzone-top"><span class="endzone-text">{abbr}</span></div>
    <div class="endzone endzone-bottom"><span class="endzone-text" style="opacity:0.6;font-size:0.9rem;">{team_name}</span></div>
    <div class="logo-watermark"></div>
    <div class="yard-line" style="top:16%;"></div>
    <div class="yard-line" style="top:28%;"></div>
    <div class="los"></div>
    <div class="los-label">Line of Scrimmage</div>
    <div class="yard-line" style="top:58%;"></div>
    <div class="yard-line" style="top:70%;"></div>
    <div class="yard-line" style="top:86%;"></div>
    <div class="hash" style="left:38%;"></div>
    <div class="hash" style="left:62%;"></div>
    <div class="ol-zone">{ol_circles}</div>
    {player_divs}
    <div class="label-box" style="left:16px;">
        <span style="color:{a};font-weight:700;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;">
            11 Personnel &middot; Spread</span>
    </div>
    <div class="label-box" style="right:16px;">
        <span style="color:rgba(255,255,255,0.5);font-weight:600;font-size:0.7rem;letter-spacing:0.08em;">
            OFFENSE 2024</span>
    </div>
</div>
</body></html>'''


@st.cache_data(show_spinner=False)
def fetch_team_locations() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/Sinbad311/CloudProject/master/NFL%20Stadium%20Latitude%20and%20Longtitude.csv"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            from io import StringIO
            loc = pd.read_csv(StringIO(r.text))
            loc = loc.rename(columns={"latitude": "lat", "longitude": "lon"})
            loc["team_norm"] = loc["team"].astype(str).str.upper().str.strip()
            return loc
    except Exception:
        pass

    data = {
        "team": [
            "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills", "Carolina Panthers",
            "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns", "Dallas Cowboys", "Denver Broncos",
            "Detroit Lions", "Green Bay Packers", "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
            "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
            "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants", "New York Jets",
            "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers", "Seattle Seahawks",
            "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"
        ],
        "lat": [33.527, 33.755, 39.278, 42.774, 35.225, 41.862, 39.095, 41.506, 32.747, 39.743, 42.340, 44.501, 29.684, 39.760, 30.323, 39.048, 36.090, 33.953, 33.953, 25.958, 44.973, 42.091, 29.951, 40.812, 40.812, 39.901, 40.446, 37.403, 47.595, 27.975, 36.166, 38.907],
        "lon": [-112.262, -84.401, -76.622, -78.787, -80.852, -87.616, -84.516, -81.699, -97.094, -105.020, -83.045, -88.062, -95.408, -86.163, -81.637, -94.483, -115.183, -118.339, -118.339, -80.238, -93.257, -71.264, -90.081, -74.074, -74.074, -75.167, -80.015, -121.970, -122.331, -82.503, -86.771, -76.864],
        "abbr": ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS"]
    }
    df_loc = pd.DataFrame(data)
    df_loc["team_norm"] = df_loc["team"].str.upper().str.strip()
    return df_loc


def normalize_team_key(team_value: str) -> str:
    if team_value is None:
        return ""
    t = str(team_value).strip().upper()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("WASHINGTON FOOTBALL TEAM", "WAS")
    t = t.replace("WASHINGTON COMMANDERS", "WAS")
    t = t.replace("LAS VEGAS RAIDERS", "LV")
    t = t.replace("LOS ANGELES RAMS", "LAR")
    t = t.replace("LOS ANGELES CHARGERS", "LAC")
    if len(t) in (2, 3) and t.isalpha():
        return t
    return t


@st.cache_data(show_spinner=False)
def fetch_espn_depth_chart(team_abbr: str) -> pd.DataFrame:
    slug = ESPN_TEAM_SLUG.get(team_abbr.upper())
    if not slug:
        return pd.DataFrame()
    url = f"https://www.espn.com/nfl/team/depth/_/name/{slug}"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            return tables[0] if tables else pd.DataFrame()
    except Exception as e:
        st.warning(f"Erreur ESPN : {e}")
    return pd.DataFrame()


def infer_offensive_formation_from_depth(depth: pd.DataFrame) -> str:
    if depth.empty:
        return "Formation non disponible."
    pos_col = depth.columns[0]
    positions = depth[pos_col].astype(str).str.upper()
    rb = positions.str.contains(r"\bRB\b").sum()
    wr = positions.str.contains(r"\bWR\b").sum()
    te = positions.str.contains(r"\bTE\b").sum()
    if wr >= 3 and rb >= 1:
        return "**11 personnel** (1 RB / 1 TE / 3 WR)"
    if te >= 2 and rb >= 1:
        return "**12 personnel** (1 RB / 2 TE)"
    if rb >= 2:
        return "**21 personnel** (2 RB)"
    return "Estimation non concluante"


def set_page(name: str):
    st.session_state["page"] = name


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0;">
        <span style="font-size: 2.5rem;">🏈</span>
        <h2 style="margin: 4px 0 0 0; font-size: 1.2rem; background: linear-gradient(135deg, #8ab4f8, #4285f4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">NFL Offense Analytics</h2>
        <p style="color: #5a7a9a; font-size: 0.75rem; margin:0;">2005 — 2024</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    PAGES = [
        "Accueil",
        "Donnees",
        "Fiche equipe",
        "Prevision des points",
        "Drivers des yards",
        "Qualite & diagnostics",
        "Simulateur de matchs",
    ]

    PAGE_ICONS = {
        "Accueil": "🏠",
        "Donnees": "📊",
        "Fiche equipe": "🏟️",
        "Prevision des points": "🎯",
        "Drivers des yards": "📈",
        "Qualite & diagnostics": "🧪",
        "Simulateur de matchs": "⚔️",
    }

    if "page" not in st.session_state:
        st.session_state.page = "Accueil"

    for p in PAGES:
        icon = PAGE_ICONS.get(p, "")
        if st.sidebar.button(
            f"{icon}  {p}",
            use_container_width=True,
            key=f"nav_{p}",
            type="primary" if st.session_state.page == p else "secondary",
        ):
            st.session_state.page = p
            st.rerun()

    st.markdown("---")
    data_path = st.text_input("Chemin CSV", str(DEFAULT_DATA))

page = st.session_state.page

df = load_data(data_path)
team_col, year_col = infer_time_columns(df)
points_guess, yards_guess = guess_target_columns(df)


# =========================================================
# SIMULATION HELPER
# =========================================================
def simulate_match(team1: str, team2: str, df: pd.DataFrame) -> dict:
    latest_year = df[year_col].max()
    t1_row = df[(df[team_col] == team1) & (df[year_col] == latest_year)].iloc[0]
    t2_row = df[(df[team_col] == team2) & (df[year_col] == latest_year)].iloc[0]

    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ["Pts", "Points", "score", "year", "Season", team_col, year_col]
        and not c.lower().startswith(('rank', 'id'))
    ]
    t1_data = {col: t1_row[col] for col in feature_cols}
    t2_data = {col: t2_row[col] for col in feature_cols}

    score1 = call_api_prediction(t1_data)
    score2 = call_api_prediction(t2_data)

    if score1 is not None and score2 is not None:
        return {
            'team1': team1, 'team2': team2,
            'team1_score': round(score1, 1), 'team2_score': round(score2, 1),
            'winner': team1 if score1 > score2 else team2,
            'point_diff': round(abs(score1 - score2), 1)
        }
    return {}


# =========================================================
# PAGE: ACCUEIL
# =========================================================
if page == "Accueil":
    st.markdown("""
    <div class="title-bar">
        <h1>NFL Offense Analytics</h1>
        <p>Exploration, prediction et analyse des performances offensives NFL (2005-2024)</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saisons", f"{df[year_col].nunique() if year_col else '—'}")
    c2.metric("Equipes", f"{df[team_col].nunique() if team_col else '—'}")
    c3.metric("Observations", f"{len(df):,}".replace(",", " "))
    c4.metric("Variables", f"{df.shape[1]}")

    st.markdown("")

    hero_path = ROOT / "assets" / "home.png"
    if hero_path.exists():
        _, c2, _ = st.columns([1, 2, 1])
        with c2:
            st.image(str(hero_path), use_container_width=True)

    st.markdown("### Acces rapide")
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("#### 📊 Donnees")
            st.caption("Filtres, apercu, valeurs manquantes, heatmap de correlations.")
            st.button("Explorer les donnees", use_container_width=True, on_click=set_page, args=("Donnees",), key="home_data")

    with c2:
        with st.container(border=True):
            st.markdown("#### 🎯 Prediction")
            st.caption("RandomForest, metriques RMSE/MAE/R², importances, simulateur what-if.")
            st.button("Faire une prediction", use_container_width=True, on_click=set_page, args=("Prevision des points",), key="home_pred")

    with c3:
        with st.container(border=True):
            st.markdown("#### ⚔️ Simulateur")
            st.caption("Simulez un match entre deux equipes et predisez le score final.")
            st.button("Lancer une simulation", use_container_width=True, on_click=set_page, args=("Simulateur de matchs",), key="home_sim")

    st.markdown("")
    c4, c5, c6 = st.columns(3)

    with c4:
        with st.container(border=True):
            st.markdown("#### 🏟️ Fiche equipe")
            st.caption("Profil, tendances, classement, carte et depth chart ESPN.")
            st.button("Voir une equipe", use_container_width=True, on_click=set_page, args=("Fiche equipe",), key="home_team")

    with c5:
        with st.container(border=True):
            st.markdown("#### 📈 Drivers des yards")
            st.caption("Variables qui expliquent le plus les yards : importances + correlations.")
            st.button("Analyser les drivers", use_container_width=True, on_click=set_page, args=("Drivers des yards",), key="home_yards")

    with c6:
        with st.container(border=True):
            st.markdown("#### 🧪 Qualite")
            st.caption("Duplicats, outliers IQR, distribution, export sample nettoye.")
            st.button("Diagnostics", use_container_width=True, on_click=set_page, args=("Qualite & diagnostics",), key="home_qa")


# =========================================================
# PAGE: DONNEES
# =========================================================
elif page == "Donnees":
    st.markdown("""
    <div class="title-bar">
        <h1>📊 Exploration des donnees</h1>
        <p>Filtres, apercu, valeurs manquantes et correlations</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{len(df):,}".replace(",", " "))
    c2.metric("Colonnes", f"{df.shape[1]}")
    c3.metric("Equipes", f"{df[team_col].nunique() if team_col else '—'}")
    c4.metric("Saisons", f"{df[year_col].nunique() if year_col else '—'}")

    with st.expander("Filtres", expanded=True):
        colA, colB, colC = st.columns(3)
        if team_col:
            teams = sorted(df[team_col].dropna().unique().tolist())
            team_sel = colA.selectbox("Equipe", ["(toutes)"] + teams)
        else:
            team_sel = "(toutes)"
        if year_col:
            years = sorted(df[year_col].dropna().unique().tolist())
            year_min, year_max = colB.select_slider("Saisons", options=years, value=(years[0], years[-1]))
        else:
            year_min, year_max = None, None
        n_rows = colC.slider("Lignes affichees", 10, 300, 50)

    view = df.copy()
    if team_col and team_sel != "(toutes)":
        view = view[view[team_col] == team_sel]
    if year_col and year_min is not None:
        view = view[(view[year_col] >= year_min) & (view[year_col] <= year_max)]

    st.dataframe(view.head(n_rows), use_container_width=True)

    st.markdown("### Valeurs manquantes")
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    miss_df = pd.DataFrame({"missing_%": missing.round(2), "dtype": df.dtypes.astype(str)})
    if missing.sum() == 0:
        st.success("Aucune valeur manquante dans le dataset !")
    else:
        st.dataframe(miss_df[miss_df["missing_%"] > 0], use_container_width=True)

    st.markdown("### Correlations")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 3:
        corr_cols = st.multiselect(
            "Colonnes pour la heatmap",
            options=num_cols,
            default=num_cols[:10] if len(num_cols) > 10 else num_cols,
        )
        if len(corr_cols) >= 3:
            plot_correlation_heatmap(df, corr_cols, "Matrice de correlation")
        else:
            st.info("Selectionne au moins 3 colonnes.")


# =========================================================
# PAGE: FICHE EQUIPE
# =========================================================
elif page == "Fiche equipe":
    if not team_col:
        st.error("Colonne equipe non detectee dans le CSV.")
        st.stop()

    teams = sorted(df[team_col].dropna().unique().tolist())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- Team selector (clean row) ---
    colA, colB, colC = st.columns(3)
    team_sel = colA.selectbox("Equipe", teams)
    points_col = colB.selectbox(
        "KPI Points", options=numeric_cols,
        index=(numeric_cols.index(points_guess) if points_guess in numeric_cols else 0),
    )
    yards_col = colC.selectbox(
        "KPI Yards", options=numeric_cols,
        index=(numeric_cols.index(yards_guess) if yards_guess in numeric_cols else min(1, len(numeric_cols) - 1)),
    )

    # --- Resolve team data ---
    team_data = match_team_to_nfl(team_sel)
    _primary = team_data["primary"] if team_data else "#1a73e8"
    _secondary = team_data["secondary"] if team_data else "#0d1b2a"

    # --- Branded header ---
    if team_data:
        subtitle = "NFL Offense Analytics · 2005 — 2024"
        st.markdown(render_team_header_html(team_sel, team_data, subtitle), unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="title-bar">
            <h1>🏟️ {team_sel}</h1>
            <p>Profil, tendances, classement et carte</p>
        </div>
        """, unsafe_allow_html=True)

    view = df[df[team_col] == team_sel].copy()

    if year_col and year_col in df.columns:
        years = sorted(df[year_col].dropna().unique().tolist())
        year_min, year_max = st.select_slider("Periode", options=years, value=(years[0], years[-1]))
        view = view[(view[year_col] >= year_min) & (view[year_col] <= year_max)]

    # --- KPI metrics with team-colored styling ---
    if year_col and year_col in view.columns and len(view) > 0:
        best_year = int(view.loc[view[points_col].idxmax(), year_col])
        best_points = float(view[points_col].max())
        mean_points = float(view[points_col].mean())
        mean_yards = float(view[yards_col].mean())

        # Inject team-colored metric override
        st.markdown(f"""
        <style>
        [data-testid="stMetric"] label {{ color: {_primary} !important; }}
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Moy. Points", f"{mean_points:.1f}")
        c2.metric("Moy. Yards", f"{mean_yards:.1f}")
        c3.metric("Meilleure saison", str(best_year))
        c4.metric("Record points", f"{best_points:.1f}")

    # --- FORMATION (compo offensive) ---
    if team_data:
        st.markdown("")
        st.markdown("### Composition offensive 2024")
        col_form, col_roster = st.columns([3, 2])

        with col_form:
            components.html(render_formation_html(team_sel, team_data), height=640)

        with col_roster:
            roster = team_data["roster"]
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, {_primary}22 0%, {_secondary}22 100%);
                        border:1px solid {_primary}44; border-radius:16px; padding:24px; height:100%;">
                <h4 style="color:{_primary}; margin:0 0 16px; text-transform:uppercase;
                           letter-spacing:0.08em; font-size:0.85rem;">Starters Offensifs</h4>
            """, unsafe_allow_html=True)

            for pos, name in roster.items():
                pos_colors = {
                    "QB": "#e31837", "RB": "#4285f4", "TE": "#fbbc04",
                    "WR1": "#34a853", "WR2": "#34a853", "WR3": "#34a853",
                }
                pc = pos_colors.get(pos, "#8ab4f8")
                display_pos = "WR" if pos.startswith("WR") else pos
                st.markdown(f"""
                <div style="display:flex; align-items:center; padding:10px 12px; margin-bottom:8px;
                            background:rgba(255,255,255,0.04); border-radius:10px;
                            border-left:3px solid {pc};">
                    <span style="color:{pc}; font-weight:700; font-size:0.75rem; min-width:36px;
                                text-transform:uppercase; letter-spacing:0.05em;">{display_pos}</span>
                    <span style="color:#e8eaed; font-weight:600; font-size:0.95rem; margin-left:12px;">
                        {name}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # --- Trend charts ---
    if year_col and year_col in view.columns and len(view) > 1:
        st.markdown("### Tendances historiques")
        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                view.sort_values(year_col), x=year_col, y=points_col,
                title=f"{team_sel} — Evolution {points_col}",
                markers=True,
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_traces(line=dict(color=_primary, width=3), marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                view.sort_values(year_col), x=year_col, y=yards_col,
                title=f"{team_sel} — Evolution {yards_col}",
                markers=True,
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_traces(line=dict(color=_secondary if _secondary != "#000000" else _primary, width=3), marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)

    # --- Ranking ---
    if year_col and year_col in df.columns:
        st.markdown("### Classement par saison")
        season_df = df.dropna(subset=[year_col]).copy()
        season_df["rank"] = season_df.groupby(year_col)[points_col].rank(ascending=False, method="min")
        team_season = season_df[season_df[team_col] == team_sel][[year_col, points_col, yards_col, "rank"]].sort_values(year_col)
        team_season["rank"] = team_season["rank"].astype(int)

        fig = px.bar(
            team_season, x=year_col, y="rank",
            title=f"{team_sel} — Classement NFL (1 = meilleur)",
            color="rank", color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- Map + ESPN side-by-side ---
    st.markdown("### Localisation & Depth Chart")
    col_map, col_espn = st.columns(2)

    with col_map:
        loc = fetch_team_locations()
        team_key = normalize_team_key(team_sel)
        map_row = pd.DataFrame()
        if not loc.empty:
            sel_norm = team_sel.upper().strip()
            map_row = loc[(loc["team_norm"] == sel_norm) | (loc.get("abbr") == team_key)]
        if not map_row.empty:
            st.map(map_row, zoom=5)
        else:
            st.warning("Coordonnees indisponibles.")

    with col_espn:
        default_abbr = (team_data["abbr"] if team_data else "")
        if default_abbr not in ESPN_TEAM_SLUG:
            default_abbr = team_key if team_key in ESPN_TEAM_SLUG else ""
        abbr = st.text_input("Abreviation ESPN (ex: DAL, NE, KC)", value=default_abbr)
        if abbr.strip():
            depth = fetch_espn_depth_chart(abbr.strip().upper())
            if depth.empty:
                st.info("Depth chart indisponible.")
            else:
                st.markdown(f"Formation estimee : {infer_offensive_formation_from_depth(depth)}")
                st.dataframe(depth, use_container_width=True)


# =========================================================
# PAGE: PREVISION DES POINTS
# =========================================================
elif page == "Prevision des points":
    st.markdown("""
    <div class="title-bar">
        <h1>🎯 Prevision des points</h1>
        <p>Modele RandomForest, metriques, importances et simulateur what-if</p>
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("Pas de colonnes numeriques.")
        st.stop()

    points_target = st.selectbox(
        "Colonne cible (points)", options=numeric_cols,
        index=(numeric_cols.index(points_guess) if points_guess in numeric_cols else 0),
    )

    exclude = [points_target]
    if team_col:
        exclude.append(team_col)
    if year_col:
        exclude.append(year_col)
    feat_cols = numeric_feature_columns(df, exclude=exclude)

    st.markdown(f'<span class="stat-badge">{len(feat_cols)} features</span> <span class="stat-badge">Cible : {points_target}</span>', unsafe_allow_html=True)

    with st.expander("Reglages du modele", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        test_years = col1.slider("Holdout (annees)", 1, 8, 3)
        n_estimators = col2.slider("Arbres", 100, 1200, 600, step=50)
        max_depth = col3.selectbox("Max depth", options=[None, 5, 10, 15, 20, 30], index=0)
        random_state = col4.number_input("Random state", value=42, step=1)
        use_saved = st.checkbox("Charger modele sauvegarde si dispo", value=True)

    model_path = MODELS_DIR / f"rf_points__{points_target}.pkl"
    model = load_model(model_path) if use_saved else None

    if st.button("Entrainer / re-entrainer", use_container_width=True):
        model = None

    if model is None:
        with st.spinner("Entrainement en cours..."):
            model, metrics, X_train, X_test, y_train, y_test = train_model(
                df, target_col=points_target, feature_cols=feat_cols,
                test_years=test_years, year_col=year_col,
                random_state=int(random_state), n_estimators=int(n_estimators), max_depth=max_depth,
            )
            if joblib is not None:
                save_model(model, model_path)
            (REPORTS_DIR / "metrics_points.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        st.success("Modele entraine avec succes !")
        nice_metric_row(metrics)

        pred = model.predict(X_test.fillna(X_test.median(numeric_only=True)))

        col1, col2 = st.columns(2)
        with col1:
            plot_scatter_plotly(y_test, pred, "Predictions vs Valeurs reelles")
        with col2:
            fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
            plot_importance_bar(fi, "Importances RandomForest", top_n=15)

        st.markdown("### Importance par permutation")
        with st.spinner("Calcul..."):
            perm = permutation_importance(
                model, X_test.fillna(X_test.median(numeric_only=True)), y_test,
                n_repeats=10, random_state=int(random_state), n_jobs=-1,
            )
        perm_imp = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)
        plot_importance_bar(perm_imp, "Permutation Importance", top_n=15)
    else:
        st.info("Modele charge depuis `models/`. Cliquez sur le bouton pour re-entrainer.")

    st.markdown("---")
    st.markdown("### Simulateur what-if")

    if model is None:
        st.warning("Entrainez un modele d'abord.")
    else:
        chosen_vars = st.multiselect(
            "Variables a piloter", options=feat_cols,
            default=feat_cols[:6] if len(feat_cols) >= 6 else feat_cols,
        )
        if chosen_vars:
            X_base = df[feat_cols].copy().fillna(df[feat_cols].median(numeric_only=True))
            med = X_base.median(numeric_only=True)

            inputs = {}
            cols = st.columns(3)
            for i, v in enumerate(chosen_vars):
                vmin = float(X_base[v].quantile(0.05))
                vmax = float(X_base[v].quantile(0.95))
                vdefault = float(med[v])
                inputs[v] = cols[i % 3].slider(v, vmin, vmax, vdefault)

            row = med.copy()
            for k, val in inputs.items():
                row[k] = val
            X_one = pd.DataFrame([row], columns=feat_cols)
            y_hat = float(model.predict(X_one)[0])
            st.metric("Points predits", f"{y_hat:.2f}")


# =========================================================
# PAGE: DRIVERS DES YARDS
# =========================================================
elif page == "Drivers des yards":
    st.markdown("""
    <div class="title-bar">
        <h1>📈 Drivers des yards</h1>
        <p>Quelles variables expliquent le plus les yards totaux ?</p>
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("Pas de colonnes numeriques.")
        st.stop()

    yards_target = st.selectbox(
        "Colonne cible (yards)", options=numeric_cols,
        index=(numeric_cols.index(yards_guess) if yards_guess in numeric_cols else 0),
    )

    exclude = [yards_target]
    if team_col:
        exclude.append(team_col)
    if year_col:
        exclude.append(year_col)
    feat_cols = numeric_feature_columns(df, exclude=exclude)

    with st.expander("Reglages", expanded=False):
        col1, col2, col3 = st.columns(3)
        test_years = col1.slider("Holdout (annees)", 1, 8, 3, key="yards_holdout")
        n_estimators = col2.slider("Arbres", 100, 1200, 600, step=50, key="yards_trees")
        random_state = col3.number_input("Random state", value=42, step=1, key="yards_rs")

    if st.button("Calculer les drivers", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            model, metrics, X_train, X_test, y_train, y_test = train_model(
                df, target_col=yards_target, feature_cols=feat_cols,
                test_years=test_years, year_col=year_col,
                random_state=int(random_state), n_estimators=int(n_estimators), max_depth=None,
            )
            pred = model.predict(X_test.fillna(X_test.median(numeric_only=True)))

        st.success("Analyse terminee !")
        nice_metric_row(metrics)

        col1, col2 = st.columns(2)
        with col1:
            plot_scatter_plotly(y_test, pred, "Predictions vs Valeurs reelles (yards)")
        with col2:
            fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
            plot_importance_bar(fi, "Importances RandomForest — Yards", top_n=15)

        st.markdown("### Permutation importance")
        with st.spinner("Calcul..."):
            perm = permutation_importance(
                model, X_test.fillna(X_test.median(numeric_only=True)), y_test,
                n_repeats=10, random_state=int(random_state), n_jobs=-1,
            )
        perm_imp = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)
        plot_importance_bar(perm_imp, "Permutation Importance — Yards", top_n=15)

        st.markdown("### Top correlations avec la cible")
        corr = df[feat_cols + [yards_target]].corr(numeric_only=True)[yards_target].drop(yards_target)
        corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)

        fig = px.bar(
            x=corr.head(15).values, y=corr.head(15).index, orientation="h",
            title="Correlation avec la cible", labels={"x": "Correlation", "y": "Variable"},
            color=corr.head(15).values, color_continuous_scale="RdBu_r",
        )
        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        out = {
            "target": yards_target,
            "metrics": metrics,
            "top_rf": fi.head(25).to_dict(),
            "top_perm": perm_imp.head(25).to_dict(),
        }
        (REPORTS_DIR / "yards_drivers.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


# =========================================================
# PAGE: SIMULATEUR DE MATCHS
# =========================================================
elif page == "Simulateur de matchs":
    st.markdown("""
    <div class="title-bar">
        <h1>⚔️ Simulateur de matchs NFL</h1>
        <p>Predisez le score d'un match a partir des statistiques offensives</p>
    </div>
    """, unsafe_allow_html=True)

    # Backend check
    try:
        requests.get(url="http://127.0.0.1:8000/", timeout=1)
        st.success("Backend FastAPI connecte")
    except Exception:
        st.error("Backend FastAPI hors-ligne — Lancez `uvicorn api:app` dans un terminal")

    if not team_col:
        st.error("Colonne equipe non detectee.")
        st.stop()

    teams = sorted(df[team_col].unique())

    col1, _, col2 = st.columns([5, 1, 5])

    with col1:
        st.markdown("### 🏠 Domicile")
        team1 = st.selectbox("Equipe domicile", teams, index=0)
        if year_col:
            latest = df[year_col].max()
            t1_stats = df[(df[team_col] == team1) & (df[year_col] == latest)]
            if not t1_stats.empty and points_guess:
                c1, c2 = st.columns(2)
                c1.metric("Points/match", f"{t1_stats[points_guess].values[0]:.1f}")
                if yards_guess:
                    c2.metric("Yards/match", f"{t1_stats[yards_guess].values[0]:.1f}")

    with col2:
        st.markdown("### ✈️ Exterieur")
        away_teams = [t for t in teams if t != team1]
        team2 = st.selectbox("Equipe exterieur", away_teams, index=min(1, len(away_teams) - 1))
        if year_col:
            t2_stats = df[(df[team_col] == team2) & (df[year_col] == latest)]
            if not t2_stats.empty and points_guess:
                c1, c2 = st.columns(2)
                c1.metric("Points/match", f"{t2_stats[points_guess].values[0]:.1f}")
                if yards_guess:
                    c2.metric("Yards/match", f"{t2_stats[yards_guess].values[0]:.1f}")

    st.markdown("")
    if st.button("Lancer la simulation", use_container_width=True, type="primary"):
        with st.spinner("Prediction via l'API..."):
            result = simulate_match(team1, team2, df)
            if result:
                st.markdown("---")
                c1, c2, c3 = st.columns([2, 1, 2])
                with c1:
                    score_color = "#34a853" if result['team1_score'] >= result['team2_score'] else "#ea4335"
                    st.markdown(f"""
                    <div style="text-align:center; padding:20px;">
                        <h2 style="color: {score_color};">{result['team1']}</h2>
                        <h1 style="font-size:3rem; color: {score_color};">{result['team1_score']}</h1>
                        <p style="color:#8899aa;">points predits</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown("""
                    <div style="text-align:center; padding:40px 0;">
                        <h1 style="font-size:2rem; color:#5a7a9a;">VS</h1>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    score_color = "#34a853" if result['team2_score'] >= result['team1_score'] else "#ea4335"
                    st.markdown(f"""
                    <div style="text-align:center; padding:20px;">
                        <h2 style="color: {score_color};">{result['team2']}</h2>
                        <h1 style="font-size:3rem; color: {score_color};">{result['team2_score']}</h1>
                        <p style="color:#8899aa;">points predits</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.balloons()
                st.success(f"Victoire predite : **{result['winner']}** (+{result['point_diff']} pts)")
            else:
                st.error("Erreur lors de la simulation. Verifiez que le backend est en marche.")


# =========================================================
# PAGE: QUALITE & DIAGNOSTICS
# =========================================================
elif page == "Qualite & diagnostics":
    st.markdown("""
    <div class="title-bar">
        <h1>🧪 Qualite & diagnostics</h1>
        <p>Verification de la qualite des donnees</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    dup = int(df.duplicated().sum())
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_total = int(df.isna().sum().sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Doublons", f"{dup}")
    c2.metric("Valeurs manquantes", f"{missing_total}")
    c3.metric("Colonnes numeriques", f"{len(num_cols)}")

    st.markdown("### Distribution & outliers")
    if num_cols:
        col = st.selectbox("Colonne", num_cols)
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_count = int(((s < lo) | (s > hi)).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Q1", f"{q1:.2f}")
        c2.metric("Q3", f"{q3:.2f}")
        c3.metric("IQR", f"{iqr:.2f}")
        c4.metric("Outliers", f"{out_count}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                s, nbins=30, title=f"Distribution — {col}",
                labels={"value": col, "count": "Frequence"},
                color_discrete_sequence=["#4285f4"],
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                df, y=col, title=f"Box plot — {col}",
                color_discrete_sequence=["#4285f4"],
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if st.button("Exporter un sample nettoye"):
        out = df.copy()
        num = out.select_dtypes(include=[np.number]).columns
        out[num] = out[num].fillna(out[num].median(numeric_only=True))
        out_path = REPORTS_DIR / "sample_cleaned.csv"
        out.head(500).to_csv(out_path, index=False)
        st.success(f"Exporte : {out_path}")
