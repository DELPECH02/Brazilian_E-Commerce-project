# app_streamlit.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import requests

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
    page_title="NFL Offense Analytics — Prédiction & Insights",
    page_icon="🏈",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "raw" / "nfl_offense.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)


# =========================================================
# HELPERS
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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


def plot_scatter(y_true: pd.Series, y_pred: np.ndarray, title: str):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.set_title(title)
    ax.set_xlabel("Vrai")
    ax.set_ylabel("Prédit")
    st.pyplot(fig, clear_figure=True)


def plot_top_bar(series: pd.Series, title: str, top_n: int = 20):
    s = series.sort_values(ascending=False).head(top_n)[::-1]  # horizontal sorted
    fig, ax = plt.subplots()
    ax.barh(s.index.astype(str), s.values)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    st.pyplot(fig, clear_figure=True)


def correlation_heatmap(df: pd.DataFrame, cols: List[str], title: str):
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(corr.values)
    ax.set_title(title)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


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

    # fill missing in features
    X = X.fillna(X.median(numeric_only=True))

    # time-based split if possible
    if year_col and year_col in data.columns and pd.api.types.is_numeric_dtype(data[year_col]):
        years_sorted = sorted(data[year_col].dropna().unique())
        if len(years_sorted) > test_years:
            cutoff = years_sorted[-test_years]
            train_idx = data[year_col] < cutoff
            test_idx = data[year_col] >= cutoff
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=max_depth,
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
# WEB ENRICHMENT: ESPN DEPTH CHART + TEAM MAP
# =========================================================
# ESPN depth chart URLs: https://www.espn.com/nfl/team/depth/_/name/<slug>
ESPN_TEAM_SLUG = {
    "ARI": "ari", "ATL": "atl", "BAL": "bal", "BUF": "buf", "CAR": "car", "CHI": "chi",
    "CIN": "cin", "CLE": "cle", "DAL": "dal", "DEN": "den", "DET": "det", "GB": "gb",
    "HOU": "hou", "IND": "ind", "JAX": "jax", "KC": "kc", "LV": "lv", "LAC": "lac",
    "LAR": "lar", "MIA": "mia", "MIN": "min", "NE": "ne", "NO": "no", "NYG": "nyg",
    "NYJ": "nyj", "PHI": "phi", "PIT": "pit", "SEA": "sea", "SF": "sf", "TB": "tb",
    "TEN": "ten", "WAS": "wsh", "WSH": "wsh",
}

@st.cache_data(show_spinner=False)
def fetch_team_locations() -> pd.DataFrame:
    """
    Dataset public (community): coordonnées de stades NFL.
    Si pas d'internet -> DF vide.
    """
    url = "https://raw.githubusercontent.com/Sinbad311/CloudProject/master/NFL%20Stadium%20Latitude%20and%20Longtitude.csv"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        from io import StringIO
        loc = pd.read_csv(StringIO(r.text))
        loc.columns = [c.strip().lower() for c in loc.columns]

        # rename to common
        if "latitude" in loc.columns:
            loc = loc.rename(columns={"latitude": "lat"})
        if "longitude" in loc.columns:
            loc = loc.rename(columns={"longitude": "lon"})

        # make sure expected cols exist
        for c in ["team", "lat", "lon"]:
            if c not in loc.columns:
                return pd.DataFrame(columns=["team", "lat", "lon"])

        loc["team_norm"] = loc["team"].astype(str).str.upper().str.strip()
        return loc[["team", "lat", "lon", "team_norm"]]
    except Exception:
        return pd.DataFrame(columns=["team", "lat", "lon", "team_norm"])


def normalize_team_key(team_value: str) -> str:
    if team_value is None:
        return ""
    t = str(team_value).strip().upper()
    t = re.sub(r"\s+", " ", t)

    # Some common name swaps
    t = t.replace("WASHINGTON FOOTBALL TEAM", "WAS")
    t = t.replace("WASHINGTON COMMANDERS", "WAS")
    t = t.replace("LAS VEGAS RAIDERS", "LV")
    t = t.replace("LOS ANGELES RAMS", "LAR")
    t = t.replace("LOS ANGELES CHARGERS", "LAC")

    # If already abbreviation-like
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
        tables = pd.read_html(url)
        # Often first table = offense, but ESPN may change
        return tables[0] if tables else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def infer_offensive_formation_from_depth(depth: pd.DataFrame) -> str:
    """
    Heuristique simple pour donner un 'label formation' qui fait pro.
    """
    if depth.empty:
        return "Formation non disponible (pas de données web)."

    # Try to find a position column (often first col)
    pos_col = depth.columns[0]
    positions = depth[pos_col].astype(str).str.upper()

    rb = positions.str.contains(r"\bRB\b").sum()
    wr = positions.str.contains(r"\bWR\b").sum()
    te = positions.str.contains(r"\bTE\b").sum()

    if wr >= 3 and rb >= 1:
        return "Formation probable : **11 personnel** (1 RB / 1 TE / 3 WR) — estimation"
    if te >= 2 and rb >= 1:
        return "Formation probable : **12 personnel** (1 RB / 2 TE) — estimation"
    if rb >= 2:
        return "Formation probable : **21 personnel** (2 RB) — estimation"
    return "Formation : estimation (voir titulaires ci-dessous)"

def set_page(name: str):
    st.session_state["page"] = name

def page_card(title: str, desc: str, cta: str, target_page: str, icon: str = "➡️"):
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(desc)
        st.button(f"{icon} {cta}", use_container_width=True, on_click=set_page, args=(target_page,))



# =========================================================
# SIDEBAR NAV
# =========================================================
st.sidebar.title("🏈 NFL Offense Analytics")

PAGES = [
    "🏠 Accueil",
    "📄 Données",
    "🏟️ Fiche équipe",
    "🎯 Prévision des points",
    "📈 Impact des variables sur les yards",
    "🧪 Qualité & diagnostics",
]

if "page" not in st.session_state:
    st.session_state.page = "🏠 Accueil"

selected = st.sidebar.radio(
    "Navigation",
    PAGES,
    index=PAGES.index(st.session_state.page),
)

# synchro radio → état global
st.session_state.page = selected
page = st.session_state.page



st.sidebar.markdown("---")
data_path = st.sidebar.text_input("Chemin CSV", str(DEFAULT_DATA))
df = load_data(data_path)

team_col, year_col = infer_time_columns(df)
points_guess, yards_guess = guess_target_columns(df)

st.sidebar.caption("Conseil : garde le CSV dans `data/raw/` (repo propre).")


# =========================================================
# HEADER
# =========================================================
st.title("NFL Offense Analytics (2005–2024)")
st.caption("Projet portfolio Data Analyst/ML : exploration, prévision, et explication des drivers.")

# =========================================================
# PAGE 0 — HOME
# =========================================================
if page == "🏠 Accueil":
    st.subheader("Bienvenue 👋")
    st.write(
        "Cette application permet d’explorer les attaques NFL (2005–2024), "
        "de **prédire les points** à partir des statistiques, et d’expliquer les **drivers** "
        "derrière la performance (yards, efficacité, turnovers, etc.)."
    )

    # Image (optionnelle)
    hero_path = ROOT / "assets" / "home.png"
    # Image centrée
    if hero_path.exists():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(str(hero_path), width=650)


    else:
        st.info("Ajoute une image dans `assets/home.png` pour afficher une bannière ici.")

    st.markdown("### 🚀 Accès rapide")
    c1, c2, c3 = st.columns(3)

    with c1:
        page_card(
            "📄 Données",
            "Aperçu du dataset, filtres équipe/saison, valeurs manquantes, heatmap de corrélations.",
            "Explorer les données",
            "📄 Données",
            icon="🔎",
        )

    with c2:
        page_card(
            "🎯 Prévision des points",
            "Entraîne un RandomForest, métriques RMSE/MAE/R², importances + simulateur what-if.",
            "Faire une prédiction",
            "🎯 Prévision des points",
            icon="🧠",
        )

    with c3:
        page_card(
            "🏟️ Fiche équipe",
            "Storytelling par équipe : profil, tendances, carte, depth chart ESPN (si web dispo).",
            "Voir une équipe",
            "🏟️ Fiche équipe",
            icon="🏈",
        )

    st.markdown("### 📌 Les autres modules")
    c4, c5 = st.columns(2)
    with c4:
        page_card(
            "📈 Drivers des yards",
            "Analyse des variables qui expliquent le plus les yards : importances + corrélations.",
            "Analyser les drivers",
            "📈 Impact des variables sur les yards",
            icon="📊",
        )
    with c5:
        page_card(
            "🧪 Qualité & diagnostics",
            "Checks qualité : duplicats, outliers IQR, export d’un sample nettoyé.",
            "Voir les diagnostics",
            "🧪 Qualité & diagnostics",
            icon="✅",
        )

    st.markdown("---")
    st.caption("Astuce : commence par la page 📄 Données pour vérifier la qualité et les colonnes détectées.")

# =========================================================
# PAGE 1 — DATA
# =========================================================
elif page == "📄 Données":
    st.subheader("Aperçu & exploration")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{len(df):,}".replace(",", " "))
    c2.metric("Colonnes", f"{df.shape[1]}")
    c3.metric("Équipes (si détecté)", f"{df[team_col].nunique() if team_col else '—'}")
    c4.metric("Saisons (si détecté)", f"{df[year_col].nunique() if year_col else '—'}")

    with st.expander("🔎 Filtres", expanded=True):
        colA, colB, colC = st.columns(3)

        if team_col:
            teams = sorted(df[team_col].dropna().unique().tolist())
            team_sel = colA.selectbox("Équipe", ["(toutes)"] + teams)
        else:
            team_sel = "(toutes)"
            colA.info("Colonne équipe non détectée.")

        if year_col:
            years = sorted(df[year_col].dropna().unique().tolist())
            year_min, year_max = colB.select_slider(
                "Plage de saisons",
                options=years,
                value=(years[0], years[-1]),
            )
        else:
            year_min, year_max = None, None
            colB.info("Colonne saison/année non détectée.")

        n_rows = colC.slider("Nombre de lignes affichées", 10, 300, 50)

    view = df.copy()
    if team_col and team_sel != "(toutes)":
        view = view[view[team_col] == team_sel]
    if year_col and year_min is not None:
        view = view[(view[year_col] >= year_min) & (view[year_col] <= year_max)]

    st.dataframe(view.head(n_rows), use_container_width=True)

    st.markdown("### Valeurs manquantes & types")
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    st.dataframe(
        pd.DataFrame({"missing_%": missing.round(2), "dtype": df.dtypes.astype(str)}),
        use_container_width=True,
    )

    st.markdown("### Corrélations (numériques)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 3:
        corr_cols = st.multiselect(
            "Choisis des colonnes pour la heatmap",
            options=num_cols,
            default=num_cols[:12] if len(num_cols) > 12 else num_cols,
        )
        if len(corr_cols) >= 3:
            correlation_heatmap(df, corr_cols, "Heatmap corrélation (sélection)")
        else:
            st.info("Sélectionne au moins 3 colonnes numériques.")
    else:
        st.info("Pas assez de colonnes numériques pour afficher une heatmap.")


# =========================================================
# PAGE 2 — TEAM PAGE
# =========================================================
elif page == "🏟️ Fiche équipe":
    st.subheader("Fiche équipe — présentation, positionnement, carte & titulaires")

    if not team_col:
        st.error("Je ne détecte pas de colonne équipe (team/franchise) dans ton CSV.")
        st.stop()

    teams = sorted(df[team_col].dropna().unique().tolist())
    team_sel = st.selectbox("Choisir une équipe", teams)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("Je n’ai pas trouvé de colonnes numériques : impossible de calculer le positionnement.")
        st.stop()

    # Pick KPI columns
    colA, colB, colC = st.columns(3)
    points_col = colA.selectbox(
        "KPI Points (pour l’analyse)",
        options=numeric_cols,
        index=(numeric_cols.index(points_guess) if points_guess in numeric_cols else 0),
    )
    yards_col = colB.selectbox(
        "KPI Yards (pour l’analyse)",
        options=numeric_cols,
        index=(numeric_cols.index(yards_guess) if yards_guess in numeric_cols else min(1, len(numeric_cols)-1)),
    )

    view = df[df[team_col] == team_sel].copy()

    if year_col and year_col in df.columns:
        years = sorted(df[year_col].dropna().unique().tolist())
        year_min, year_max = colC.select_slider("Période", options=years, value=(years[0], years[-1]))
        view = view[(view[year_col] >= year_min) & (view[year_col] <= year_max)]
    else:
        year_min, year_max = None, None
        colC.info("Pas de colonne saison/année détectée.")

    st.markdown("### 📝 Présentation (auto-générée)")
    if year_col and year_col in view.columns and len(view) > 0:
        best_year = int(view.loc[view[points_col].idxmax(), year_col])
        best_points = float(view[points_col].max())
        mean_points = float(view[points_col].mean())
        mean_yards = float(view[yards_col].mean())
        st.write(
            f"**{team_sel}** — Sur la période sélectionnée, l’équipe affiche en moyenne "
            f"**{mean_points:.2f}** ({points_col}) et **{mean_yards:.2f}** ({yards_col}). "
            f"Son meilleur pic de {points_col} est atteint en **{best_year}** avec **{best_points:.2f}**."
        )
    else:
        st.write(
            f"**{team_sel}** — Profil basé sur ton dataset (sur la sélection actuelle). "
            f"Ajoute une colonne year/season pour un storytelling saison par saison."
        )

    st.markdown("### 🏁 Positionnement par saison (classement vs ligue)")
    if year_col and year_col in df.columns:
        season_df = df.dropna(subset=[year_col]).copy()

        # rank points per season (1=best)
        season_df["__rank_points"] = season_df.groupby(year_col)[points_col].rank(ascending=False, method="min")
        season_df["__n"] = season_df.groupby(year_col)[points_col].transform("count")

        team_season = season_df[season_df[team_col] == team_sel][
            [year_col, points_col, yards_col, "__rank_points", "__n"]
        ].copy()
        team_season = team_season.sort_values(year_col)
        team_season["rank_points"] = team_season["__rank_points"].astype(int)
        team_season["teams_in_season"] = team_season["__n"].astype(int)

        st.dataframe(
            team_season[[year_col, points_col, yards_col, "rank_points", "teams_in_season"]],
            use_container_width=True
        )

        # trend
        fig, ax = plt.subplots()
        ax.plot(team_season[year_col], team_season[points_col])
        ax.set_title(f"{team_sel} — évolution {points_col}")
        ax.set_xlabel("Saison")
        ax.set_ylabel(points_col)
        st.pyplot(fig, clear_figure=True)

        # rank trend
        fig, ax = plt.subplots()
        ax.plot(team_season[year_col], team_season["rank_points"])
        ax.invert_yaxis()
        ax.set_title(f"{team_sel} — classement (1 = meilleur)")
        ax.set_xlabel("Saison")
        ax.set_ylabel("Rang")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Pas de colonne year/season détectée → impossible de faire un classement par saison.")

    st.markdown("### 🗺️ Carte : emplacement de l’équipe (USA)")
    st.caption("Coordonnées issues d’un dataset public (si internet disponible).")

    loc = fetch_team_locations()
    team_key = normalize_team_key(team_sel)

    # try to match coordinates
    map_row = pd.DataFrame()
    if not loc.empty:
        # exact
        map_row = loc[loc["team_norm"] == team_key]
        # fallback contains
        if map_row.empty:
            map_row = loc[loc["team_norm"].str.contains(team_key, na=False)]

    if map_row is not None and not map_row.empty:
        map_df = pd.DataFrame({
            "lat": [float(map_row.iloc[0]["lat"])],
            "lon": [float(map_row.iloc[0]["lon"])],
            "team": [team_sel],
        })
        st.map(map_df, latitude="lat", longitude="lon", size=80)
    else:
        st.warning("Coordonnées indisponibles (internet bloqué ou nom d’équipe non reconnu).")

    st.markdown("### 👥 Titulaires / formation (via ESPN depth chart)")
    st.caption("Si l’accès web est bloqué, cette section peut rester vide.")

    # Let user choose an ESPN abbreviation
    default_abbr = team_key if team_key in ESPN_TEAM_SLUG else ""
    abbr = st.text_input("Abréviation ESPN (ex: DAL, NE, KC…)", value=default_abbr)

    if abbr.strip():
        depth = fetch_espn_depth_chart(abbr.strip().upper())
        if depth.empty:
            st.info("Depth chart indisponible (abréviation invalide ou pas d’accès internet).")
        else:
            st.write(infer_offensive_formation_from_depth(depth))
            st.dataframe(depth, use_container_width=True)
    else:
        st.info("Entre une abréviation ESPN pour charger le depth chart.")


# =========================================================
# PAGE 3 — POINTS PREDICTION
# =========================================================
elif page == "🎯 Prévision des points":
    st.subheader("Prédire les points + expliquer le modèle (et faire du what-if)")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("Pas de colonnes numériques → impossible de faire un modèle.")
        st.stop()

    points_target = st.selectbox(
        "Colonne cible (points)",
        options=numeric_cols,
        index=(numeric_cols.index(points_guess) if points_guess in numeric_cols else 0),
    )

    exclude = [points_target]
    if team_col:
        exclude.append(team_col)
    if year_col:
        exclude.append(year_col)

    feat_cols = numeric_feature_columns(df, exclude=exclude)
    st.caption(f"{len(feat_cols)} variables numériques utilisées pour prédire `{points_target}`.")

    with st.expander("⚙️ Réglages", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        test_years = col1.slider("Holdout (années)", 1, 8, 3)
        n_estimators = col2.slider("Arbres", 100, 1200, 600, step=50)
        max_depth = col3.selectbox("Max depth", options=[None, 5, 10, 15, 20, 30], index=0)
        random_state = col4.number_input("Random state", value=42, step=1)
        use_saved = st.checkbox("Charger modèle sauvegardé si dispo", value=True)

    model_path = MODELS_DIR / f"rf_points__{points_target}.pkl"
    model = load_model(model_path) if use_saved else None

    if st.button("🚀 Entraîner / re-entraîner"):
        model = None  # force retrain

    if model is None:
        with st.spinner("Entraînement…"):
            model, metrics, X_train, X_test, y_train, y_test = train_model(
                df,
                target_col=points_target,
                feature_cols=feat_cols,
                test_years=test_years,
                year_col=year_col,
                random_state=int(random_state),
                n_estimators=int(n_estimators),
                max_depth=max_depth,
            )
            if joblib is not None:
                save_model(model, model_path)

            (REPORTS_DIR / "metrics_points.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )

        st.success("Modèle prêt ✅")
        nice_metric_row(metrics)

        pred = model.predict(X_test.fillna(X_test.median(numeric_only=True)))
        plot_scatter(y_test, pred, "Prédit vs Vrai (holdout)")

        st.markdown("### Variables les plus importantes (RandomForest)")
        fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        plot_top_bar(fi, "Top importances — Points", top_n=25)

        st.markdown("### Importance par permutation (plus robuste)")
        with st.spinner("Permutation importance…"):
            perm = permutation_importance(
                model,
                X_test.fillna(X_test.median(numeric_only=True)),
                y_test,
                n_repeats=10,
                random_state=int(random_state),
                n_jobs=-1,
            )
        perm_imp = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)
        plot_top_bar(perm_imp, "Permutation importance — Points", top_n=25)
    else:
        st.info("Modèle chargé depuis `models/` (coche re-entraîner si tu veux recalculer).")

    st.markdown("---")
    st.markdown("## 🎛️ Simulateur what-if : prédire des points avec tes valeurs")

    if model is None:
        st.warning("Entraîne ou charge un modèle d’abord.")
    else:
        chosen_vars = st.multiselect(
            "Variables à piloter",
            options=feat_cols,
            default=feat_cols[:8] if len(feat_cols) >= 8 else feat_cols,
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
            st.metric("Points prédits", f"{y_hat:.2f}")


# =========================================================
# PAGE 4 — YARDS DRIVERS
# =========================================================
elif page == "📈 Impact des variables sur les yards":
    st.subheader("Drivers des yards : quelles variables expliquent le plus ?")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("Pas de colonnes numériques → impossible de faire l’analyse.")
        st.stop()

    yards_target = st.selectbox(
        "Colonne cible (yards)",
        options=numeric_cols,
        index=(numeric_cols.index(yards_guess) if yards_guess in numeric_cols else 0),
    )

    exclude = [yards_target]
    if team_col:
        exclude.append(team_col)
    if year_col:
        exclude.append(year_col)

    feat_cols = numeric_feature_columns(df, exclude=exclude)

    with st.expander("⚙️ Réglages", expanded=True):
        col1, col2, col3 = st.columns(3)
        test_years = col1.slider("Holdout (années)", 1, 8, 3, key="yards_holdout")
        n_estimators = col2.slider("Arbres", 100, 1200, 600, step=50, key="yards_trees")
        random_state = col3.number_input("Random state", value=42, step=1, key="yards_rs")

    if st.button("🔍 Calculer les drivers (yards)"):
        with st.spinner("Entraînement + calcul importances…"):
            model, metrics, X_train, X_test, y_train, y_test = train_model(
                df,
                target_col=yards_target,
                feature_cols=feat_cols,
                test_years=test_years,
                year_col=year_col,
                random_state=int(random_state),
                n_estimators=int(n_estimators),
                max_depth=None,
            )

            pred = model.predict(X_test.fillna(X_test.median(numeric_only=True)))

        st.success("Analyse prête ✅")
        nice_metric_row(metrics)
        plot_scatter(y_test, pred, "Prédit vs Vrai (yards holdout)")

        st.markdown("### RandomForest importances")
        fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        plot_top_bar(fi, "Top importances — Yards", top_n=25)

        st.markdown("### Permutation importance (recommandé)")
        with st.spinner("Permutation importance…"):
            perm = permutation_importance(
                model,
                X_test.fillna(X_test.median(numeric_only=True)),
                y_test,
                n_repeats=10,
                random_state=int(random_state),
                n_jobs=-1,
            )
        perm_imp = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)
        plot_top_bar(perm_imp, "Permutation importance — Yards", top_n=25)

        st.markdown("### Top corrélations avec la cible (simple & parlant)")
        corr = df[feat_cols + [yards_target]].corr(numeric_only=True)[yards_target].drop(yards_target)
        corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)
        st.dataframe(pd.DataFrame({"corr_with_target": corr.round(3)}).head(25), use_container_width=True)

        out = {
            "target": yards_target,
            "metrics": metrics,
            "top_rf": fi.head(25).to_dict(),
            "top_perm": perm_imp.head(25).to_dict(),
        }
        (REPORTS_DIR / "yards_drivers.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        st.caption("Sauvegardé : `reports/yards_drivers.json`")


# =========================================================
# PAGE 5 — QA
# =========================================================
elif page == "🧪 Qualité & diagnostics":
    st.subheader("Qualité des données & checks (style entreprise)")

    st.markdown("### Duplicates")
    dup = int(df.duplicated().sum())
    st.write(f"- Lignes dupliquées : **{dup}**")

    st.markdown("### Outliers (IQR)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.info("Aucune colonne numérique.")
    else:
        col = st.selectbox("Choisir une colonne", num_cols)
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_count = int(((s < lo) | (s > hi)).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Q1", f"{q1:.3f}")
        c2.metric("Q3", f"{q3:.3f}")
        c3.metric("Outliers (IQR)", f"{out_count}")

        fig, ax = plt.subplots()
        ax.hist(s.values, bins=30)
        ax.set_title(f"Distribution — {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.markdown("### Export rapide (sample nettoyé)")
    if st.button("💾 Exporter un sample nettoyé (numériques fill median)"):
        out = df.copy()
        num = out.select_dtypes(include=[np.number]).columns
        out[num] = out[num].fillna(out[num].median(numeric_only=True))
        out_path = REPORTS_DIR / "sample_cleaned.csv"
        out.head(500).to_csv(out_path, index=False)
        st.success(f"Exporté : {out_path}")
