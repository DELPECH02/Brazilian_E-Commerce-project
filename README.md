# NFL Offense Analytics (2005–2024) — projet Data Analyst (Python)

Projet *end-to-end* basé sur un dataset que tu as scrapé (stats d'attaque NFL par équipe et par saison, 2005→2024).
L'objectif est de montrer un workflow réaliste de Data Analyst : **nettoyage**, **feature engineering**, **EDA**, **modélisation**, **reproductibilité**, **tests**, et une **mini-app Streamlit**.

---

## 1) Questions métier simulées (exemples)

- Comment évoluent les attaques NFL dans le temps (points, yards, style de jeu) ?
- Quelles équipes dominent offensivement sur la dernière saison disponible ?
- Peut-on **prédire `Pts` (points par match)** à partir d'indicateurs d'efficacité et de volume ?

---

## 2) Contenu du repo

```api.py              # Backend : Serveur FastAPI (Moteur de prédiction)
app_streamlit.py    # Frontend : Interface utilisateur interactive
data/
  raw/              # Dataset NFL original (nfl_offense.csv)
  processed/        # (Optionnel) Datasets nettoyés
models/             # Modèles entraînés (.joblib)
reports/            # Graphiques et exports d'analyse
requirements.txt    # Dépendances (FastAPI, Streamlit, Scikit-Learn...)
tests/              # Tests unitaires (Pytest)
```

---

## 3) Installation rapide

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 4) Lancer l'EDA (graphiques + tables)

```bash
python -m src.eda --data data/raw/nfl_offense.csv
```

Sorties :
- `reports/figures/trend_points_per_game.png`
- `reports/figures/scatter_yds_per_play_vs_pts.png`
- `reports/figures/top10_pts_latest_year.png`
- `reports/yearly_summary.csv`

---

## 5) Entraîner un modèle (split temporel)

On garde les **3 dernières saisons** en test (time-aware split).

```bash
python -m src.train --data data/raw/nfl_offense.csv --model random_forest --test-years 3
```

Le modèle et les métriques sont sauvegardés dans `models/`.

**Résultat (exemple, RandomForest, test = 3 dernières saisons):**
- RMSE ≈ 1.39
- MAE  ≈ 1.16
- R²   ≈ 0.89

> Note : le pipeline inclut une protection contre la **fuite de cible** (features dérivées de `Pts` retirées pour prédire `Pts`).

---

## 6) Lancer la mini-app Streamlit 

Ce projet nécessite deux terminaux ouverts simultanément :

Terminal 1 - L'API (Le Cerveau) :
```bash
python -m uvicorn api:app --reload
```

Terminal 2 - L'Interface (Le Visuel) :
```bash
python -m streamlit run app_streamlit.py
```

Fonctions :
- filtres par période + équipes
- tendances ligue
- comparaison d'équipes sur `Pts`
- table exploratoire

---

## 7) Tests unitaires

```bash
pytest -q
```

---

## 8) Exemples de bullet points CV (à adapter)

- Conçu un pipeline **ETL/cleaning** sur un dataset scrapé (normalisation de franchises, typage numérique, clés stables `team_key`)
- Réalisé une **EDA** et produit des visualisations (tendances temporelles, top équipes, relations efficacité ↔ scoring)
- Mis en place du **feature engineering** (ratios pass/run, shares, discipline, ball security)
- Développé un modèle de régression (**RandomForest + split temporel**) pour estimer les points par match (R² ~ 0.89)
- Industrialisé le projet : scripts CLI, artefacts sauvegardés (modèle + métriques), **tests pytest**, mini-dashboard **Streamlit**

---

## 9) Sources & éthique

Dataset scrapé à partir de données publiques.  
Pense à citer la source originale dans ton portfolio (page web / site) si tu le publies.
