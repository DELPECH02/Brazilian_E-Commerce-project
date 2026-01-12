import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import joblib

from .data import load_raw_csv, clean, save_processed_csv
from .features import model_features, add_features
from .config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
def time_split(df: pd.DataFrame, test_years: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split last N years as test to respect time ordering."""
    years = sorted(df["year"].unique())
    if test_years >= len(years):
        raise ValueError("test_years too large for available data")
    cutoff = years[-test_years]
    train = df[df["year"] < cutoff].copy()
    test = df[df["year"] >= cutoff].copy()
    return train, test

def build_models(random_state: int = RANDOM_STATE):
    ridge = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=2.0, random_state=random_state)),
    ])

    rf = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    return {"ridge": ridge, "random_forest": rf}



def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": rmse,
        "r2": r2
    }


def main():
    p = argparse.ArgumentParser(description="Train a points-per-game model from team offense stats.")
    p.add_argument("--data", required=True, help="Path to raw CSV")
    p.add_argument("--target", default="Pts", help="Target column (default: Pts)")
    p.add_argument("--test-years", type=int, default=3, help="Number of last years for test split")
    p.add_argument("--model", choices=["ridge", "random_forest"], default="random_forest")
    p.add_argument("--save-name", default=None, help="Optional model filename (without extension)")
    args = p.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_csv(args.data)
    df = clean(raw)
    df_fe = add_features(df)

    # Persist processed dataset for reproducibility (no parquet dependency)
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    save_processed_csv(df_fe, str(DATA_PROCESSED))

    train_df, test_df = time_split(df_fe, test_years=args.test_years)

    X_train, y_train = model_features(train_df, target=args.target)
    X_test, y_test = model_features(test_df, target=args.target)

    models = build_models()
    model = models[args.model]
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = evaluate(y_test, pred)

    save_name = args.save_name or f"{args.model}_{args.target.lower()}_last{args.test_years}y"
    model_path = MODELS_DIR / f"{save_name}.joblib"
    joblib.dump(model, model_path)

    metrics_path = MODELS_DIR / f"{save_name}_metrics.json"
    payload = {
        "model": args.model,
        "target": args.target,
        "test_years": int(args.test_years),
        "train_years": [int(x) for x in sorted(train_df["year"].unique())],
        "test_years_list": [int(x) for x in sorted(test_df["year"].unique())],
        "metrics": metrics,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Test metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
