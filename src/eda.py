import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from .data import load_raw_csv, clean
from .features import add_features
from .config import FIGURES_DIR

def run_eda(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    d = add_features(df)

    # 1) League trend of points per game over time (mean across teams)
    yearly = d.groupby("year", as_index=False).agg(
        avg_pts=("Pts", "mean"),
        avg_total_yds=("TotalYds", "mean"),
        avg_to=("TO", "mean"),
        avg_pass_rate=("pass_rate", "mean"),
    )

    fig = plt.figure()
    plt.plot(yearly["year"], yearly["avg_pts"])
    plt.title("NFL offense — points per game (league average)")
    plt.xlabel("Year")
    plt.ylabel("Avg Pts (per team per game)")
    fig.tight_layout()
    fig.savefig(out_dir / "trend_points_per_game.png", dpi=160)
    plt.close(fig)

    # 2) Scatter: yards per play vs points
    fig = plt.figure()
    plt.scatter(d["YdsPerPlay"], d["Pts"], alpha=0.5)
    plt.title("Efficiency vs scoring")
    plt.xlabel("Yards per play")
    plt.ylabel("Points per game")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_yds_per_play_vs_pts.png", dpi=160)
    plt.close(fig)

    # 3) Top offenses (by Pts) for latest year in the dataset
    latest_year = int(d["year"].max())
    top = d.loc[d["year"] == latest_year, ["team", "Pts"]].sort_values("Pts", ascending=False).head(10)

    fig = plt.figure()
    plt.barh(top["team"][::-1], top["Pts"][::-1])
    plt.title(f"Top 10 offenses by points — {latest_year}")
    plt.xlabel("Pts per game")
    fig.tight_layout()
    fig.savefig(out_dir / "top10_pts_latest_year.png", dpi=160)
    plt.close(fig)

    # Save a quick table used in the README / portfolio
    yearly.to_csv(out_dir.parent / "yearly_summary.csv", index=False)
    top.to_csv(out_dir.parent / f"top10_{latest_year}.csv", index=False)

def main():
    p = argparse.ArgumentParser(description="Generate EDA figures and summary tables.")
    p.add_argument("--data", required=True, help="Path to raw CSV")
    p.add_argument("--out", default=str(FIGURES_DIR), help="Output directory for figures")
    args = p.parse_args()

    df = clean(load_raw_csv(args.data))
    run_eda(df, Path(args.out))
    print(f"EDA artifacts saved to: {args.out}")

if __name__ == "__main__":
    main()
