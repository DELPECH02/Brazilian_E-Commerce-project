from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "nfl_offense.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "nfl_offense_processed.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
