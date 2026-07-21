"""Shared paths and constants for the pipeline stages.

Every stage reads/writes under ARTIFACTS_DIR so the pipeline can run
locally, in Colab, or on any machine by just setting PROJECT_ROOT.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models" / "ai_sentiment_model"

for d in (DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR, MODEL_DIR.parent):
    d.mkdir(parents=True, exist_ok=True)

RAW_NEWS_URL = (
    "https://storage.googleapis.com/msca-bdp-data-open/"
    "news_final_project/news_final_project.parquet"
)

DATE_COL = "date"
URL_COL = "url"
LANG_COL = "language"
TITLE_COL = "title"
TEXT_COL = "text"

MIN_TEXT_LEN = 500
MAX_TEXT_LEN = 50_000
