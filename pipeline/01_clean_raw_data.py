"""Stage 1: load the raw news corpus, profile it, and apply basic filtering.

Input:  public parquet dataset (RAW_NEWS_URL)
Output: artifacts/clean_news_v1.parquet  (date, title_clean, text_clean)
        artifacts/01_time_report.json
        artifacts/01_sample_100.jsonl
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, DATE_COL, LANG_COL, MAX_TEXT_LEN, MIN_TEXT_LEN, RAW_NEWS_URL, TEXT_COL, TITLE_COL, URL_COL

_WHITESPACE_RE = re.compile(r"\s+")
_NULL_RE = re.compile(r"\x00")


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = _NULL_RE.sub("", s)
    s = s.replace("\r", "\n")
    return _WHITESPACE_RE.sub(" ", s).strip()


def main() -> None:
    df = pd.read_parquet(RAW_NEWS_URL, engine="pyarrow")
    print("Loaded shape:", df.shape)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)
    time_report = {
        "parse_success_pct": float(df[DATE_COL].notna().mean() * 100),
        "min_date_utc": str(df[DATE_COL].min()),
        "max_date_utc": str(df[DATE_COL].max()),
        "n_missing_date": int(df[DATE_COL].isna().sum()),
    }
    (ARTIFACTS_DIR / "01_time_report.json").write_text(json.dumps(time_report, indent=2))

    sample_df = df.sample(n=100, random_state=42)[[URL_COL, DATE_COL, TITLE_COL, TEXT_COL]]
    sample_df.to_json(ARTIFACTS_DIR / "01_sample_100.jsonl", orient="records", lines=True, force_ascii=False)

    # English-only, sane length range
    df = df[df[LANG_COL].astype(str).str.lower().eq("en")].copy()
    text_len = df[TEXT_COL].fillna("").astype(str).str.len()
    df = df[(text_len >= MIN_TEXT_LEN) & (text_len <= MAX_TEXT_LEN)].copy()
    print("After language + length filter:", df.shape)

    df["title_norm"] = df[TITLE_COL].map(normalize_text)
    df["text_norm"] = df[TEXT_COL].map(normalize_text)

    df = df.drop_duplicates(subset=[URL_COL])
    clean = df[[DATE_COL, "title_norm", "text_norm"]].rename(
        columns={"title_norm": "title_clean", "text_norm": "text_clean"}
    )

    out_path = ARTIFACTS_DIR / "clean_news_v1.parquet"
    clean.to_parquet(out_path, index=False)
    print("Saved:", out_path, clean.shape)


if __name__ == "__main__":
    main()
