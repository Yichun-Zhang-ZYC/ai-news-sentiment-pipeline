"""Stage 2: strip boilerplate (emails, phone numbers, newsletter/cookie
notices) out of the article text with a light regex pass.

Input:  artifacts/clean_news_v1.parquet
Output: artifacts/clean_news_v2_light_regex.parquet
"""
import gc
import re

import pandas as pd

from config import ARTIFACTS_DIR

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(\+?\d{1,2}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")

BOILERPLATE_SNIPPETS = [
    r"subscribe\s+to\s+our\s+newsletter",
    r"sign\s+up\s+for\s+our\s+newsletter",
    r"privacy\s+policy",
    r"terms\s+of\s+service",
    r"cookie(s)?\s+policy",
    r"all\s+rights\s+reserved",
    r"advertisement",
    r"read\s+more",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_SNIPPETS), re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

CHUNK_SIZE = 20_000


def light_clean(s: str) -> str:
    if s is None:
        return ""
    s = EMAIL_RE.sub(" ", str(s))
    s = PHONE_RE.sub(" ", s)
    s = BOILERPLATE_RE.sub(" ", s)
    return WHITESPACE_RE.sub(" ", s).strip()


def main() -> None:
    df = pd.read_parquet(ARTIFACTS_DIR / "clean_news_v1.parquet")

    texts = df["text_clean"].tolist()
    cleaned = []
    for i in range(0, len(texts), CHUNK_SIZE):
        cleaned.extend(light_clean(t) for t in texts[i : i + CHUNK_SIZE])
        if (i // CHUNK_SIZE) % 5 == 0:
            gc.collect()

    df["text_clean"] = cleaned
    out_path = ARTIFACTS_DIR / "clean_news_v2_light_regex.parquet"
    df.to_parquet(out_path, index=False)
    print("Saved:", out_path, df.shape)


if __name__ == "__main__":
    main()
