"""Stage 3: keep only articles that are actually about AI's impact on
business/industry, using unsupervised semantic similarity instead of
a keyword filter.

An article is scored by cosine similarity between its (title + first
1000 chars) embedding and a single hand-written "ideal article" query,
using sentence-transformers/all-MiniLM-L6-v2. The top 70% by score are kept.

Input:  artifacts/clean_news_v2_light_regex.parquet
Output: artifacts/news_relevant_v1.parquet
        artifacts/semantic_filter_report_v1.json
"""
import gc
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from config import ARTIFACTS_DIR

QUERY = (
    "Articles discussing how artificial intelligence impacts industries, companies, "
    "workforce, productivity, adoption, automation, regulation, business "
    "transformation, and enterprise deployment."
)
MAX_CHARS = 1000
BATCH_SIZE = 256
KEEP_TOP = 0.70


def iter_batches(df: pd.DataFrame, batch_size: int):
    n = len(df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        titles = df["title_clean"].iloc[start:end].astype(str).tolist()
        texts = df["text_clean"].iloc[start:end].astype(str).tolist()
        batch = [(t + " " + x[:MAX_CHARS]).strip() for t, x in zip(titles, texts)]
        yield start, end, batch


def main() -> None:
    df = pd.read_parquet(ARTIFACTS_DIR / "clean_news_v2_light_regex.parquet")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    query_emb = model.encode([QUERY], normalize_embeddings=True)

    scores = np.empty(len(df), dtype=np.float32)
    for start, end, batch in iter_batches(df, BATCH_SIZE):
        emb = model.encode(batch, normalize_embeddings=True)
        scores[start:end] = (emb @ query_emb.T).squeeze(1).astype(np.float32)
        if (start // BATCH_SIZE) % 200 == 0:
            gc.collect()
    df["semantic_score"] = scores

    threshold = float(np.quantile(df["semantic_score"], 1 - KEEP_TOP))
    df_relevant = df[df["semantic_score"] >= threshold].copy()
    print(f"Before: {len(df)}  After: {len(df_relevant)}  Keep rate: {len(df_relevant) / len(df):.3f}")

    out_path = ARTIFACTS_DIR / "news_relevant_v1.parquet"
    df_relevant[["date", "title_clean", "text_clean", "semantic_score"]].to_parquet(out_path, index=False)

    report = {
        "input": "clean_news_v2_light_regex.parquet",
        "output": out_path.name,
        "rows_before": int(len(df)),
        "rows_after": int(len(df_relevant)),
        "keep_rate": len(df_relevant) / len(df),
        "threshold_quantile": 1 - KEEP_TOP,
        "threshold_value": threshold,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_chars_used": MAX_CHARS,
        "batch_size": BATCH_SIZE,
    }
    (ARTIFACTS_DIR / "semantic_filter_report_v1.json").write_text(json.dumps(report, indent=2))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
