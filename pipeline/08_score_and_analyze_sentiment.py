"""Stage 8: apply the fine-tuned sentiment model to every article, then
break the results down by topic, by entity (company), and over time.

Input:  artifacts/ner_input_articles.parquet
        artifacts/entities_raw.parquet
        models/ai_sentiment_model/
Output: artifacts/article_sentiment.parquet
        artifacts/sentiment_over_time.parquet
        artifacts/topic_sentiment.csv
        artifacts/entity_sentiment.csv
"""
import pandas as pd
import torch
from transformers import pipeline

from config import ARTIFACTS_DIR, MODEL_DIR

BATCH_SIZE = 64
MAX_CHARS = 2000


def score_articles(df: pd.DataFrame) -> pd.DataFrame:
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipe = pipeline(
        "text-classification", model=str(MODEL_DIR), tokenizer=str(MODEL_DIR), device=device
    )

    texts = df["text_clean"].fillna("").astype(str).str[:MAX_CHARS].tolist()
    results = []
    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        results.extend(sentiment_pipe(texts[start:end], truncation=True, max_length=512))
        if start % 5000 == 0:
            print(f"processed: {start} -> {end}")

    df["sentiment"] = [r["label"].lower() for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    return df


def main() -> None:
    df = pd.read_parquet(ARTIFACTS_DIR / "ner_input_articles.parquet")
    df = score_articles(df)
    df.to_parquet(ARTIFACTS_DIR / "article_sentiment.parquet", index=False)
    print("Saved article_sentiment.parquet:", df.shape)

    # Sentiment breakdown by topic
    topic_sentiment = (
        df.groupby(["topic_name", "sentiment"]).size().reset_index(name="count")
        .sort_values(["topic_name", "count"], ascending=[True, False])
    )
    topic_sentiment["total"] = topic_sentiment.groupby("topic_name")["count"].transform("sum")
    topic_sentiment["pct"] = topic_sentiment["count"] / topic_sentiment["total"]
    topic_sentiment.to_csv(ARTIFACTS_DIR / "topic_sentiment.csv", index=False)

    # Sentiment breakdown by entity (company)
    df_entities = pd.read_parquet(ARTIFACTS_DIR / "entities_raw.parquet")
    df_entities = df_entities[df_entities["entity_label"] == "ORG"].merge(
        df[["title_clean", "sentiment", "sentiment_score"]], on="title_clean", how="left"
    )
    entity_sentiment = (
        df_entities.groupby(["entity_text", "sentiment"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    entity_sentiment.to_csv(ARTIFACTS_DIR / "entity_sentiment.csv", index=False)

    # Sentiment over time (monthly)
    df["date"] = pd.to_datetime(df["date"])
    sentiment_time = (
        df.groupby([pd.Grouper(key="date", freq="M"), "sentiment"]).size().reset_index(name="count")
    )
    sentiment_time["total"] = sentiment_time.groupby("date")["count"].transform("sum")
    sentiment_time["pct"] = sentiment_time["count"] / sentiment_time["total"]
    sentiment_time.to_parquet(ARTIFACTS_DIR / "sentiment_over_time.parquet", index=False)
    print("Saved sentiment_over_time.parquet, topic_sentiment.csv, entity_sentiment.csv")


if __name__ == "__main__":
    main()
