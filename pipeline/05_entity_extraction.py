"""Stage 5: named-entity extraction (organizations) over the topic-labeled
corpus, using a RoBERTa NER model, then normalize company names.

Input:  artifacts/ner_input_articles.parquet
Output: artifacts/entities_raw.parquet
        artifacts/industry_company_counts.csv
"""
import re

import pandas as pd
import torch
from transformers import pipeline

from config import ARTIFACTS_DIR

MAX_CHARS = 2000
CHUNK_SIZE = 512
BATCH_SIZE = 32

CORP_SUFFIXES = {"inc", "inc.", "corp", "corp.", "corporation", "ltd", "ltd.", "llc", "plc", "group", "holdings", "holding", "co", "co.", "company"}


def normalize_company(name: str) -> str:
    name = re.sub(r"[^\w\s]", " ", str(name)).lower()
    words = [w for w in name.split() if w not in CORP_SUFFIXES]
    return " ".join(words).title().strip()


def main() -> None:
    df = pd.read_parquet(ARTIFACTS_DIR / "ner_input_articles.parquet")
    df["ner_text"] = (df["title_clean"].fillna("") + ". " + df["text_clean"].fillna("")).str[:3000]

    device = 0 if torch.cuda.is_available() else -1
    ner_pipe = pipeline(
        "token-classification",
        model="Jean-Baptiste/roberta-large-ner-english",
        aggregation_strategy="simple",
        device=device,
    )

    texts = df["ner_text"].fillna("").astype(str).str[:MAX_CHARS].tolist()
    records = []
    for start in range(0, len(texts), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(texts))
        outputs = ner_pipe(texts[start:end], batch_size=BATCH_SIZE)
        rows = df.iloc[start:end]

        for row, ents in zip(rows.itertuples(), outputs):
            for ent in ents:
                if ent["entity_group"] != "ORG":
                    continue
                entity_text = re.sub(r"\s+", " ", ent["word"]).strip(" .,;:|()[]{}\"'")
                if len(entity_text) >= 2:
                    records.append(
                        {
                            "date": row.date,
                            "topic": row.topic,
                            "topic_name": row.topic_name,
                            "industry_label": row.industry_label,
                            "title_clean": row.title_clean,
                            "entity_text": entity_text,
                            "entity_label": ent["entity_group"],
                            "score": ent["score"],
                        }
                    )
        print(f"processed: {start} -> {end}")

    df_entities = pd.DataFrame(records)
    df_entities.to_parquet(ARTIFACTS_DIR / "entities_raw.parquet", index=False)
    print("Saved entities_raw.parquet:", df_entities.shape)

    df_entities["company"] = df_entities["entity_text"].apply(normalize_company)
    df_entities = df_entities[df_entities["company"].str.len() > 2]
    df_entities = df_entities[~df_entities["company"].str.contains("Ai$", case=False)]

    company_counts = (
        df_entities.groupby(["industry_label", "company"]).size().reset_index(name="count")
        .sort_values(["industry_label", "count"], ascending=[True, False])
    )
    company_counts.to_csv(ARTIFACTS_DIR / "industry_company_counts.csv", index=False)
    print("Saved industry_company_counts.csv")


if __name__ == "__main__":
    main()
