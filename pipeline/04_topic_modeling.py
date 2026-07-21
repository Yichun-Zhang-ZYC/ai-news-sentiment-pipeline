"""Stage 4: BERTopic topic modeling over the relevance-filtered corpus.

Produces raw BERTopic clusters, which are then manually curated into
8 final industry-labeled topics (see data/topic_labels_manual.csv,
already committed) and joined back onto the article-level data.

Input:  artifacts/news_relevant_v1.parquet
Output: artifacts/topic_info_full.csv           (raw BERTopic clusters)
        artifacts/ner_input_articles.parquet    (kept articles + topic label)
"""
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text as sk_text
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from config import ARTIFACTS_DIR, DATA_DIR

MAX_TOPIC_WORDS = 300

# Domain stopwords on top of sklearn's English list: without these, BERTopic's
# top clusters are dominated by generic AI/business/newswire vocabulary
# ("ai", "company", "said", month names, ...) instead of the concepts that
# actually distinguish one topic from another.
CUSTOM_STOPWORDS = sk_text.ENGLISH_STOP_WORDS.union(
    {
        "ai", "artificial", "intelligence", "machine", "learning", "model", "models",
        "data", "system", "systems", "technology", "technologies", "tool", "tools",
        "platform", "platforms", "software", "solution", "solutions", "generative",
        "research", "company", "companies", "business", "industry", "industries",
        "digital", "development", "management", "future", "global", "world",
        "market", "markets", "share", "stock", "stocks", "public", "high",
        "news", "content", "home", "search", "menu", "subscribe", "watch",
        "videos", "video", "press", "updated", "latest", "live", "contact",
        "support", "help", "events", "insights", "local", "sign", "skip",
        "features", "releases", "view", "best", "year", "month", "day", "time",
        "india", "china", "country", "united", "states", "like", "people", "work",
        "ceo", "human", "users", "team", "free", "real", "based", "use", "today",
        "prnewswire", "google", "nasdaq", "sports", "music", "entertainment",
        "politics", "tv", "media", "published", "including", "just", "make",
        "made", "announced", "announce", "power", "email", "leading", "used",
        "using", "key", "create", "creates", "created", "years", "said", "says",
        "according", "report", "reported", "reports", "statement", "release",
        "via", "across", "major", "next", "first", "top", "big", "growing",
        "grows", "growth", "million", "billion", "percent", "per", "cent",
        "service", "services", "product", "products", "experience", "customer",
        "customers", "capabilities", "access", "helps", "enable", "enables",
        "allow", "allows", "thing", "things", "way", "ways", "part", "well",
        "much", "many", "also", "still", "even", "am", "pm", "ago", "days",
        "need", "making", "open", "openai", "chatgpt", "one", "two", "three",
        "four", "five", "six", "seven", "eight", "nine", "ten", "hours", "state",
        "monday", "tuesday", "wednesday", "thursday", "friday", "january",
        "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "microsoft", "apple",
        "facebook", "meta", "twitter", "nvidia", "applications", "application",
        "language", "large", "supports", "type", "types", "com", "www", "http",
        "https", "10", "00",
    }
)


def build_topic_text(row: pd.Series) -> str:
    words = str(row["text_clean"]).split()
    return str(row["title_clean"]) + ". " + " ".join(words[:MAX_TOPIC_WORDS])


def main() -> None:
    df = pd.read_parquet(ARTIFACTS_DIR / "news_relevant_v1.parquet")
    df["topic_text"] = df.apply(build_topic_text, axis=1)
    docs = df["topic_text"].tolist()
    print("Fitting BERTopic on", len(docs), "documents")

    vectorizer_model = CountVectorizer(
        stop_words=list(CUSTOM_STOPWORDS), ngram_range=(1, 2), min_df=10, max_df=0.6
    )
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=200, min_samples=20, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs)
    df["topic"] = topics
    topic_model.reduce_outliers(docs, topics)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(ARTIFACTS_DIR / "topic_info_full.csv", index=False)
    print(topic_info.head(20))

    # Manual curation (already committed at data/topic_labels_manual.csv):
    # collapse ~19 raw clusters into 8 final industry-labeled topics and
    # drop the newswire-template / geography-noise clusters.
    topic_labels = pd.read_csv(DATA_DIR / "topic_labels_manual.csv")
    df_labeled = df.merge(topic_labels, on="topic", how="left")
    df_keep = df_labeled[df_labeled["keep"] == 1].copy()
    print("Kept articles after topic curation:", df_keep.shape)

    out_path = ARTIFACTS_DIR / "ner_input_articles.parquet"
    df_keep[["date", "title_clean", "text_clean", "topic", "topic_name", "industry_label"]].to_parquet(
        out_path, index=False
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
