"""Stage 6: rule-based tagging of *how* AI is described as affecting a
business (impact mode) and *what kind* of AI is being discussed
(technology), via curated keyword/phrase lists per article.

Input:  artifacts/ner_input_articles.parquet
Output: artifacts/industry_impact_counts.csv
        artifacts/industry_technology_counts.csv
        artifacts/impact_article_level.parquet
"""
import pandas as pd

from config import ARTIFACTS_DIR

IMPACT_KEYWORDS = {
    "automation": [
        "automate", "automated", "automation", "fully automated", "autonomously",
        "autonomous processing", "machine-led", "replace manual work", "replace human work",
        "replace workers", "replace staff", "reduce headcount", "eliminate routine tasks",
        "task automation", "document automation", "underwriting automation",
        "decision automation", "robotic process automation", "rpa", "self-service",
        "without human intervention",
    ],
    "augmentation": [
        "augment", "augmentation", "assist", "assistance", "support workers",
        "support employees", "decision support", "copilot", "co-pilot", "assistant",
        "ai assistant", "human in the loop", "human-in-the-loop", "collaborate with",
        "enhance human", "improve clinician performance", "recommendation engine",
        "advisory tool", "decision aid", "support clinicians", "support teachers",
        "support developers",
    ],
    "workflow_redesign": [
        "workflow redesign", "workflow transformation", "workflow optimization",
        "workflow automation", "business process redesign", "business process transformation",
        "process redesign", "process transformation", "process optimization",
        "operational redesign", "operational transformation", "streamline operations",
        "streamline workflow", "integrated workflow", "embedded into workflow",
        "digital transformation", "reengineer processes", "re-engineer processes",
        "change how work is done", "reshape operations",
    ],
    "cost_reduction": [
        "reduce cost", "reduce costs", "cut cost", "cut costs", "lower cost", "lower costs",
        "cost reduction", "cost savings", "save money", "saving money", "reduce spending",
        "lower spending", "operational savings", "efficiency savings", "labor savings",
        "lower labor costs", "decrease expenses", "reduced overhead", "improve margins",
        "margin improvement",
    ],
    "productivity_gain": [
        "productivity", "increase productivity", "boost productivity", "improve efficiency",
        "increase efficiency", "greater efficiency", "faster", "faster turnaround",
        "speed up", "speeds up", "accelerate", "accelerates", "shorten turnaround",
        "reduce turnaround time", "save time", "time savings", "handle more work",
        "scale output", "higher throughput", "fewer bottlenecks", "improve performance",
    ],
    "risk_or_disruption": [
        "job loss", "job losses", "layoffs", "workforce reduction", "displace workers",
        "worker displacement", "disruption", "operational risk", "compliance risk", "bias",
        "hallucination", "safety concern", "security risk", "cyber risk", "ethical concern",
        "governance risk", "regulatory risk", "uncertainty", "hard to say", "unclear impact",
        "mixed impact", "unintended consequences", "threat to jobs", "replace jobs",
    ],
}

TECHNOLOGY_KEYWORDS = {
    "llm": ["large language model", "large language models", "llm", "llms", "foundation model", "foundation models", "generative ai", "genai", "language model"],
    "computer_vision": ["computer vision", "image recognition", "image analysis", "vision model", "medical imaging", "object detection", "visual inspection"],
    "predictive_analytics": ["predictive analytics", "prediction model", "forecasting", "risk scoring", "risk model", "scoring model"],
    "decision_support_system": ["decision support", "recommendation engine", "clinical decision support", "advisory system", "decision engine"],
    "automation_platform": ["robotic process automation", "rpa", "workflow automation", "process automation", "orchestration"],
    "chatbot_or_assistant": ["chatbot", "virtual assistant", "ai assistant", "copilot", "co-pilot"],
}


def detect(text: str, keyword_dict: dict) -> list[str]:
    text = str(text).lower()
    return sorted({label for label, kws in keyword_dict.items() if any(kw in text for kw in kws)})


def main() -> None:
    df = pd.read_parquet(ARTIFACTS_DIR / "ner_input_articles.parquet")
    df["impact_modes"] = df["text_clean"].apply(lambda t: detect(t, IMPACT_KEYWORDS))
    df["technologies"] = df["text_clean"].apply(lambda t: detect(t, TECHNOLOGY_KEYWORDS))

    impact_long = df.explode("impact_modes").dropna(subset=["impact_modes"])
    tech_long = df.explode("technologies").dropna(subset=["technologies"])

    industry_impact_counts = (
        impact_long.groupby(["industry_label", "impact_modes"]).size().reset_index(name="count")
        .sort_values(["industry_label", "count"], ascending=[True, False])
    )
    industry_tech_counts = (
        tech_long.groupby(["industry_label", "technologies"]).size().reset_index(name="count")
        .sort_values(["industry_label", "count"], ascending=[True, False])
    )

    df.to_parquet(ARTIFACTS_DIR / "impact_article_level.parquet", index=False)
    industry_impact_counts.to_csv(ARTIFACTS_DIR / "industry_impact_counts.csv", index=False)
    industry_tech_counts.to_csv(ARTIFACTS_DIR / "industry_technology_counts.csv", index=False)
    print("Saved industry_impact_counts.csv and industry_technology_counts.csv")


if __name__ == "__main__":
    main()
