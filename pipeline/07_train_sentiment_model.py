"""Stage 7: fine-tune DistilBERT for 3-class sentiment classification
(negative / neutral / positive) on the tweet_eval sentiment dataset.

The resulting model is later applied to the AI news corpus for
topic-level, entity-level, and time-series sentiment analysis.

Output: models/ai_sentiment_model/  (HuggingFace format: config.json,
        tokenizer files, model.safetensors)
"""
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score  # noqa: F401  (kept for parity with notebook metrics)
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import MODEL_DIR

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    dataset = load_dataset("tweet_eval", "sentiment")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize, batched=True).remove_columns(["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1_weighted": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
        }

    training_args = TrainingArguments(
        output_dir="./_train_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print("Model saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
