import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

MODEL_NAME     = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR     = "./models/sentiment_model"
DATASET_NAME   = os.getenv("DATASET_NAME", "tweet_eval")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "sentiment")
TRAIN_SAMPLES  = int(os.getenv("TRAIN_SAMPLES", 1000))
EVAL_SAMPLES   = int(os.getenv("EVAL_SAMPLES", 1000))

def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }

def train():
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    train_data = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
    eval_data  = dataset["validation"].shuffle(seed=42).select(range(EVAL_SAMPLES))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    train_data = train_data.map(lambda b: tokenize(b, tokenizer), batched=True)
    eval_data  = eval_data.map(lambda b: tokenize(b, tokenizer), batched=True)

    train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_data.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(f"Accuracy: {metrics['eval_accuracy']:.2f}")
    print(f"Macro F1: {metrics['eval_macro_f1']:.2f}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modello salvato in {OUTPUT_DIR}")

if __name__ == "__main__":
    train()