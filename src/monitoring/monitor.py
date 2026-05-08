import os
import csv
from datetime import datetime, timezone
from src.model import load_classifier

LOG_FILE  = os.getenv("LOG_FILE", "./monitoring/predictions_log.csv")
LOG_FIELDS = ["timestamp", "text", "predicted_label", "confidence"]

def init_log() -> None:
    """
    Create the log file if it does not exist.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()


def log_predictions(texts, classifier=None):
    """
    Log predictions for new texts to CSV.

    Args:
        texts:
            Single text or iterable of texts to predict.
        classifier (TextClassificationPipeline, optional):
            Hugging Face sentiment-analysis pipeline.
            If None, loads the default classifier.

    Returns:
        List of raw prediction results from the classifier.
    """
    if classifier is None:
        classifier = load_classifier()

    init_log()

    raw_results = classifier(texts, truncation=True, max_length=512)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        for text, result in zip(texts, raw_results):
            writer.writerow({
                "timestamp":       datetime.now(timezone.utc).isoformat(),
                "text":            text,
                "predicted_label": result["label"].lower(),
                "confidence":      round(result["score"], 4)
            })

    return raw_results