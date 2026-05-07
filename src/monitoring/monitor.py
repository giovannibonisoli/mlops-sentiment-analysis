import os
import csv
from datetime import datetime
from src.model import load_classifier, predict

LOG_FILE  = os.getenv("LOG_FILE", "./monitoring/predictions_log.csv")
LOG_FIELDS = ["timestamp", "text", "predicted_label", "confidence"]

def init_log():
    """Crea il file di log se non esiste."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()

def log_predictions(texts, classifier=None):
    """Logga le predizioni su nuovi testi nel CSV."""
    if classifier is None:
        classifier = load_classifier()

    init_log()

    raw_results = classifier(texts, truncation=True, max_length=512)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        for text, result in zip(texts, raw_results):
            writer.writerow({
                "timestamp":       datetime.utcnow().isoformat(),
                "text":            text,
                "predicted_label": result["label"].lower(),
                "confidence":      round(result["score"], 4)
            })

    return raw_results