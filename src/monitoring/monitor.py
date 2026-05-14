# ============================================================
# MONITORAGGIO PREDIZIONI PER SENTIMENT ANALYSIS
# ============================================================
# SCOPO: Registrare su file CSV tutte le predizioni fatte dal modello
# in produzione, per consentire successiva analisi di drift.
#
# OUTPUT: predictions_log.csv con colonne:
#   - timestamp: momento della predizione (UTC)
#   - text: testo originale analizzato
#   - predicted_label: sentiment predetto (negative/neutral/positive)
#   - confidence: confidenza del modello (0.0 - 1.0)
#
# ============================================================

import os
import csv
from datetime import datetime, timezone
from src.model import load_classifier


# Path del file CSV dove salvare i log delle predizioni
LOG_FILE   = os.getenv("LOG_FILE", "./predictions_log/predictions_log.csv")

# Nomi delle colonne nel file CSV (ordine mantenuto)
LOG_FIELDS = ["timestamp", "text", "predicted_label", "confidence"]


# Crea il file di log con l'intestazione delle colonne se non esiste già.
# La directory viene creata automaticamente se non presente.
def init_log() -> None:
    """
    Create the log file if it does not exist.

    Returns:
        None
    """

    # Crea la directory del file se non esiste (es. ./monitoring)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # Se il file non esiste, crealo con l'intestazione
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()


# Registra su file CSV le predizioni per uno o più testi.
# - Se texts è una stringa singola, la converte in lista
# - Se classifier è None, carica il classificatore di default
# - Le predizioni usano truncation=True e max_length=512
# - Il timestamp viene salvato in UTC, la confidence arrotondata a 4 decimali
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

    if isinstance(texts, str):
        texts = [texts]

    # Se non viene fornito un classifier esterno, carica quello di default
    # In produzione, conviene passarlo già caricato per riutilizzarlo
    # su più chiamate (evita di ricaricare il modello ad ogni batch)
    if classifier is None:
        classifier = load_classifier()


    # Crea directory e file CSV se non esistono già
    init_log()


    # truncation=True e max_length=512 per gestire testi più lunghi
    # del contesto massimo del modello RoBERTa (512 token)
    raw_results = classifier(texts, truncation=True, max_length=512)

    # Apre il file in modalità append per non sovrascrivere i log esistenti
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        for text, result in zip(texts, raw_results):
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": text,
                "predicted_label": result["label"].lower(),
                "confidence": round(result["score"], 4)
            })

    return raw_results