import os
import csv
import pytest
from datasets import load_dataset
from src.model import load_classifier, predict
from src.evaluate import evaluate
from src.monitoring.monitor import log_predictions
from src.monitoring.drift import (
    compute_sentiment_distribution,
    compute_avg_confidence,
    compute_psi,
    check_distribution_shift,
    run_monitoring_report
)

# ============================================================
# INTEGRATION TESTS — PIPELINE
# ============================================================
# Verifica che il classifier produca predizioni valide su dati reali
def test_pipeline_end_to_end():
    """
    Verify that the classifier produces valid predictions on real data.

    Returns:
        None
    """
    dataset = load_dataset("tweet_eval", "sentiment")
    sample = dataset["test"].select(range(10))

    classifier = load_classifier()
    predictions = predict(classifier, list(sample["text"]))

    assert len(predictions) == 10
    assert all(p in ["positive", "negative", "neutral"] for p in predictions)

# Verifica che le label prodotte dalla pipeline siano sempre nel set atteso
def test_pipeline_labels_are_valid():
    """
    Verify that the labels produced by the pipeline are always in the expected set.

    Returns:
        None
    """
    dataset = load_dataset("tweet_eval", "sentiment")
    sample = dataset["test"].select(range(50))

    classifier = load_classifier()
    predictions = predict(classifier, list(sample["text"]))

    valid_labels = {"positive", "negative", "neutral"}
    assert all(p in valid_labels for p in predictions)

# Verifica che le confidence scores siano sempre nel range [0, 1]
def test_pipeline_confidence_range():
    """
    Verify that confidence scores are always in the range [0, 1].

    Returns:
        None
    """
    os.environ["LOG_FILE"] = "./test_predictions_confidence.csv"

    dataset = load_dataset("tweet_eval", "sentiment")
    sample = dataset["test"].select(range(10))

    classifier = load_classifier()
    results = log_predictions(list(sample["text"]), classifier=classifier)

    assert all(0.0 <= r["score"] <= 1.0 for r in results)

    if os.path.exists("./test_predictions_confidence.csv"):
        os.remove("./test_predictions_confidence.csv")

# ============================================================
# INTEGRATION TESTS — EVALUATE
# ============================================================
# Verifica che evaluate() ritorni sia accuracy che macro_f1
def test_evaluate_returns_both_metrics():
    """
    Verify that evaluate() returns both accuracy and macro_f1.

    Returns:
        None
    """
    metrics = evaluate()
    assert "accuracy" in metrics
    assert "macro_f1" in metrics

# Verifica che evaluate() ritorni un valore di accuracy nel range [0, 1]
def test_evaluate_returns_valid_accuracy():
    """
    Verify that evaluate() returns an accuracy value in the range [0, 1].

    Returns:
        None
    """
    metrics = evaluate()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] > 0.5

# Verifica che evaluate() ritorni un valore di macro F1 nel range [0, 1]
def test_evaluate_returns_valid_macro_f1():
    """
    Verify that evaluate() returns a macro F1 value in the range [0, 1].

    Returns:
        None
    """
    metrics = evaluate()
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["macro_f1"] > 0.5

# ============================================================
# INTEGRATION TESTS — MONITOR
# ============================================================
# Verifica che log_predictions() crei il file CSV se non esiste
def test_log_predictions_creates_file():
    """
    Verify that log_predictions() creates the CSV file if it does not exist.

    Returns:
        None
    """
    log_path = "./test_predictions_create.csv"
    os.environ["LOG_FILE"] = log_path

    if os.path.exists(log_path):
        os.remove(log_path)

    classifier = load_classifier()
    log_predictions(["This is a test."], classifier=classifier)

    assert os.path.exists(log_path)
    os.remove(log_path)

# Verifica che log_predictions() appenda le predizioni senza sovrascrivere
def test_log_predictions_appends():
    """
    Verify that log_predictions() appends predictions without overwriting.

    Returns:
        None
    """
    log_path = "./test_predictions_append.csv"
    os.environ["LOG_FILE"] = log_path

    if os.path.exists(log_path):
        os.remove(log_path)

    classifier = load_classifier()
    log_predictions(["First text."], classifier=classifier)
    log_predictions(["Second text."], classifier=classifier)

    with open(log_path, "r") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    os.remove(log_path)

# Verifica che il CSV contenga tutte le colonne attese
def test_log_predictions_csv_fields():
    """
    Verify that the CSV contains all expected columns.

    Returns:
        None
    """
    log_path = "./test_predictions_fields.csv"
    os.environ["LOG_FILE"] = log_path

    if os.path.exists(log_path):
        os.remove(log_path)

    classifier = load_classifier()
    log_predictions(["This is a test."], classifier=classifier)

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames

    assert "timestamp" in fields
    assert "text" in fields
    assert "predicted_label" in fields
    assert "confidence" in fields
    os.remove(log_path)

# ============================================================
# INTEGRATION TESTS — DRIFT
# ============================================================
# Verifica che compute_sentiment_distribution() calcoli percentuali corrette
def test_compute_sentiment_distribution():
    """
    Verify that compute_sentiment_distribution() calculates correct percentages.

    Returns:
        None
    """
    rows = [
        {"predicted_label": "positive"},
        {"predicted_label": "positive"},
        {"predicted_label": "negative"},
        {"predicted_label": "neutral"},
    ]
    dist = compute_sentiment_distribution(rows)
    assert dist["positive"] == 0.5
    assert dist["negative"] == 0.25
    assert dist["neutral"] == 0.25

# Verifica che compute_avg_confidence() calcoli la media corretta
def test_compute_avg_confidence():
    """
    Verify that compute_avg_confidence() calculates the correct mean.

    Returns:
        None
    """
    rows = [
        {"confidence": "0.8"},
        {"confidence": "0.6"},
        {"confidence": "0.4"},
    ]
    avg = compute_avg_confidence(rows)
    assert abs(avg - 0.6) < 0.001

# Verifica che compute_avg_confidence() gestisca una lista vuota
def test_compute_avg_confidence_empty():
    """
    Verify that compute_avg_confidence() handles an empty list.

    Returns:
        None
    """
    avg = compute_avg_confidence([])
    assert avg == 0.0

# Verifica che il PSI sia ~0 su distribuzioni identiche
def test_compute_psi_identical_distributions():
    """
    Verify that PSI is ~0 on identical distributions.

    Returns:
        None
    """
    scores = [0.8, 0.7, 0.9, 0.6, 0.75] * 20
    psi = compute_psi(scores, scores)
    assert psi < 0.01

# Verifica che il PSI sia alto su distribuzioni molto diverse
def test_compute_psi_different_distributions():
    """
    Verify that PSI is high on very different distributions.

    Returns:
        None
    """
    baseline = [0.9] * 100
    current  = [0.3] * 100
    psi = compute_psi(baseline, current)
    assert psi > 0.2

# Verifica che check_distribution_shift() rilevi uno shift superiore al 15%
def test_check_distribution_shift_detected():
    """
    Verify that check_distribution_shift() detects a shift greater than 15%.

    Returns:
        None
    """
    # Very skewed distribution toward negative (simulates reputation crisis)
    current = {"negative": 0.70, "neutral": 0.20, "positive": 0.10}
    shifts = check_distribution_shift(current)
    assert "negative" in shifts
    assert "neutral" in shifts

# Verifica che check_distribution_shift() non rilevi shift su distribuzione stabile
def test_check_distribution_shift_not_detected():
    """
    Verify that check_distribution_shift() does not detect shift on stable distribution.

    Returns:
        None
    """
    # Nearly identical distribution to baseline
    current = {"negative": 0.33, "neutral": 0.48, "positive": 0.19}
    shifts = check_distribution_shift(current)
    assert len(shifts) == 0

# Verifica che run_monitoring_report() gestisca correttamente un log vuoto
def test_run_monitoring_report_no_data(tmp_path):
    """
    Verify that run_monitoring_report() handles an empty log correctly.

    Returns:
        None
    """
    log_path = str(tmp_path / "empty_log.csv")
    os.environ["LOG_FILE"] = log_path

    # Crea un CSV vuoto con solo l'intestazione
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "text", "predicted_label", "confidence"])
        writer.writeheader()

    # Non deve sollevare eccezioni
    run_monitoring_report(hours=24)