import os
import csv
from collections import Counter
from datetime import datetime, timedelta, timezone

LOG_FILE           = os.getenv("LOG_FILE", "./monitoring/predictions_log.csv")
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", 0.60))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.65))

# Distribuzione di riferimento dalla valutazione baseline
BASELINE_DISTRIBUTION = {
    "negative": 0.323,
    "neutral":  0.483,
    "positive": 0.193
}

def load_recent_logs(hours: int = 24) -> list[dict[str, str]]:
    """
    Load prediction logs from the last N hours.

    Args:
        hours (int, optional):
            Number of hours to look back from the
            current UTC time. Defaults to 24.

    Returns:
        list[dict[str, str]]:
            List of log rows loaded from the CSV file.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = []
    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp"])
            if ts >= cutoff:
                rows.append(row)
    return rows

def compute_sentiment_distribution(rows: list[dict[str, str]]) -> dict[str, float]:
    """
    Compute the percentage distribution of predicted sentiments.

    Args:
        rows (list[dict[str, str]]):
            Prediction log rows.

    Returns:
        dict[str, float]:
            Mapping between sentiment labels and
            percentage distribution values.
    """
    counts = Counter(row["predicted_label"] for row in rows)
    total  = sum(counts.values())
    return {label: round(count / total, 3) for label, count in counts.items()}

def compute_avg_confidence(rows: list[dict[str, str]]) -> float:
    """
    Compute the average confidence score of predictions.

    Args:
        rows (list[dict[str, str]]):
            Prediction log rows.

    Returns:
        float:
            Average prediction confidence rounded
            to 4 decimal places.
    """
    if not rows:
        return 0.0
    return round(sum(float(row["confidence"]) for row in rows) / len(rows), 4)

def check_distribution_shift(current_distribution: dict[str, float]) -> dict[str, dict[str, float]]:
    """
    Compare the current sentiment distribution against the baseline distribution.
    A shift is reported when the difference between the current and baseline percentage exceeds 15%.

    Args:
        current_distribution (dict[str, float]):
            Current sentiment distribution.

    Returns:
        dict[str, dict[str, float]]:
            Dictionary containing detected shifts
            for each affected sentiment label.
    """
    shifts = {}
    for label, baseline_pct in BASELINE_DISTRIBUTION.items():
        current_pct = current_distribution.get(label, 0.0)
        delta = abs(current_pct - baseline_pct)
        if delta > 0.15:
            shifts[label] = {
                "baseline": baseline_pct,
                "current":  current_pct,
                "delta":    round(delta, 3)
            }
    return shifts

def run_monitoring_report(hours: int = 24) -> None:
    """
    Generate a monitoring report for recent model predictions.

    The report includes:
    - total predictions
    - sentiment distribution
    - average confidence score
    - distribution shift detection
    - confidence threshold alerts

    Args:
        hours (int, optional):
            Number of recent hours to analyze.
            Defaults to 24.

    Returns:
        None
    """
    rows = load_recent_logs(hours=hours)

    if not rows:
        print("Nessuna predizione nelle ultime ore.")
        return

    distribution  = compute_sentiment_distribution(rows)
    avg_confidence = compute_avg_confidence(rows)
    shifts        = check_distribution_shift(distribution)

    print(f"\n=== Monitoring Report — ultime {hours}h ===")
    print(f"Predizioni totali: {len(rows)}")
    print(f"\nDistribuzione sentiment:")
    for label, pct in distribution.items():
        print(f"  {label}: {pct:.1%}")
    print(f"\nConfidence media: {avg_confidence:.2f}")

    if avg_confidence < CONFIDENCE_THRESHOLD:
        print(f"\n[ALERT] Confidence media sotto soglia ({CONFIDENCE_THRESHOLD})")
        print("  Il modello mostra incertezza sui dati recenti.")

    if shifts:
        print(f"\n[ALERT] Shift nella distribuzione del sentiment rilevato:")
        for label, info in shifts.items():
            print(f"  {label}: baseline {info['baseline']:.1%} → attuale {info['current']:.1%} (Δ {info['delta']:.1%})")
        print("  Considerare un retraining del modello.")
    else:
        print("\n[OK] Distribuzione sentiment stabile.")

if __name__ == "__main__":
    run_monitoring_report()