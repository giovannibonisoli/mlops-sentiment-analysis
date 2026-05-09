import os
import csv
import numpy as np
from collections import Counter
from datetime import datetime, timedelta, timezone

LOG_FILE = os.getenv("LOG_FILE", "./monitoring/predictions_log.csv")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.65))
PSI_THRESHOLD_WARNING = 0.1   # leggero drift
PSI_THRESHOLD_ALERT = 0.2   # drift significativo, considerare retraining

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
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                rows.append(row)
    return rows

def load_baseline_logs(hours: int = 24) -> list[dict[str, str]]:
    """
    Load older predictions as baseline reference.

    In production, this would be the validation set from initial training.

    Args:
        hours (int, optional):
            Time window width for the baseline period.
            Defaults to 24.

    Returns:
        list[dict[str, str]]:
            List of baseline log rows from the CSV file.
    """
    cutoff_end   = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_start = cutoff_end - timedelta(hours=hours)
    rows = []
    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if cutoff_start <= ts < cutoff_end:
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

def compute_psi(baseline_scores: list[float], current_scores: list[float], bins: int = 10) -> float:
    """
    Compute the Population Stability Index (PSI) between baseline and current
    confidence score distributions.

    Interpretation:
        - PSI < 0.1:   stable distribution
        - PSI 0.1-0.2: slight drift, monitor
        - PSI > 0.2:   significant drift, consider retraining

    Uses confidence score distribution instead of labels alone because it
    provides an earlier signal of drift — the model may become uncertain
    before labels change noticeably.

    Args:
        baseline_scores (list[float]):
            List of baseline confidence scores.
        current_scores (list[float]):
            List of current confidence scores.
        bins (int, optional):
            Number of bins for histogram comparison. Defaults to 10.

    Returns:
        float:
            PSI value rounded to 4 decimal places.
    """
    baseline_scores = np.array(baseline_scores, dtype=float)
    current_scores  = np.array(current_scores,  dtype=float)

    breakpoints = np.linspace(0, 1, bins + 1)

    baseline_counts = np.histogram(baseline_scores, bins=breakpoints)[0]
    current_counts  = np.histogram(current_scores,  bins=breakpoints)[0]

    # Evita divisioni per zero aggiungendo un piccolo epsilon
    epsilon = 1e-8
    baseline_pct = (baseline_counts + epsilon) / (len(baseline_scores) + epsilon)
    current_pct  = (current_counts  + epsilon) / (len(current_scores)  + epsilon)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return round(float(psi), 4)

def check_distribution_shift(current_distribution: dict[str, float]) -> dict[str, dict[str, float]]:
    """
    Compare the current sentiment distribution against the baseline distribution.
    A shift is reported when the difference between the current and baseline
    percentage exceeds 15%.

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
    - PSI drift detection
    - retraining recommendations

    Args:
        hours (int, optional):
            Number of recent hours to analyze.
            Defaults to 24.

    Returns:
        None
    """
    current_rows  = load_recent_logs(hours=hours)
    baseline_rows = load_baseline_logs(hours=hours)

    if not current_rows:
        print("No predictions in the last hours.")
        return

    distribution   = compute_sentiment_distribution(current_rows)
    avg_confidence = compute_avg_confidence(current_rows)
    shifts         = check_distribution_shift(distribution)

    print(f"\n=== Monitoring Report — last {hours}h ===")
    print(f"Total predictions: {len(current_rows)}")
    print(f"\nSentiment distribution:")
    for label, pct in distribution.items():
        print(f"  {label}: {pct:.1%}")
    print(f"\nAverage confidence: {avg_confidence:.2f}")

    retraining_needed  = False
    retraining_reasons = []

    # Alert confidence
    if avg_confidence < CONFIDENCE_THRESHOLD:
        print(f"\n[ALERT] Average confidence below threshold ({CONFIDENCE_THRESHOLD})")
        retraining_needed = True
        retraining_reasons.append(f"average confidence {avg_confidence} below threshold {CONFIDENCE_THRESHOLD}")

    # Alert distribuzione
    if shifts:
        print(f"\n[ALERT] Sentiment distribution shift detected:")
        for label, info in shifts.items():
            print(f"  {label}: baseline {info['baseline']:.1%} → current {info['current']:.1%} (Δ {info['delta']:.1%})")
        retraining_needed = True
        retraining_reasons.append(f"distribution shift: {list(shifts.keys())}")
    else:
        print("\n[OK] Sentiment distribution stable.")

    # PSI
    if baseline_rows:
        baseline_scores = [float(r["confidence"]) for r in baseline_rows]
        current_scores  = [float(r["confidence"]) for r in current_rows]
        psi = compute_psi(baseline_scores, current_scores)

        print(f"\nPSI (confidence scores): {psi}")
        if psi > PSI_THRESHOLD_ALERT:
            print("[ALERT] Significant drift detected.")
            retraining_needed = True
            retraining_reasons.append(f"PSI {psi} above threshold {PSI_THRESHOLD_ALERT}")
        elif psi > PSI_THRESHOLD_WARNING:
            print("[WARNING] Slight drift detected. Monitor.")
        else:
            print("[OK] No significant drift.")
    else:
        print("\n[INFO] Insufficient baseline data for PSI calculation.")

    # Notifica retraining se necessario
    if retraining_needed:
        reason = " | ".join(retraining_reasons)
        print(f"\n[ACTION REQUIRED] Retraining recommended.")
        print(f"Reason: {reason}")
        print(f"To trigger retraining, run the CI workflow manually from GitHub Actions.")

if __name__ == "__main__":
    run_monitoring_report()