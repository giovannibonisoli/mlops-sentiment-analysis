# ============================================================
# DRIFT DETECTION PER MODELLI DI SENTIMENT ANALYSIS
# ============================================================
# SCOPO: Rilevare deterioramento del modello in produzione attraverso:
#   - PSI (Population Stability Index) sulle confidence scores
#   - Shift nella distribuzione delle classi sentiment
#   - Abbassamento della confidenza media del modello
#
# REGOLE DI RETRAINING:
#   - Automatico (alert): PSI > 0.2 OPPURE shift classe > 15%
#   - Manuale (warning): confidenza media < 0.65 (richiede intervento umano)
#
# SOGLIE UTILIZZATE:
#   - PSI < 0.1:  distribuzione stabile
#   - PSI 0.1-0.2: leggero drift, monitorare
#   - PSI > 0.2:  drift significativo, retraining necessario
#   - Shift 15%:  soglia empirica per fluttuazioni giornaliere normali
#   - Confidence 0.65: sotto questa soglia il modello è mediamente insicuro
# ============================================================

import os
import csv
import numpy as np
from collections import Counter
from datetime import datetime, timedelta, timezone

# ============================================================
# COSTANTI DI CONFIGURAZIONE
# ============================================================

# Path del file CSV contenente i log delle predizioni
# Formato atteso: colonne "timestamp", "predicted_label", "confidence"
LOG_FILE = os.getenv("LOG_FILE", "./monitoring/predictions_log/predictions_log.csv")

# Soglia critica per la confidenza media del modello
# Valori inferiori indicano che il modello è sistematicamente insicuro
CONFIDENCE_THRESHOLD  = float(os.getenv("CONFIDENCE_THRESHOLD", 0.65))

# Soglie PSI (Population Stability Index)
PSI_THRESHOLD_WARNING = 0.1   # Leggero drift: monitorare ma non intervenire
PSI_THRESHOLD_ALERT = 0.2     # Drift grave: retraining necessario

# Distribuzione attesa delle classi sentiment, calcolata sul validation set
# Valori ottenuti da tweet_eval/sentiment con 1000 campioni, seed=42
# Serve come baseline per rilevare cambiamenti nel comportamento degli utenti
BASELINE_DISTRIBUTION = {
    "negative": 0.323,
    "neutral":  0.483,
    "positive": 0.193
}

# ============================================================
# CARICAMENTO LOG RECENTI
# ============================================================
# Carica le predizioni delle ultime N ore dal file CSV.
# Il timestamp viene normalizzato in UTC se non ha timezone.
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


# ============================================================
# CARICAMENTO BASELINE PER PSI
# ============================================================
# Carica i log delle 24-48 ore fa come riferimento per il calcolo del PSI.
# NOTA: In produzione, questa funzione caricherebbe un file statico
# con le confidence scores del validation set al momento del training.
# L'implementazione attuale è una semplificazione DIDATTICA.
def load_baseline_logs(hours: int = 24) -> list[dict[str, str]]:
    """
    Load older predictions as baseline reference.

    Args:
        hours (int, optional):
            Time window width for the baseline period.
            Defaults to 24.

    Returns:
        list[dict[str, str]]:
            List of baseline log rows from the CSV file.
    """

    # Baseline periodo delle 24-48 ore fa (non le ultime 24 ore)
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


# ============================================================
# CALCOLO DISTRIBUZIONE SENTIMENT
# ============================================================
# Calcola la percentuale di ciascuna classe sentiment nei log forniti.
# Restituisce un dizionario label -> percentuale (arrotondata a 3 decimali).
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

# ============================================================
# CALCOLO CONFIDENZA MEDIA
# ============================================================
# Calcola la confidenza media delle predizioni.
# Restituisce 0.0 se la lista è vuota, altrimenti media arrotondata a 4 decimali.
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

# ============================================================
# CALCOLO POPULATION STABILITY INDEX (PSI)
# ============================================================
# Calcola il PSI tra due distribuzioni di confidence scores.
# Formula: Σ ( (%_current_i - %_baseline_i) * ln(%_current_i / %_baseline_i) )
# Interpretazione:
#   - PSI < 0.1:  stabile
#   - PSI 0.1-0.2: leggero drift (monitorare)
#   - PSI > 0.2:  drift significativo (retraining necessario)
# Usa le confidence scores perché danno segnali più precoci del cambio dei label.
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

    # Crea breakpoints equidistanti tra 0 e 1
    breakpoints = np.linspace(0, 1, bins + 1)

    # Conta quanti punteggi cadono in ogni bin
    baseline_counts = np.histogram(baseline_scores, bins=breakpoints)[0]
    current_counts  = np.histogram(current_scores,  bins=breakpoints)[0]

    # Epsilon per evitare divisioni per zero e log(0)
    epsilon = 1e-8
    baseline_pct = (baseline_counts + epsilon) / (len(baseline_scores) + epsilon)
    current_pct  = (current_counts  + epsilon) / (len(current_scores)  + epsilon)

    # Calcolo PSI
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return round(float(psi), 4)

# ============================================================
# RILEVAZIONE SHIFT DISTRIBUZIONE CLASSI
# ============================================================
# Confronta la distribuzione corrente con quella baseline.
# Segnala shift quando la differenza supera il 15% (soglia empirica).
# Variazioni inferiori sono considerate normali fluttuazioni giornaliere.
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
        if delta > 0.15: # Soglia del 15%
            shifts[label] = {
                "baseline": baseline_pct,
                "current":  current_pct,
                "delta":    round(delta, 3)
            }
    return shifts


# ============================================================
# REPORT DI MONITORAGGIO
# ============================================================
# Genera un report completo con:
#   - numero totale di predizioni analizzate
#   - distribuzione sentiment corrente
#   - confidenza media
#   - shift distribuzione classi
#   - PSI sulle confidence
#   - raccomandazioni per retraining
#
# REGOLE:
#   - Retraining automatico: PSI > 0.2 OPPURE shift classe > 15%
#   - Solo warning (non retraining): confidenza media < 0.65
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

    Automatic retraining is triggered only for severe conditions:
    - PSI > 0.2 (significant data drift)
    - Sentiment distribution shift > 15% on any class

    A confidence drop alone is not considered severe enough for
    automatic retraining and requires manual evaluation instead.

    Args:
        hours (int, optional):
            Number of recent hours to analyze.
            Defaults to 24.

    Returns:
        None
    """

    # Caricamento dati
    current_rows = load_recent_logs(hours=hours)
    baseline_rows = load_baseline_logs(hours=hours)

    # Se non ci sono dati recenti, non si può fare monitoring
    if not current_rows:
        print("No predictions in the last hours.")
        return

    # Calcolo metriche di base
    distribution = compute_sentiment_distribution(current_rows)
    avg_confidence = compute_avg_confidence(current_rows)
    shifts = check_distribution_shift(distribution)

    # ===== STAMPA REPORT =====
    print(f"\n=== Monitoring Report — last {hours}h ===")
    print(f"Total predictions: {len(current_rows)}")
    print(f"\nSentiment distribution:")
    for label, pct in distribution.items():
        print(f" {label}: {pct:.1%}")
    print(f"\nAverage confidence: {avg_confidence:.2f}")

    # ===== 1. VERIFICA CONFIDENZA BASSA =====
    # Nota: questo è solo un WARNING, non attiva retraining automatico
    retraining_needed  = False
    retraining_reasons = []

    if avg_confidence < CONFIDENCE_THRESHOLD:
        print(f"\n[ALERT] Average confidence below threshold ({CONFIDENCE_THRESHOLD})")
        retraining_reasons.append(
            f"average confidence {avg_confidence} below threshold {CONFIDENCE_THRESHOLD}"
        )
        print("[ACTION REQUIRED] Manual review recommended.")
        print("To trigger retraining, run the CI workflow manually from GitHub Actions.")

    # ===== 2. VERIFICA SHIFT DISTRIBUZIONE CLASSI =====
    # Shift > 15% indica un cambiamento significativo nel tipo di utenti
    if shifts:
        print(f"\n[ALERT] Sentiment distribution shift detected:")
        for label, info in shifts.items():
            print(f"  {label}: baseline {info['baseline']:.1%} → current {info['current']:.1%} (Δ {info['delta']:.1%})")
        retraining_needed = True
        retraining_reasons.append(f"distribution shift: {list(shifts.keys())}")
    else:
        print("\n[OK] Sentiment distribution stable.")

    # ===== 3. CALCOLO PSI (DRIFT SULLE CONFIDENCE) =====
    if baseline_rows:
        baseline_scores = [float(r["confidence"]) for r in baseline_rows]
        current_scores = [float(r["confidence"]) for r in current_rows]
        psi = compute_psi(baseline_scores, current_scores)

        print(f"\nPSI (confidence scores): {psi}")
        if psi > PSI_THRESHOLD_ALERT:
            print("[ALERT] Significant drift detected.")
            retraining_needed = True
            retraining_reasons.append(f"PSI {psi} above threshold {PSI_THRESHOLD_ALERT}")
        elif psi > PSI_THRESHOLD_WARNING:
            print("[WARNING] Slight drift detected. Manual review recommended.")
        else:
            print("[OK] No significant drift.")
    else:
        print("\n[INFO] Insufficient baseline data for PSI calculation.")

    # ===== 4. RACCOMANDAZIONE FINALE =====
    # Se una delle condizioni severe è vera, suggerisci retraining
    if retraining_needed:
        reason = " | ".join(retraining_reasons)
        print(f"\n[ACTION REQUIRED] Retraining recommended.")
        print(f"Reason: {reason}")
        print("To trigger retraining, run the CI workflow manually from GitHub Actions.")
    else:
        print("\n[OK] Model appears stable. No retraining needed.")

if __name__ == "__main__":
    run_monitoring_report()