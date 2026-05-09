import os
import sys
import time

from datasets import load_dataset, concatenate_datasets
from src.monitoring.monitor import log_predictions
from src.monitoring.drift import run_monitoring_report
from src.model import load_classifier

SIMULATE_SAMPLES = int(os.getenv("SIMULATE_SAMPLES", 100))
MONITORING_HOURS = int(os.getenv("MONITORING_HOURS", 24))
SIMULATE_SEED = int(time.time() / 3600)  # Cambia ogni ora


def simulate(samples: int = SIMULATE_SAMPLES) -> None:
    """
    Simulate incoming social media texts for monitoring.

    Uses the tweet_eval test set as data source. In a real system,
    these texts would come from monitored social media APIs.

    Args:
        samples (int, optional):
            Number of samples to simulate. Defaults to SIMULATE_SAMPLES.

    Returns:
        None
    """
    print("Loading dataset...")
    dataset = load_dataset("tweet_eval", "sentiment")
    test_data = dataset["test"].shuffle(seed=SIMULATE_SEED).select(range(samples))

    print("Loading model...")
    classifier = load_classifier()

    print(f"\nSimulating {samples} new texts...\n")
    log_predictions(list(test_data["text"]), classifier=classifier)
    print(f"{samples} predictions logged to {os.getenv('LOG_FILE', './monitoring/predictions_log.csv')}")


def simulate_drift(samples: int = SIMULATE_SAMPLES) -> None:
    """
    Simulate distribution drift by oversampling the negative class.

    Produces an artificially imbalanced distribution (70% negative)
    compared to the baseline (32% negative), simulating a reputational
    crisis and ensuring drift is detected.

    Args:
        samples (int, optional):
            Number of samples to simulate. Defaults to SIMULATE_SAMPLES.

    Returns:
        None
    """
    print("Loading dataset for drift simulation...")
    dataset = load_dataset("tweet_eval", "sentiment")
    test_data = dataset["test"]

    # Filtra per classe
    negative = test_data.filter(lambda x: x["label"] == 0)
    neutral  = test_data.filter(lambda x: x["label"] == 1)
    positive = test_data.filter(lambda x: x["label"] == 2)

    # Sovracampiona i negative per simulare una crisi reputazionale
    drifted = concatenate_datasets([
        negative.select(range(min(70, len(negative)))),
        neutral.select(range(min(20, len(neutral)))),
        positive.select(range(min(10, len(positive))))
    ]).shuffle(seed=SIMULATE_SEED)

    print("Loading model...")
    classifier = load_classifier()

    print(f"\nSimulating drift with {len(drifted)} imbalanced texts...\n")
    log_predictions(list(drifted["text"]), classifier=classifier)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "drift":
        simulate_drift()
    else:
        simulate()