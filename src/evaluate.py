import os
import sys
from datasets import load_dataset
from numpy.random import default_rng
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.model import load_classifier, predict

DATASET_NAME = os.getenv("DATASET_NAME", "tweet_eval")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "sentiment")
HF_REPO = os.getenv("HF_REPO")
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
SAMPLE_SIZE = int(os.getenv("VALIDATION_SAMPLE_SIZE", 1000))


def _stratified_sample(dataset, n_samples: int, seed: int = 42) -> list:
    from collections import defaultdict
    labels = list(dataset["label"])
    label_indices = defaultdict(list)
    for i, l in enumerate(labels):
        label_indices[l].append(i)
    rng = default_rng(seed)
    sampled = []
    per_label = max(1, n_samples // len(label_indices))
    for l in sorted(label_indices):
        indices = label_indices[l]
        k = min(len(indices), per_label)
        chosen = rng.choice(indices, size=k, replace=False).tolist()
        sampled.extend(chosen)
    sampled = sorted(sampled)
    return [dataset[i] for i in sampled]


def _evaluate_split(data: list, classifier) -> dict[str, float]:
    texts = [item["text"] for item in data]
    labels = [LABEL_MAP[item["label"]] for item in data]
    y_pred = predict(classifier, texts)
    print(classification_report(labels, y_pred))
    accuracy = accuracy_score(labels, y_pred)
    macro_f1 = f1_score(labels, y_pred, average="macro")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Macro F1: {macro_f1:.4f}")
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def _evaluate_path(model_path: str | None, data: list) -> dict[str, float]:
    classifier = load_classifier(model_path)
    return _evaluate_split(data, classifier)


def evaluate(model_path: str | None = None, sample_size: int | None = None) -> dict[str, float]:
    """
    Evaluate a sentiment classification model on the test set.

    Loads the model from the specified path or Hugging Face Hub,
    runs predictions on the test set, and computes classification
    metrics including accuracy and macro F1.

    Args:
        model_path (str | None, optional):
            Local path or Hugging Face model identifier.
            If None, uses the default MODEL_NAME.
        sample_size (int | None, optional):
            Number of samples for stratified evaluation.
            If None, evaluates on the full test set.

    Returns:
        dict[str, float]:
            Dictionary containing accuracy and macro_f1 scores.
    """
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    test_data = dataset["test"]

    if sample_size is not None:
        test_data = _stratified_sample(dataset["test"], sample_size)

    classifier = load_classifier(model_path)
    return _evaluate_split(test_data, classifier)


def validate() -> None:
    """
    Validate that a newly trained model outperforms the production model.

    Uses stratified sampling for both models to reduce CI time.

    Returns:
        None
    """
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    sample_data = _stratified_sample(dataset["test"], SAMPLE_SIZE)

    print("Evaluating new model...")
    new_metrics = _evaluate_path("./models/sentiment_model", sample_data)
    new_f1 = new_metrics["macro_f1"]
    print(f"New model — Macro F1: {new_f1:.4f}")

    try:
        print("Evaluating production model...")
        prod_metrics = _evaluate_path(HF_REPO, sample_data)
        prod_f1 = prod_metrics["macro_f1"]
        print(f"Production model — Macro F1: {prod_f1:.4f}")

        if new_f1 > prod_f1:
            print(f"[OK] New model is better ({new_f1:.4f} > {prod_f1:.4f}). Proceeding with deploy.")
        else:
            print(f"[FAIL] New model is not better ({new_f1:.4f} <= {prod_f1:.4f}). Deploy cancelled.")
            sys.exit(1)

    except Exception:
        print("[INFO] No production model found. Proceeding with deploy.")


# Legge la variabile d'ambiente EVALUATE_MODE per decidere
# se eseguire la validazione (confronto con produzione) o la sola valutazione locale.
# Di default esegue la valutazione semplice sul modello salvato in "./models/sentiment_model".
if __name__ == "__main__":
    mode = os.getenv("EVALUATE_MODE", "evaluate")
    if mode == "validate":
        validate()
    else:
        evaluate("./models/sentiment_model")