import os
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score

from src.model import load_classifier, predict

DATASET_NAME   = os.getenv("DATASET_NAME", "tweet_eval")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "sentiment")
LABEL_MAP      = {0: "negative", 1: "neutral", 2: "positive"}

def evaluate(model_path=None):
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # Questa volta, la valutazione viene effettuata sull'intero dataset per rendere l'analisi più attendibile
    test_data = dataset["test"]

    # utilizzo la funzione load_classifier per prendere il modello allenato
    classifier = load_classifier(model_path)

    y_true = [LABEL_MAP[l] for l in list(test_data["label"])]
    y_pred = predict(classifier, list(test_data["text"]))

    print(classification_report(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy

if __name__ == "__main__":
    model_path = os.getenv("MODEL_PATH", "./models/sentiment_model")
    evaluate(model_path)