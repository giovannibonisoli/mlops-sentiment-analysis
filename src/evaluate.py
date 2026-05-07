import os
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score

from src.model import load_classifier, predict

DATASET_NAME   = os.getenv("DATASET_NAME", "tweet_eval")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "sentiment")
LABEL_MAP      = {0: "negative", 1: "neutral", 2: "positive"}

def evaluate(model_path=None):
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    test_data = dataset["test"]

    classifier = load_classifier(model_path)

    y_true = [LABEL_MAP[l] for l in test_data["label"]]
    y_pred = predict(classifier, test_data["text"])

    print(classification_report(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy

if __name__ == "__main__":
    model_path = os.getenv("MODEL_PATH", "./models/sentiment_model")
    evaluate(model_path)