import os
import sys
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.model import load_classifier, predict

DATASET_NAME = os.getenv("DATASET_NAME", "tweet_eval")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "sentiment")
HF_REPO = os.getenv("HF_REPO")
HF_TOKEN = os.getenv("HF_TOKEN")
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def evaluate(model_path=None):
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # La valutazione viene effettuata sull'intero test set per rendere l'analisi più attendibile
    test_data = dataset["test"]

    classifier = load_classifier(model_path)

    y_true = [LABEL_MAP[l] for l in list(test_data["label"])]
    y_pred = predict(classifier, list(test_data["text"]))

    print(classification_report(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    # Macro F1 viene restituito perché più rappresentativo su dataset sbilanciato
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Macro F1: {macro_f1:.4f}")
    return macro_f1

def validate():
    print("Valutazione nuovo modello...")
    new_f1 = evaluate(model_path="./models/sentiment_model")
    print(f"Nuovo modello — Macro F1: {new_f1:.4f}")

    try:
        print("Valutazione modello in produzione...")
        prod_f1 = evaluate(model_path=HF_REPO)
        print(f"Modello in produzione — Macro F1: {prod_f1:.4f}")

        if new_f1 > prod_f1:
            print(f"[OK] Nuovo modello migliore ({new_f1:.4f} > {prod_f1:.4f}). Procedo con il deploy.")
        else:
            print(f"[FAIL] Nuovo modello non migliore ({new_f1:.4f} <= {prod_f1:.4f}). Deploy annullato.")
            sys.exit(1)

    except Exception:
        print("[INFO] Nessun modello in produzione trovato. Procedo con il deploy.")

if __name__ == "__main__":
    mode = os.getenv("EVALUATE_MODE", "evaluate")
    model_path = os.getenv("MODEL_PATH", "./models/sentiment_model")
    if mode == "validate":
        validate()
    else:
        evaluate(model_path)