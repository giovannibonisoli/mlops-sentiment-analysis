import os
import sys
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.model import load_classifier, predict

DATASET_NAME = os.getenv("DATASET_NAME", "tweet_eval")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "sentiment")
HF_REPO = os.getenv("HF_REPO")
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


# FUNZIONE PER LA VALUTAZIONE 
# Carica il dataset di test, esegue le predizioni usando il classificatore specificato
# e restituisce un dizionario con accuratezza e macro F1, stampando anche un report dettagliato
def evaluate(model_path: str | None = None) -> dict[str, float]:
    """
    Evaluate a sentiment classification model on the test set.

    Loads the model from the specified path or Hugging Face Hub,
    runs predictions on the test set, and computes classification
    metrics including accuracy and macro F1.

    Args:
        model_path (str | None, optional):
            Local path or Hugging Face model identifier.
            If None, uses the default MODEL_NAME.

    Returns:
        dict[str, float]:
            Dictionary containing accuracy and macro_f1 scores.
    """
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # La valutazione viene effettuata sull'intero test set per maggiore affidabilità
    test_data = dataset["test"]

    classifier = load_classifier(model_path)

    y_true = [LABEL_MAP[l] for l in list(test_data["label"])]
    y_pred = predict(classifier, list(test_data["text"]))

    print(classification_report(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Macro F1: {macro_f1:.4f}")

    return {"accuracy": accuracy, "macro_f1": macro_f1}


# FUNZIONE PER LA VALIDAZIONE DI UN MODELLO
# Confronta le performance (macro F1) del modello appena addestrato con quelle del modello
# attualmente in produzione. Se il nuovo modello non è superiore, termina il processo con errore 
# che farà terminare la pipeline senza procedere con il deploy.
# Se non esiste ancora un modello in produzione, procede automaticamente con il deploy.
def validate() -> None:
    """
    Validate that a newly trained model outperforms the production model.

    Compares the macro F1 score of the local model against the production
    model on Hugging Face Hub. Exits with failure if the new model is not
    better.

    Returns:
        None
    """
    print("Evaluating new model...")
    new_metrics = evaluate(model_path="./models/sentiment_model")
    new_f1 = new_metrics["macro_f1"]
    print(f"New model — Macro F1: {new_f1:.4f}")

    try:
        print("Evaluating production model...")
        prod_metrics = evaluate(model_path=HF_REPO)
        prod_f1 = prod_metrics["macro_f1"]
        print(f"Production model — Macro F1: {prod_f1:.4f}")

        # Se
        if new_f1 > prod_f1:
            print(f"[OK] New model is better ({new_f1:.4f} > {prod_f1:.4f}). Proceeding with deploy.")
        else:
            print(f"[FAIL] New model is not better ({new_f1:.4f} <= {prod_f1:.4f}). Deploy cancelled.")
            sys.exit(1)

    except Exception:

        # Se non esiste nessun modello su Hugging Face, si procede al deploy del modello corrente
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