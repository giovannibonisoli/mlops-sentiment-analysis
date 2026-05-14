import os
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import (
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from src.model import load_model_and_tokenizer

# Modello base
BASE_MODEL  = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Repository hugging face del modello in produzione
HF_REPO = os.getenv("HF_REPO", "sentiment-model")

# Questa directory verrà creata se non esiste e verrà usata per salvare temporaneamente il modello che sarà poi usato nello step di valutazione
OUTPUT_DIR = "./models/sentiment_model"

# Di default viene usato il dataset tweet_eval con la configurazione sentiment
DATASET_NAME = os.getenv("DATASET_NAME", "tweet_eval")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "sentiment")

# Il numero di dati di allenamento viene impostato a 1000
# come compromesso tra tempo di esecuzione in CI e qualità del training.
TRAIN_SAMPLES = int(os.getenv("TRAIN_SAMPLES", 1000))

# Il numero di dati di validazione è impostato a 1000 su un validation set di ~1500 esempi
# per garantire una stima affidabile delle metriche senza usare l'intero set.
VALIDATION_SAMPLES = int(os.getenv("VALIDATION_SAMPLES", 1000))

# Il numero di epoche è settato a 3, che rappresenta un valore standard per il fine-tuning
# di modelli transformer pre-addestrati, sufficiente a specializzare il modello sui nuovi
# dati senza incorrere in overfitting.
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))

# Seed per la randomizzazione del dataset
TRAIN_SEED = int(os.getenv("TRAIN_SEED", 42))

# TOKENIZZAZIONE IL TESTO
def tokenize(batch, tokenizer):

    # L'impostazione prevede di troncare il testo se supera la lunghezza massima di 512 token perché è la lunghezza massima supportata dal modello base
    # Il padding è impostato a max_length per garantire che tutti i batch abbiano la stessa lunghezza
    return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Le metriche scelte sono accuracy e macro F1, comunemente usate per la classificazione di testi
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


def train() -> None:
    """
    Train and evaluate the sentiment classification model.

    The function:
    - loads the dataset
    - prepares training and validation splits
    - tokenizes the data
    - configures Hugging Face Trainer arguments
    - trains the model
    - evaluates performance metrics
    - saves the trained model and tokenizer

    Returns:
        None
    """
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # Caricamento dei dati di training e valutazione
    train_data = dataset["train"].shuffle(seed=TRAIN_SEED).select(range(TRAIN_SAMPLES))
    eval_data  = dataset["validation"].shuffle(seed=TRAIN_SEED).select(range(VALIDATION_SAMPLES))

    # Caricamento del modello e del tokenizer
    if HF_REPO:
        try:
            api = HfApi()
            api.repo_info(repo_id=HF_REPO)
            print(f"Modello trovato su {HF_REPO}!")
            model, tokenizer = load_model_and_tokenizer(HF_REPO)
        except Exception:
            print(f"Nessun modello trovato su HuggingFace Hub. Utilizzo {BASE_MODEL}")
            model, tokenizer = load_model_and_tokenizer()
    else:
        model, tokenizer = load_model_and_tokenizer()

    # Tokenizzazione dei dati di training e valutazione
    train_data = train_data.map(lambda b: tokenize(b, tokenizer), batched=True)
    eval_data  = eval_data.map(lambda b: tokenize(b, tokenizer), batched=True)

    # Impostazione del formato dei dati di training e valutazione
    train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_data.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

    # Impostazione degli argomenti per l'allenamento
    # Viene scelto 16 come batch size in quanto standard per il fine-tuning di modelli transformer pre-addestrati.
    # Tramite eval_strategy="epoch" e save_strategy="epoch" il modello viene valutato e salvato alla fine di ogni epoca.
    # Tramite load_best_model_at_end=True il modello con le migliori metriche viene caricato alla fine dell'allenamento,
    # Questo viene selezionato in base a macro_f1, poiché in caso di dataset sbilanciato e l'accuracy rischia di essere fuorviante. 
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none"
    )

    # Creazione del trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics
    )

    # Allenamento del modello
    trainer.train()

    # Valutazione del modello
    metrics = trainer.evaluate()
    print(f"Accuracy: {metrics['eval_accuracy']:.2f}")
    print(f"Macro F1: {metrics['eval_macro_f1']:.2f}")

    # salvataggio temporaneo del modello allenato
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    train()