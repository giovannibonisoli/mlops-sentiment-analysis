from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import TextClassificationPipeline
from transformers import PreTrainedModel, PreTrainedTokenizer

from typing import Iterable

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Carica un modello transformer e il relativo tokenizer per l'addestramento 
# Se model_path è None, utilizza il modello predefinito da Hugging Face Hub
def load_model_and_tokenizer(model_path: str | None = None) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer for training.

    Args:
        model_path (str | None, optional):
            Local path or Hugging Face model identifier.
            If None, the default MODEL_NAME is used.

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizer]:
            A tuple containing the model and tokenizer.
    """
    source = model_path if model_path else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=3)
    return model, tokenizer

# Crea e restituisce una pipeline Hugging Face per sentiment-analysis pronta all'uso
# Il modello e il tokenizer vengono caricati dalla stessa sorgente (locale o Hub)
def load_classifier(model_path: str | None = None) -> TextClassificationPipeline:
    """
    Load a Hugging Face sentiment-analysis pipeline.

    Args:
        model_path (str | None, optional):
            Local path or Hugging Face model identifier.
            If None, the default MODEL_NAME is used.

    Returns:
        TextClassificationPipeline:
            Configured sentiment-analysis pipeline.
    """

    source = model_path if model_path else MODEL_NAME

    return pipeline(
        "sentiment-analysis",
        model=source,
        tokenizer=source
    )

# Esegue predizioni sentimentali su uno o più testi usando il classificatore fornito
# Gestisce sia input singolo (stringa) che multiplo (iterabile) e restituisce i label in lowercase
def predict(classifier: TextClassificationPipeline, texts: str | Iterable[str]) -> list[str]:
    """
    Predict sentiment labels for one or more texts.

    Args:
        classifier (TextClassificationPipeline):
            Hugging Face sentiment-analysis pipeline.

        texts (str | Iterable[str]):
            Single text string or iterable of text strings
            to analyze.

    Returns:
        list[str]:
            List of predicted labels in lowercase format.
    """

    if isinstance(texts, str):
        texts = [texts]

    results = classifier(texts, truncation=True, max_length=512)

    return [r["label"].lower() for r in results]