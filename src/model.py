from transformers import pipeline
from transformers.pipelines import TextClassificationPipeline

from typing import Iterable

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


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