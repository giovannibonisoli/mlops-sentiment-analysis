from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def load_classifier(model_path=None):
    source = model_path if model_path else MODEL_NAME
    return pipeline("sentiment-analysis", model=source, tokenizer=source)

def predict(classifier, texts):
    if isinstance(texts, str):
        texts = [texts]
    results = classifier(texts, truncation=True, max_length=512)
    return [r["label"].lower() for r in results]