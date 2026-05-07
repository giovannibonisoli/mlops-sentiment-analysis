import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = os.getenv("MODEL_PATH", "./models/sentiment_model")
HF_REPO    = os.environ["HF_REPO"]
HF_TOKEN   = os.environ["HF_TOKEN"]

def deploy():
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"Deploy completato su {HF_REPO}")

if __name__ == "__main__":
    deploy()