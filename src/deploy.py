import os
from pathlib import Path
from src.model import load_model_and_tokenizer

MODEL_PATH = os.getenv("MODEL_PATH", "./models/sentiment_model")
HF_REPO    = os.environ["HF_REPO"]
HF_TOKEN   = os.environ["HF_TOKEN"]

# Carica il modello e il tokenizer addestrato che sono stati precedentemente 
# salvati localmente  e li pubblica sull'Hugging Face Hub
def deploy() -> None:
    """
    Deploy the trained sentiment-analysis model to
    the Hugging Face Hub.

    The function:
    - loads the locally saved model and tokenizer
    - uploads both artifacts to the configured
      Hugging Face repository

    Environment variables:
        MODEL_PATH:
            Local path of the trained model directory.

        HF_REPO:
            Hugging Face repository identifier.

        HF_TOKEN:
            Hugging Face authentication token.

    Returns:
        None
    """
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    baseline_path = Path(MODEL_PATH) / "baseline_distribution.json"
    if baseline_path.exists():
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(baseline_path),
            path_in_repo="baseline_distribution.json",
            repo_id=HF_REPO,
            token=HF_TOKEN,
        )
    print(f"Deploy completato su {HF_REPO}")

if __name__ == "__main__":
    deploy()