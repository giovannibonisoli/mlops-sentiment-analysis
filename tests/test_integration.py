from datasets import load_dataset
from src.model import load_classifier, predict
from src.evaluate import evaluate

def test_pipeline_end_to_end():
    """Verifica che il classifier produca predizioni valide su dati reali."""
    dataset = load_dataset("tweet_eval", "sentiment")
    sample = dataset["test"].select(range(10))
    
    classifier = load_classifier()
    predictions = predict(classifier, sample["text"])
    
    assert len(predictions) == 10
    assert all(p in ["positive", "negative", "neutral"] for p in predictions)

def test_evaluate_returns_accuracy():
    """Verifica che evaluate() ritorni un valore di accuracy sensato."""
    accuracy = evaluate()
    assert 0.0 <= accuracy <= 1.0
    assert accuracy > 0.5  # soglia minima accettabile