import pytest
from src.model import load_classifier, predict


# Verifica che predict() riconosca correttamente testi positivi
def test_predict_positive():
    """
    Verify that predict() correctly recognizes a positive text.

    Returns:
        None
    """
    classifier = load_classifier()
    result = predict(classifier, "I love this, it's amazing!")
    assert result[0] == "positive"

# Verifica che predict() riconosca correttamente testi negativi
def test_predict_negative():
    """
    Verify that predict() correctly recognizes a negative text.

    Returns:
        None
    """
    classifier = load_classifier()
    result = predict(classifier, "This is terrible, I hate it.")
    assert result[0] == "negative"


# Verifica che predict() gestisca correttamente un batch di testi
def test_predict_batch():
    """
    Verify that predict() correctly handles a batch of texts.

    Returns:
        None
    """
    classifier = load_classifier()
    texts = ["I love this!", "This is awful.", "It is what it is."]
    results = predict(classifier, texts)
    assert len(results) == 3
    assert all(r in ["positive", "negative", "neutral"] for r in results)

# Verifica che predict() gestisca correttamente un singolo testo come stringa
def test_predict_single_string():
    """
    Verify that predict() correctly handles a single text as a string.

    Returns:
        None
    """
    classifier = load_classifier()
    result = predict(classifier, "This is a test.")
    assert isinstance(result, list)
    assert len(result) == 1

# Verifica che predict() gestisca testi lunghi senza errori (truncation a 512 token)
def test_predict_long_text():
    """
    Verify that predict() handles long texts without errors (truncation at 512 tokens).

    Returns:
        None
    """
    classifier = load_classifier()
    long_text = "This is a very long text. " * 100
    result = predict(classifier, long_text)
    assert result[0] in ["positive", "negative", "neutral"]

# Verifica che load_classifier() carichi il modello di default senza errori
def test_load_classifier_default():
    """
    Verify that load_classifier() loads the default model without errors.

    Returns:
        None
    """
    classifier = load_classifier()
    assert classifier is not None

# Verifica che load_classifier() sollevi un errore su path inesistente
def test_load_classifier_invalid_path():
    """
    Verify that load_classifier() raises an error on a non-existent path.

    Returns:
        None
    """
    with pytest.raises(Exception):
        load_classifier("./non_existent_model")