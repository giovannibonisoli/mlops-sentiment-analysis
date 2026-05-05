from src.model import load_classifier, predict

def test_predict_positive():
    classifier = load_classifier()
    result = predict(classifier, "I love this, it's amazing!")
    assert result[0] == "positive"

def test_predict_negative():
    classifier = load_classifier()
    result = predict(classifier, "This is terrible, I hate it.")
    assert result[0] == "negative"

def test_predict_batch():
    classifier = load_classifier()
    texts = ["I love this!", "This is awful.", "It is what it is."]
    results = predict(classifier, texts)
    assert len(results) == 3
    assert all(r in ["positive", "negative", "neutral"] for r in results)