import pytest
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from starter.ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100  # default value

def test_compute_model_metrics():
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 0.5
    assert fbeta == 0.6666666666666666

def test_inference():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    X_test = np.array([[1, 2], [3, 4]])
    preds = inference(model, X_test)
    assert len(preds) == 2
    assert all(isinstance(pred, np.integer) for pred in preds)

if __name__ == "__main__":
    pytest.main()