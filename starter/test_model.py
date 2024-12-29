import pytest
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

model = pickle.load(open('model/model.pkl', 'rb'))
encoder = pickle.load(open('model/encoder.pkl', 'rb'))
lb = pickle.load(open('model/lb.pkl', 'rb'))


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

def test_model_output_size():
    X_test = np.array([[1, 2], [3, 4], [5, 6]])
    preds = inference(model, X_test)
    assert preds.shape == (3,)

def test_model_predictions():
    feature = {
        "age": 42,
        "workclass": "Private",
        "fnlgt": 116632,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }
    feature = {key.replace('_', '-'): [value] for key, value in feature.__dict__.items()}
    data = pd.DataFrame.from_dict(feature)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, _, _, _ = process_data(data, categorical_features=cat_features, label=None, 
        training=False, encoder=encoder, lb=lb)
    pred = inference(model, X)[0]
    return {'predict':'Less than or equal to 50K'} if pred == 0 else {'predict':'Greater than 50K'}

def test_preprocessing():
    X_test = np.array([[1, 2], [3, 4]])
    X_transformed = encoder.transform(X_test)
    assert X_transformed.shape == (2, X_test.shape[1]) 

if __name__ == "__main__":
    pytest.main()