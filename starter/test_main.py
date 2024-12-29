from fastapi.testclient import TestClient
from main import app, Feature

client = TestClient(app)

def test_get_hello_world():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello World!"

def test_post_predict_less_than_equal_50k():
    feature = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/", json=feature)
    assert response.status_code == 200
    assert response.json() == {'predict': 'Less than or equal to 50K'}

def test_post_predict_greater_than_50k():
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
    response = client.post("/", json=feature)
    assert response.status_code == 200
    assert response.json() == {'predict': 'Greater than 50K'}