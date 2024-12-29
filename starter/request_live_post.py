import json
import requests

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

response = requests.post('https://nd0821-c3-starter-code-oozs.onrender.com', data=json.dumps(feature))

print(response.status_code)
print(response.json())