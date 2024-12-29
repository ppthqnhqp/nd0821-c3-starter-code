# Put the code for your API here.
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference


app = FastAPI()
model = pickle.load(open('model/model.pkl', 'rb'))
encoder = pickle.load(open('model/encoder.pkl', 'rb'))
lb = pickle.load(open('model/lb.pkl', 'rb'))


class Feature(BaseModel):
    # Using the first row of census.csv as sample
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')


@app.get("/")
async def hello_world():
  return "Hello World!"


@app.post("/")
async def predict(feature: Feature):
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