# Script to train machine learning model.
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Add code to load in the data.
data = pd.read_csv('./starter/data/cleaned_census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save the model
pickle.dump(model, open("./starter/model/model.pkl", 'wb'))
pickle.dump(encoder, open('./starter/model/encoder.pkl', 'wb'))
pickle.dump(lb, open('./starter/model/lb.pkl', 'wb'))

# Evaluate the model
preds = inference(model, X_test)
print('precision: {}, recall: {}, fbeta: {}'.format(
    *compute_model_metrics(y_test, preds)
))