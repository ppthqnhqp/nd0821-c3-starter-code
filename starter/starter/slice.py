import sys
import os
import pandas as pd
import pickle
from ml.data import process_data
from ml.model import compute_model_metrics, inference

def evaluate_model_slices_performance(model, encoder, lb, data, features, categorical_features=[]):
    output_file = os.path.join("starter/starter/slice_output.txt")
    X, y, _, _ = process_data(data, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb)
    preds = inference(model, X)
    with open(output_file, "a+") as f:
        for feature in features:
            f.write(f"Evaluating model performance on slices of data for feature: {feature}\n")

            for slice_value in data[feature].unique():
                slice_index = data.index[data[feature] == slice_value]
                f.write(f"\t* Feature {feature}: {slice_value}\n")
                f.write(f"\t\tData points: {len(slice_index)}\n")
                precision, recall, fbeta = compute_model_metrics(y[slice_index], preds[slice_index])
                f.write(f"\t\tPrecision: {precision}, recall: {recall}, fbeta: {fbeta}\n")
                
            f.write('-' * 50 + '\n')

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

if __name__ == '__main__':
    if os.path.exists("starter/starter/slice_output.txt"):
        os.remove("starter/starter/slice_output.txt")
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
    data = pd.read_csv('./starter/data/cleaned_census.csv')
    model = load_pickle('./starter/model/model.pkl')
    encoder = load_pickle('./starter/model/encoder.pkl')
    lb = load_pickle('./starter/model/lb.pkl')

    features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    evaluate_model_slices_performance(model, encoder, lb, data, features, categorical_features=cat_features)
