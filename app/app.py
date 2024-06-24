from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import shap

CSV_FILE_NAME = "data.csv"
CLASSIFIER_FILE_PATH = "../models/best_model.pkl"
ACCEPTANCE_FILE_PATH = "../models/acceptance.txt"


def load_data(csv_file_name, classifier_file_path) :
    df = pd.read_csv(csv_file_name)
    with open(classifier_file_path, "rb") as file :
        classifier = pickle.load(file)
    explainer = shap.Explainer(classifier, df)
    return df, classifier, explainer

def read_request(data, df) :
    index = data["selected_index"]
    row = df.iloc[index : index + 1]
    max_display = data["shap_max_display"]
    return row, max_display

def get_shap_values(row, explainer, shap_max_display) :
    shap_values = explainer.shap_values(row)
    top_features_indices = np.argsort(np.abs(shap_values[0]))[::-1][:shap_max_display]
    top_features = row.columns[top_features_indices].tolist()
    top_features_values = row.values[0][top_features_indices].tolist()
    top_shap_values = shap_values[0][top_features_indices].tolist()
    return {"top_features": top_features, "top_features_values": top_features_values, "top_shap_values": top_shap_values}

def get_acceptance(acceptance_file_path, model_file_path) :
    with open(acceptance_file_path, "r") as file :
        lines = file.readlines()
    for line in lines :
        content = line.split(" : ")
        if content[0] == model_file_path :
            return float(content[1])

def get_prediction(row, classifier, acceptance) :
    proba = classifier.predict_proba(row)[0, 1]
    prediction = int(proba > acceptance)
    return prediction


def create_app() :

    df, classifier, explainer = load_data(CSV_FILE_NAME, CLASSIFIER_FILE_PATH)
    
    app = Flask(__name__)

    @app.route("/api/predict", methods = ["POST"])
    def predict() :
        row, shap_max_display = read_request(request.json, df)
        output = get_shap_values(row, explainer, shap_max_display)
        acceptance = get_acceptance(ACCEPTANCE_FILE_PATH, CLASSIFIER_FILE_PATH)
        output["prediction"] = get_prediction(row, classifier, acceptance)
        return jsonify(output)
    
    return app


if __name__ == "__main__" :
    app = create_app()
    app.run(host = "0.0.0.0", port=5000)