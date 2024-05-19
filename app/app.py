from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import shap

CSV_FILE_NAME = "../X_test.csv"
CLASSIFIER_FILE_NAME = "../classifier.pkl"
ACCEPTANCE = 0.51


def load_data(csv_file_name, classifier_file_name) :
    df = pd.read_csv(csv_file_name)
    with open(classifier_file_name, "rb") as file :
        classifier = pickle.load(file)
    explainer = shap.Explainer(classifier, df)
    return df, classifier, explainer

def read_request(data, df) :
    index = data["selected_index"]
    row = df.iloc[index : index + 1]
    max_display = data["shap_max_display"]
    return row, max_display

def get_shap_values(row, explainer, shap_max_display):
    shap_values = explainer.shap_values(row)
    top_features_indices = np.argsort(np.abs(shap_values[0]))[::-1][:shap_max_display]
    top_features = row.columns[top_features_indices].tolist()
    top_features_values = row.values[0][top_features_indices].tolist()
    top_shap_values = shap_values[0][top_features_indices].tolist()
    return {"top_features": top_features, "top_features_values": top_features_values, "top_shap_values": top_shap_values}

def get_prediction(row, classifier, acceptance) :
    proba = classifier.predict_proba(row)[0, 1]
    prediction = int(proba > acceptance)
    return prediction


def create_app() :

    df, classifier, explainer = load_data(CSV_FILE_NAME, CLASSIFIER_FILE_NAME)
    
    app = Flask(__name__)
    
    @app.route("/api/predict", methods = ["POST"])
    def predict() :
        row, shap_max_display = read_request(request.json, df)
        output = get_shap_values(row, explainer, shap_max_display)
        output["prediction"] = get_prediction(row, classifier, ACCEPTANCE)
        return jsonify(output)
    
    app.run(debug = True)


if __name__ == "__main__":
    create_app()





"""
def create_app(df, classifier, explainer) :

    app = Flask(__name__)
  
    @app.route("/api/predict", methods = ["POST"])
    def predict():
        data = request.json
        index = data.get("index")
        shap_max_display = data.get("shap_max_display")
        row = df.iloc[index : index + 1]
        proba = classifier.predict_proba(row)[0, 1]
        output = {"prediction" : int(proba > ACCEPTANCE)}
        shap_values = explainer.shap_values(row)
        top_features_indices = np.argsort(np.abs(shap_values[0]))[::-1][:shap_max_display]
        output["top_features"] = df.columns[top_features_indices].tolist()
        output["top_features_values"] = row.values[0][top_features_indices].tolist()
        output["top_shap_values"] = shap_values[0][top_features_indices].tolist()
        return jsonify(output)

    return app

def main() :
    df = pd.read_csv(CSV_FILE_NAME)
    with open(CLASSIFIER_FILE_NAME, "rb") as file:
        classifier = pickle.load(file)
    explainer = shap.Explainer(classifier, df)
    app = create_app(df, classifier, explainer)
    app.run(debug = True)

if __name__ == "__main__":
    main()
"""