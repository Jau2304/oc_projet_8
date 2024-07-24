from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import shap
import lightgbm as lgb

CSV_FILE_NAME = "data.csv"
CLASSIFIER_FILE_NAME = "classifier.pkl"
ACCEPTANCE = 0.5


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

def get_shap_values(row, explainer, shap_max_display, output) :
    shap_values = explainer.shap_values(row)
    indices = np.argsort(np.abs(shap_values[0]))[::-1][:shap_max_display]
    output["top_features"] = row.columns[indices].tolist()
    output["top_features_values"] = row.values[0][indices].tolist()
    output["top_shap_values"] = shap_values[0][indices].tolist()

def get_prediction(row, classifier, acceptance, output) :
    proba = classifier.predict_proba(row)[0, 1]
    output["pred_proba"] = proba
    output["acceptance"] = acceptance
    output["pred_binary"] = int(proba > acceptance)

def get_feature_importance(classifier, df) :
    importances = classifier.feature_importances_
    output = pd.DataFrame({"feature": df.columns, "importance": importances})
    return output.sort_values(by = "importance", ascending = False)

def create_app() :

    df, classifier, explainer = load_data(CSV_FILE_NAME, CLASSIFIER_FILE_NAME)
    
    app = Flask(__name__)

    @app.route("/api/predict", methods = ["POST"])
    def predict() :
        row, shap_max_display = read_request(request.json, df)
        output = {}
        get_shap_values(row, explainer, shap_max_display, output)
        get_prediction(row, classifier, ACCEPTANCE, output)
        return jsonify(output)
    
    @app.route("/api/importance", methods = ["POST"])
    def importance() :
        data = request.json
        max_display = data.get("max_display")
        feature_importances = get_feature_importance(classifier, df)
        top_features = feature_importances.head(max_display)
        return jsonify({
            "features": top_features["feature"].tolist(),
            "importances": top_features["importance"].tolist()
        })

    return app


if __name__ == "__main__" :
    app = create_app()
    app.run(host = "0.0.0.0", port=5000)