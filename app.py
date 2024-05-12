from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import shap

CSV_FILE_NAME = "X_test.csv"
MODEL_FILE_NAME = "classifier.pkl"
ACCEPTANCE = 0.51

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
    with open(MODEL_FILE_NAME, "rb") as file:
        classifier = pickle.load(file)
    explainer = shap.Explainer(classifier, df)
    app = create_app(df, classifier, explainer)
    app.run(debug = True)

if __name__ == "__main__":
    main()
