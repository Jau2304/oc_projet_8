import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import requests

CSV_FILE_NAME = "../csv/preprocessed/app_test.csv"

def display_prediction(prediction) :
    if prediction == 0 :
        st.write("La demande d'emprunt est acceptée.")
    else :
        st.write("La demande d'emprunt est refusée.")

def display_waterfall(features, features_values, shap_values) :
    fig = plt.figure()
    explanation = shap.Explanation(values = np.array(shap_values), base_values = 0, data = features)
    waterfall = shap.plots.waterfall(explanation, max_display = len(features), show = False)
    labels = [f"{feature} = {value:.3f}" for feature, value in zip(features, features_values)]
    plt.gca().set_yticks(np.arange(len(labels)))
    plt.gca().set_yticklabels(labels)
    waterfall.set_size_inches(8, 12)
    st.pyplot(fig)

def main() :

    df = pd.read_csv(CSV_FILE_NAME)

    with open("../selected_features.txt", "r") as file :
        lines = file.readlines()

    selected_features = [line.strip() for line in lines]
    df = df[selected_features]

    st.title("Fichier client")
    st.write(df)

    st.title("Sélection du client")
    selected_index = st.number_input("Index du client :", min_value = 0, max_value = len(df) - 1)
    st.write(df.iloc[selected_index])

    st.title("Demande d'emprunt")
    if st.button("Lancer la simulation") :
        data = {"index": selected_index, "shap_max_display" : 10}
        response = requests.post("http://127.0.0.1:5000/api/predict", json = data)
        data = response.json()
        display_prediction(data["prediction"])
        display_waterfall(data["top_features"], data["top_features_values"], data["top_shap_values"])

if __name__ == "__main__" :
    main()