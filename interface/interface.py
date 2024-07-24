import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

CSV_FILE_NAME = "data.csv"
#API_URL = "http://localhost:5000/api/"
API_URL = "http://app:8000/api/"

FILTERS = ["(aucun filtre)", "ORGANIZATION_TYPE", "EMERGENCYSTATE_MODE",
           "NAME_HOUSING_TYPE", "NAME_EDUCATION_TYPE", "NAME_TYPE_SUITE",
           "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "OCCUPATION_TYPE",
           "WEEKDAY_APPR_PROCESS_START", "CODE_GENDER", "HOUSETYPE_MODE",
           "FONDKAPREMONT_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"]

def main():
    df, features = load_dataframe()
    customer = select_customer(df)
    display_feature(df, features, customer)
    compare_features(df, features, customer)
    display_feature_importance()
    predict_score(customer)

def load_dataframe() :
    df = pd.read_csv(CSV_FILE_NAME)
    features = list(df.columns)
    df = df[features]
    st.title("Fichier client")
    st.write(df)
    return df, features

def select_customer(df) :
    st.title("Sélection du client")
    label = "Index du client :"
    customer = st.number_input(label, min_value = 0, max_value = len(df) - 1)
    st.write(df.iloc[customer])
    return customer

def apply_filter(df, customer, key) :
    prefix = st.selectbox("Sélectionnez un filtre :", FILTERS, key = key)
    if prefix == "(aucun filtre)" :
        return df, ""
    features = []
    customer_feature = None
    for feature in df.columns :
        if feature.startswith(prefix) :
            features.append(feature)
            if df.iloc[customer][feature] == 1 :
                customer_feature = feature
    label = "Sélectionnez la valeur du filtre "
    if customer_feature == None :
        label += ":"
        default = 0
    else :
        label += f"(le client est dans la catégorie {customer_feature}) :"
        default = features.index(customer_feature)
    key += "value"
    feature = st.selectbox(label, features, index = default, key = key)
    return df[df[feature] == 1], f" (filtré avec {feature})"

def display_feature(df, features, customer) :
    st.title("Distribution des variables")
    label = "Sélectionnez une variable pour afficher sa distribution :"
    default = features.index("AMT_CREDIT")
    feature = st.selectbox(label, features, index = default)
    highlight = st.checkbox("Afficher le le client sélectionné")
    df_filter, title_suffix = apply_filter(df, customer, "univ")
    if feature :
        title = f"Distribution de {feature}" + title_suffix
        fig = px.histogram(df_filter, x = feature, title = title)
        fig.update_layout(bargap = 0.02)
        if highlight :
            value = df.iloc[customer][feature]
            fig.add_vline(x = value, line_width = 3, line_color = "red")
        st.plotly_chart(fig)

def compare_features(df, features, customer) :
    st.title("Comparaison des variables")
    label_x = "Sélectionnez la variable en abscisse :"
    default_x = features.index("AMT_CREDIT")
    feature_x = st.selectbox(label_x, features, index = default_x)
    label_y = "Sélectionnez la variable en ordonnée :"
    default_y = features.index("AMT_ANNUITY")
    feature_y = st.selectbox(label_y, features, index = default_y)
    highlight = st.checkbox("Afficher le le client sélectionné", key = "biv")
    df_filter, title_suffix = apply_filter(df, customer, "biv_filter")
    if feature_x and feature_y :
        title = f"Comparaison de {feature_x} et {feature_y}" + title_suffix
        fig = px.scatter(df_filter, x = feature_x, y = feature_y, title = title)
        if highlight :
            value_x = df.iloc[customer][feature_x]
            value_y = df.iloc[customer][feature_y]
            fig.add_trace(go.Scatter(
                x = [value_x], y = [value_y],
                mode = "markers", name = "client", showlegend = False,
                marker = dict(color = "red", size = 15, symbol = "x")
            ))
        st.plotly_chart(fig)

def display_feature_importance() :
    st.title("Importance globale des variables")
    label = "Nombre de variables à afficher :"
    max_display = st.slider(label, min_value = 5, max_value = 15, value = 10)
    data = {"max_display" : max_display}
    response = requests.post(API_URL + "importance", json = data)
    data = response.json()
    content = {"Variable": data["features"], "Importance": data["importances"]}
    df_importance = pd.DataFrame(content)
    fig = px.bar(df_importance, x="Importance", y="Variable",
                 title = "Importance globale des variables", orientation = "h")
    fig.update_layout(xaxis_title = "Importance",
                      yaxis_categoryorder = "total ascending")
    st.plotly_chart(fig)

def display_score(data) :
    if data["pred_binary"] == 0 :
        title = "La demande d'emprunt est acceptée"
    else :
        title = "La demande d'emprunt est refusée"
    gauge = {
        "axis" : {"range" : [0, 1]},
        "bar" : {"color" : "red" if data["pred_binary"] == 1 else "green"},
        "steps": [{"range": [0, data["acceptance"]], "color" : "lightgray"},
                  {'range': [data["acceptance"], 1], "color": "lightgray"}],
        "threshold" : {"line" : {"color" : "black", "width" : 4},
                       "thickness" : 0.75,
                       "value" : data["acceptance"]}
    }
    fig = go.Figure(go.Indicator(mode = "gauge+number",
                                 value = 1 - data["pred_proba"],
                                 title = {"text": title},
                                 gauge = gauge))
    st.plotly_chart(fig)

def display_waterfall(data) :
    features = data["top_features"]
    shap_values = np.around(-np.array(data["top_shap_values"]), decimals = 3)
    fig = go.Figure()
    colors = ["blue" if value > 0 else "red" for value in shap_values]
    fig.add_trace(go.Bar(x = shap_values, y = features,
                         text = [f"{value:.3f}" for value in shap_values],
                         textposition = "inside", orientation = "h",
                         marker = dict(color = colors)))
    fig.update_layout(title = "Importance locale des variables",
                      xaxis_title = "Valeur SHAP")
    st.plotly_chart(fig)

def predict_score(customer) :
    st.title("Demande d'emprunt")
    if st.button("Lancer la simulation") :
        data = {"selected_index": customer, "shap_max_display" : 10}
        response = requests.post(API_URL + "predict", json = data)
        data = response.json()
        display_score(data)
        display_waterfall(data)

if __name__ == "__main__":
    main()