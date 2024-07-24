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
    """
    Fonction principale du script.
    Charge le dataframe, sélectionne un client, affiche des caractéristiques, 
    compare des variables avec possibilité de filtrer, affiche l'importance
    des variables et prédit le score du client.
    """
    df, features = load_dataframe()
    customer = select_customer(df)
    display_feature(df, features, customer)
    compare_features(df, features, customer)
    display_feature_importance()
    predict_score(customer)

def load_dataframe() :
    """
    Charge les données depuis un fichier CSV et affiche le dataframe.
    Returns :
        pd.DataFrame : Le dataframe contenant les données.
        list : La liste des noms des colonnes du dataframe.
    """
    df = pd.read_csv(CSV_FILE_NAME)
    features = list(df.columns)
    df = df[features]
    st.title("Fichier client")
    st.write("Le tableau ci-dessous affiche les données des clients.")
    st.write(f"Il contient {df.shape[0]} lignes correspondant aux clients et \
             {df.shape[1]} colonnes correspondant à leurs caractéristiques.")
    st.write(df)
    return df, features

def select_customer(df) :
    """
    Permet à l'utilisateur de sélectionner un client à partir d'un index
    et affiche les informations du client sélectionné.
    Args :
        df (pd.DataFrame) : Le dataframe contenant les données des clients.
    Returns :
        int : L'index du client sélectionné.
    """
    st.title("Sélection du client")
    label = "Utilisez le sélecteur numérique ci-dessous pour \
        choisir l'index d'un client, ce qui permettra d'afficher \
        ses caractéristiques dans un tableau :"
    customer = st.number_input(label, min_value = 0, max_value = len(df) - 1)
    st.write(f"Tableau des caractéristiques du client à l'index {customer} :")
    st.write(df.iloc[customer])
    return customer

def apply_filter(df, customer, key) :
    """
    Applique un filtre basé sur les caractéristiques sélectionnées
    par l'utilisateur et retourne un dataframe filtré.
    Args :
        df (pd.DataFrame) : Le dataframe contenant les données des clients.
        customer (int) : L'index du client sélectionné.
        key (str) : La clé pour l'élément de filtre dans Streamlit.
    Returns:
        pd.DataFrame : Le dataframe filtré.
        str : La chaîne ajoutée au titre en fonction du filtre appliqué.
    """
    label = "Utilisez la liste déroulante ci-dessous pour choisir une \
        catégorie suivant laquelle filtrer les données à afficher :"
    prefix = st.selectbox(label, FILTERS, key = key)
    if prefix == "(aucun filtre)" :
        return df, None
    features = []
    customer_feature = None
    for feature in df.columns :
        if feature.startswith(prefix) :
            features.append(feature)
            if df.iloc[customer][feature] == 1 :
                customer_feature = feature
    label = "Utilisez la liste déroulante ci-dessous pour sélectionner \
         la valeur du filtre "
    if customer_feature == None :
        label += ":"
        default = 0
    else :
        label += "(le client sélectionné est dans la "
        label += f"catégorie {customer_feature}) :"
        default = features.index(customer_feature)
    key += "value"
    filter_value = st.selectbox(label, features, index = default, key = key)
    return df[df[filter_value] == 1], filter_value

def display_feature(df, features, customer):
    """
    Affiche la distribution d'une variable sélectionnée dans un histogramme.
    Args :
        df (pd.DataFrame) : Le dataframe contenant les données des clients.
        features (list) : La liste des noms des colonnes du dataframe.
        customer (int) : L'index du client sélectionné.
    """
    st.title("Distribution des variables")
    # Sélection de la variable
    st.write("Sélectionnez une variable (caractéristique client) pour \
             afficher sa distribution dans un histogramme.")
    label = "Utilisez la liste déroulante ci-dessous pour sélectionner la \
        caractéristique à afficher :"
    default = features.index("AMT_CREDIT")
    feature = st.selectbox(label, features, index = default)
    # Affichage du client
    st.write("")
    label = "Cochez cette case pour mettre en évidence sur \
        l'histogramme la valeur du client sélectionné."
    highlight = st.checkbox(label)
    # Application d'un filtre
    st.write("")
    df_filter, filter_value = apply_filter(df, customer, "univ")
    # Affichage de l'histogramme
    if feature :
        title = f"Distribution de {feature}"
        if filter_value != None :
            title += f" (filtré avec {filter_value})"
        fig = px.histogram(df_filter, x = feature, title = title)
        fig.update_layout(bargap = 0.02)
        if highlight :
            value = df.iloc[customer][feature]
            fig.add_vline(x = value, line_width = 3, line_color = "red")
        st.plotly_chart(fig, use_container_width = True)
        # Description pour les utilisateurs de lecteurs d'écran
        label = "Cochez cette case pour afficher une description \
        textuelle du graphique."
        show_description = st.checkbox(label, key = "distribution")
        if show_description :
            st.write(f"L'histogramme ci-dessus montre la distribution de \
                     la variable {feature}.")
            if filter_value != None :
                st.write(f"Seuls les clients de la catégorie \
                         {filter_value} sont représentés.")
            if highlight :
                st.write(f"Une ligne rouge indique la valeur de la variable \
                         pour le client sélectionné (index {customer}).")

def compare_features(df, features, customer) :
    """
    Compare deux variables sélectionnées avec un graphique de dispersion.
    Args :
        df (pd.DataFrame) : Le dataframe contenant les données des clients.
        features (list) : La liste des noms des colonnes du dataframe.
        customer (int) : L'index du client sélectionné.
    """
    st.title("Comparaison des variables")
    st.write("Sélectionnez deux variables pour comparer leurs distributions \
             à l'aide d'un graphique de dispersion.")
    # Sélection de la variable en abscisse
    label_x = "Utilisez la liste déroulante ci-dessous pour sélectionner \
        la variable en abscisse :"
    default_x = features.index("AMT_CREDIT")
    feature_x = st.selectbox(label_x, features, index = default_x)
    # Sélection de la variable en ordonnée
    label_y = "Utilisez la liste déroulante ci-dessous pour sélectionner \
        la variable en ordonnée :"
    default_y = features.index("AMT_ANNUITY")
    feature_y = st.selectbox(label_y, features, index = default_y)
    # Affichage du client
    st.write("")
    label = "Cochez cette case pour mettre en évidence sur \
        le graphique la valeur du client sélectionné."
    highlight = st.checkbox(label, key = "biv")
    # Application d'un filtre
    st.write("")
    df_filter, filter_value = apply_filter(df, customer, "biv_filter")    
    # Affichage du graphique de dispersion
    if feature_x and feature_y :
        title = f"Comparaison de {feature_x} et {feature_y}"
        if filter_value != None :
            title += f" (filtré avec {filter_value})"
        fig = px.scatter(df_filter, x = feature_x, y = feature_y, title = title)
        if highlight :
            value_x = df.iloc[customer][feature_x]
            value_y = df.iloc[customer][feature_y]
            fig.add_trace(go.Scatter(
                x = [value_x], y = [value_y],
                mode = "markers", name = "Client", showlegend = False,
                marker = dict(color = "red", size = 15, symbol = "x")
            ))
        st.plotly_chart(fig)
        # Description pour les utilisateurs de lecteurs d'écran
        label = "Cochez cette case pour afficher une description \
        textuelle du graphique."
        show_description = st.checkbox(label, key = "comparison")
        if show_description :
            st.write(f"Le nuage de points ci-dessus montre la comparaison \
                     entre les variables {feature_x} et {feature_y}.")
            if filter_value != None :
                st.write(f"Seuls les clients de la catégorie \
                         {filter_value} sont représentés.")
            if highlight :
                st.write(f"Une croix rouge indique la position du client \
                         sélectionné (index {customer}).")

def display_feature_importance() :
    """
    Affiche l'importance globale des variables à partir d'une requête API.
    """
    st.title("Importance globale des variables")
    st.write("Affichez l'histogramme de l'importance globale des variables \
             ayant le plus d'impact sur l'apprentissage du modèle.")
    # Sélection du nombre de variables à afficher
    label = "Utilisez le slider ci-dessous pour indiquer le nombre \
        de variables à afficher dans l'histogramme (entre 5 et 15) :"
    max_display = st.slider(label, min_value = 5, max_value = 15, value = 10)
    # Requête à l'API pour obtenir l'importance des features
    data = {"max_display": max_display}
    response = requests.post(API_URL + "importance", json = data)
    data = response.json()
    content = {"Variable": data["features"], "Importance": data["importances"]}
    df_importance = pd.DataFrame(content)
    # Affichage du graphique
    fig = px.bar(df_importance, x = "Importance", y = "Variable",
                 title = "Importance globale des variables", orientation = "h")
    fig.update_layout(xaxis_title = "Importance",
                      yaxis_categoryorder = "total ascending")
    st.plotly_chart(fig, use_container_width = True)
    # Description pour les utilisateurs de lecteurs d'écran
    # Description textuelle pour les utilisateurs de lecteurs d'écran
    label = "Cochez cette case pour afficher une description \
        textuelle du graphique."
    show_description = st.checkbox(label, key = "importance")
    if show_description :
        st.write(f"L'histogramme ci-dessus montre l'importance des \
                {max_display} variables ayant eu le plus d'impact \
                sur l'apprentissage du modèle.")
        st.write("Les variables sont affichées sur l'axe vertical et leur \
                importance est représentée sur l'axe horizontal.")
        st.write("Par ordre décroissant d'importance, ces variables sont :")
        names = ", ".join(df_importance["Variable"])
        st.write(names)

def display_score(data) :
    """
    Affiche le score prédit et le statut d'acceptation sous forme
    d'un graphique de jauge.
    Args :
        data (dict) : Données sur les prédictions et le seuil d'acceptation.
    """
    # Affichage de la jauge
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
    # Description textuelle pour les utilisateurs de lecteurs d'écrans
    score = 1 - data["pred_proba"]
    if score < data["acceptance"] :
        comparison = "inférieur"
        status = "refusée"
    else :
        comparison = "supérieur"
        status = "acceptée"
    st.write(f"Le client sélectionné a obtenu un score de {score:.2f} ce \
             qui est {comparison} au seuil de {data['acceptance']:.2f} : \
             sa demande d'emprunt est donc {status}.")

def display_waterfall(data):
    """
    Affiche un graphique en cascade des valeurs SHAP pour les variables
    les plus importantes et fournit une description textuelle détaillée.
    Args :
        data (dict) : Données sur les variables et les valeurs SHAP.
    """
    # Affichage du diagramme waterfall
    features = data["top_features"]
    shap_values = np.around(-np.array(data["top_shap_values"]), decimals = 3)
    fig = go.Figure()
    colors = ["blue" if value > 0 else "red" for value in shap_values]
    fig.add_trace(go.Bar(x = shap_values, y = features,
                         text = [f"{value:.3f}" for value in shap_values],
                         textposition = "inside", orientation = "h",
                         marker = dict(color=colors)))
    fig.update_layout(title = "Importance locale des variables",
                      xaxis_title = "Valeur SHAP")
    st.plotly_chart(fig, use_container_width = True)
    # Description textuelle pour les utilisateurs de lecteurs d'écran
    label = "Cochez cette case pour afficher une description \
        textuelle du graphique."
    show_description = st.checkbox(label, key = "shap")
    if show_description :
        st.write("Le graphique en cascade ci-dessus montre les valeurs \
                SHAP des 10 variables les plus importantes pour la \
                prédiction du score du client sélectionné.")
        st.write("Les valeurs positives en bleu contribuent à l'acceptation \
                de la demande tandis que les valeurs négatives en rouge \
                contribuent au refus de la demande.")
        st.write("Voici les 10 variables les plus importantes et leurs \
                valeurs SHAP correspondantes :")
        for feature, value in zip(features, shap_values):
            st.write(f"- {feature} : {value:.3f}")

def predict_score(customer) :
    """
    Effectue une prédiction de score d'emprunt et affiche les résultats.
    Args :
        customer (int) : L'index du client sélectionné.
    """
    st.title("Demande d'emprunt")
    data = {"selected_index": customer, "shap_max_display" : 10}
    response = requests.post(API_URL + "predict", json = data)
    data = response.json()
    display_score(data)
    display_waterfall(data)

if __name__ == "__main__":
    main()