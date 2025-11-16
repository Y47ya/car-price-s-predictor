import pandas as pd
import streamlit as st
import joblib
import requests
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

df_encoded_ref = pd.read_csv('../Data/voitures_preprocessed.csv', encoding='latin')
marque_modele_map = dict(zip(df_encoded_ref['Marque_Modele'], df_encoded_ref['Marque_Modele_Encoded']))

st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f0f4f8;
            color: #333333;
        }
        .title {
            font-size: 50px !important;
            font-weight: 900;
            color: #2c3e50;
            text-align: center;
            padding: 20px 0;
            letter-spacing: 1px;
        }
        label {
            font-weight: 600 !important;
            font-size: 15px !important;
            color: #2c3e50 !important;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            transition: 0.3s ease-in-out;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            background-color: #2980b9;
            transform: scale(1.05);
        }
        .custom-header {
            font-size:22px;
            font-weight:600;
            color:#333333;
            margin-top: 20px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">Estimation du prix d\'une voiture au Maroc 🚗💰</p>', unsafe_allow_html=True)
st.markdown('<p class="custom-header">Veuillez entrer les caractéristiques de votre voiture :</p>',
            unsafe_allow_html=True)

Kilometrage = st.number_input("🔽 Kilométrage (km)", min_value=0, step=1000)
Nombre_de_portes = st.selectbox("🚪 Nombre de portes", [2, 3, 4, 5, 6], index=0)
Premiere_main = st.radio("🆕 Première main", ["Oui", "Non"])
Puissance_fiscale = st.number_input("🐎 Puissance fiscale (CV)", min_value=1, step=1)
Annee_de_fabriquation = st.number_input("📆 Année du modèle", min_value=1990, max_value=2025, step=1)

Carburant = st.selectbox("⛽ Type de carburant", ["Diesel", "Electrique", "Essence", "Hybride", "LPG"], index=0)
Boite_vitesse = st.radio("⚙️ Boîte à vitesse", ["Automatique", "Manuelle"])
Origine = st.selectbox("🌍 Origine du véhicule", ["Importée neuve", "Pas encore dédouanée", "WW au Maroc", "Dédouanée"],
                       index=0)

Marque = st.text_input("🏷️ Marque du véhicule").strip().lower()
Modele = st.text_input("📛 Modèle du véhicule").strip().lower()


def validate_inputs():
    champs = [Kilometrage, Nombre_de_portes, Premiere_main, Puissance_fiscale, Annee_de_fabriquation, Carburant,
              Boite_vitesse, Origine, Marque, Modele]

    if not all(champs):
        st.error("❗Tous les champs doivent être remplis.")
        return False

    return True


def prepare_input_data():
    return {
        "Kilometrage": f"{Kilometrage}",
        "Nombre_de_portes": Nombre_de_portes,
        "Premiere_main": Premiere_main,
        "Puissance_fiscale": Puissance_fiscale,
        "Carburant": Carburant,
        "BoiteaV": Boite_vitesse,
        "Origine": Origine,
        "Marque": Marque,
        "Modele": Modele,
        "Annee": Annee_de_fabriquation,

    }


if st.button('🔍 Prédire le prix'):
    if validate_inputs():
        api_url = "http://127.0.0.1:8001/predict"
        payload = prepare_input_data()
        try:
            with st.spinner("🔄 Prédiction en cours..."):
                response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                price = response.json()['predicted_price']
                st.success(f"💰 Prix estimé : {price:,.2f} MAD")
            else:
                st.error(f"🚨 Erreur de l'API : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"🔌 Erreur de connexion à l’API : {e}")

