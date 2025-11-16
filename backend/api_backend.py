from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# ------------------------------
# FastAPI instance
# ------------------------------
app = FastAPI(title="Car Price Prediction API")

# ------------------------------
# Load trained model
# ------------------------------
predictor = joblib.load(
    "C:/Users/yahya/Desktop/Projects/PFM-ML-S3-P1/Models/price_predictor.pkl"
)

# ------------------------------
# Load dataset for Marque_Modele mapping
# ------------------------------
df = pd.read_csv(
    "C:/Users/yahya/Desktop/Projects/PFM-ML/Data/voitures_preprocessed.csv",
    encoding="latin1"
)
marque_modele_map = dict(zip(df['Marque_Modele'], df['Marque_Modele_Encoded']))

class VehicleData(BaseModel):
    Kilometrage: int
    Nombre_de_portes: int
    Premiere_main: str       # "Oui" or "Non"
    Puissance_fiscale: int
    Carburant: str
    BoiteaV: str
    Origine: str
    Marque: str
    Modele: str
    Annee: int

# ------------------------------
# Transform input data for model
# ------------------------------
def transform_data(input_dict):
    car_infos = pd.DataFrame([input_dict])

    # Premiere_main: Oui -> 1, Non -> 0
    car_infos['Premiere_main'] = car_infos['Premiere_main'].map({'Oui': 1, 'Non': 0}).fillna(0)

    categories = [
        ['Diesel', 'Electrique', 'Essence', 'Hybride', 'LPG'],
        ['Automatique', 'Manuelle'],
        ['Dédouanée', 'Importée_neuve', 'Pas_encore_dédouanée', 'WW_au_Maroc']
    ]

    encoder = OneHotEncoder(categories=categories, dtype=int, sparse_output=False, handle_unknown='ignore')

    encoded_array = encoder.fit_transform(car_infos[['Carburant', 'BoiteaV', 'Origine']])

    encoded_cols = encoder.get_feature_names_out(['Carburant', 'BoiteaV', 'Origine'])
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

    car_infos = car_infos.drop(['Carburant', 'BoiteaV', 'Origine'], axis=1)
    car_infos = pd.concat([car_infos, encoded_df], axis=1)

    # Marque_Modèle encodée
    car_infos['Marque_Modele'] = (car_infos['Marque'].str.strip() + '_' + car_infos['Modele'].str.strip()).str.replace(
        ' ', '_')

    car_infos['Marque_Modele_Encoded'] = car_infos['Marque_Modele'].map(marque_modele_map)

    # Calculer l'age
    car_infos['Age'] = datetime.now().year - int(car_infos['Annee'].iloc[0])






    # Drop unused columns
    car_infos = car_infos.drop(["Marque", "Modele", "Marque_Modele", "Annee"], axis=1)

    car_infos.columns = ['Kilometrage', 'Nombre_de_portes', 'Premiere_main', 'Puissance_fiscale', 'Carburant_Diesel',
                         'Carburant_Electrique', 'Carburant_Essence', 'Carburant_Hybride', 'Carburant_LPG',
                         'BoiteaV_Automatique', 'BoiteaV_Manuelle', 'Origine_Dedouanee', 'Origine_Importee_neuve',
                         'Origine_Pas_encore_dedouanee', 'Origine_WW_au_Maroc', 'Marque_Modele_Encoded', 'Age']

    return np.array(car_infos.values.flatten().tolist()).reshape(1, -1)

# ------------------------------
# Prediction route
# ------------------------------
@app.post("/predict")
def predict_price(data: VehicleData):
    try:
        transformed = transform_data(data.dict())
        prediction = predictor.predict(transformed)
        return {"predicted_price": round(float(prediction[0]), 2)}
    except Exception as e:
        # Always return JSON even on error
        return {"error": str(e)}

# ------------------------------
# Optional test
# ------------------------------
if __name__ == "__main__":
    test_car = {
        "Kilometrage": 1300,
        "Nombre_de_portes": 5,
        "Premiere_main": "Non",
        "Puissance_fiscale": 6,
        "Carburant": "Diesel",
        "BoiteaV": "Automatique",
        "Origine": "WW_au_Maroc",
        "Marque": "Toyota",
        "Modele": "Corolla",
        "Annee": 2018
    }
    data = transform_data(test_car)
    print("Transformed array shape:", data.shape)
    print("Prediction:", predictor.predict(data)[0])
