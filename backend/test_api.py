import requests

url = "http://127.0.0.1:8000/predict"

car_json = {
    "Kilometrage": 1300,
    "Nombre_de_portes": 5,
    "Premiere_main": "Non",
    "Puissance_fiscale": 6,
    "Carburant": "Diesel",
    "BoiteaV": "Automatique",
    "Origine": "WW_au_Maroc",
    "Marque": "Dacia",
    "Modele": "Logan",
    "Annee": 2018
}

response = requests.post(url, json=car_json)
print(response.status_code)
print(response.json())
