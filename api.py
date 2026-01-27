# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="NFL Prediction API")

# Chargement du modèle (utilise ton fichier .pkl ou .joblib)
MODEL_PATH = Path("models/rf_points__Pts.pkl")
model = joblib.load(MODEL_PATH)

class PredictionRequest(BaseModel):
    # Liste des features attendues par ton modèle (ex: Yards, FirstDowns, etc.)
    features: dict 


@app.get("/")
async def root():
    return {"status": "API is running"}


@app.get("/predict")
async def predict_info():
    return {"message": "Cet endpoint nécessite une requête POST avec les statistiques de l'équipe."}


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        df = pd.DataFrame([request.features])
        prediction = model.predict(df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))