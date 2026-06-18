from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Predictor API")

MODEL_PATH = "model/xgb_churn.pkl"
ENCODER_PATH = "model/encoders.pkl"

FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

# Chargement du modèle au démarrage de l'API
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)


class ClientData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def root():
    return {"message": "Churn Predictor API — fonctionne !"}


@app.post("/predict")
def predict(client: ClientData):
    df = pd.DataFrame([client.model_dump()])

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    proba = model.predict_proba(df[FEATURES])[0][1]
    prediction = "Churné" if proba >= 0.5 else "Fidèle"

    return {
        "churn_probability": round(float(proba), 4),
        "prediction": prediction,
    }
