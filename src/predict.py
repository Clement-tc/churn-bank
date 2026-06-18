import joblib
import pandas as pd

MODEL_PATH = "model/xgb_churn.pkl"
ENCODER_PATH = "model/encoders.pkl"

FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]


def load_model():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders


def predict_proba(input_dict: dict) -> float:
    model, encoders = load_model()
    df = pd.DataFrame([input_dict])

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    df = df[FEATURES]
    proba = model.predict_proba(df)[0][1]
    return float(proba)
