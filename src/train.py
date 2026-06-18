import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

DATA_PATH = "data/Churn_Modelling.csv"
MODEL_PATH = "model/xgb_churn.pkl"
ENCODER_PATH = "model/encoders.pkl"

FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
TARGET = "Exited"


def load_and_prepare(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    encoders = {}
    for col in ["Geography", "Gender"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[FEATURES]
    y = df[TARGET]
    return X, y, encoders


def train():
    mlflow.set_experiment("churn-bank")

    X, y, encoders = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "random_state": 42,
        "eval_metric": "auc",
    }

    with mlflow.start_run():
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params(params)
        mlflow.log_metric("auc", auc)
        mlflow.xgboost.log_model(model, "model")

        print(f"AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")

    return model, encoders, X_test, y_test


if __name__ == "__main__":
    train()
