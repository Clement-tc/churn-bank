import pandas as pd
import joblib
import mlflow
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
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


def objective(trial, X_train, y_train):
    """
    Optuna appelle cette fonction des dizaines de fois.
    A chaque appel il propose des hyperparamètres différents
    et mesure l'AUC en cross-validation.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "scale_pos_weight": 4,
        "random_state": 42,
        "eval_metric": "auc",
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    return scores.mean()


def optimize(n_trials: int = 30):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-bank")

    X, y, encoders = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Lancement Optuna — {n_trials} trials...")

    # Optuna cherche les meilleurs hyperparamètres
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_cv_auc = study.best_value
    print(f"\nMeilleurs params : {best_params}")
    print(f"Meilleur AUC CV  : {best_cv_auc:.4f}")

    # On log le meilleur trial dans MLflow
    with mlflow.start_run(run_name="optuna_best"):
        mlflow.set_tag("algo", "xgboost")
        mlflow.set_tag("variant", "optuna_optimized")
        mlflow.log_params(best_params)
        mlflow.log_metric("auc_cv_mean", best_cv_auc)

        # Entrainement final avec les meilleurs params
        best_params.update({
            "scale_pos_weight": 4,
            "random_state": 42,
            "eval_metric": "auc",
        })
        final_model = XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)

        y_proba = final_model.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("auc_test", auc_test)

        print(f"AUC test final   : {auc_test:.4f}")

    # Sauvegarde si meilleur que le modèle actuel
    current_model = joblib.load(MODEL_PATH)
    current_auc = roc_auc_score(
        y_test, current_model.predict_proba(X_test)[:, 1]
    )

    if auc_test > current_auc:
        joblib.dump(final_model, MODEL_PATH)
        joblib.dump(encoders, ENCODER_PATH)
        print(f"Nouveau modèle sauvegardé (AUC {auc_test:.4f} > {current_auc:.4f})")
    else:
        print(f"Modèle actuel conservé (AUC {current_auc:.4f} >= {auc_test:.4f})")


if __name__ == "__main__":
    optimize(n_trials=30)
