import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

DATA_PATH = "data/Churn_Modelling.csv"
MODEL_PATH = "model/xgb_churn.pkl"
ENCODER_PATH = "model/encoders.pkl"

FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
TARGET = "Exited"

EXPERIMENTS = [
    {
        "name": "xgb_baseline",
        "model": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            scale_pos_weight=4, random_state=42, eval_metric="auc"
        ),
        "tags": {"algo": "xgboost", "variant": "baseline"},
    },
    {
        "name": "xgb_deep",
        "model": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=4, random_state=42, eval_metric="auc"
        ),
        "tags": {"algo": "xgboost", "variant": "deep"},
    },
    {
        "name": "xgb_fast",
        "model": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.2,
            scale_pos_weight=4, random_state=42, eval_metric="auc"
        ),
        "tags": {"algo": "xgboost", "variant": "fast"},
    },
    {
        "name": "rf_baseline",
        "model": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "tags": {"algo": "random_forest", "variant": "baseline"},
    },
    {
        "name": "rf_deep",
        "model": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "tags": {"algo": "random_forest", "variant": "deep"},
    },
    {
        "name": "logistic_regression",
        "model": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
        ),
        "tags": {"algo": "logistic_regression", "variant": "baseline"},
    },
]


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
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-bank")

    X, y, encoders = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    for exp in EXPERIMENTS:
        print(f"\n>> Entrainement : {exp['name']}")

        with mlflow.start_run(run_name=exp["name"]):
            model = exp["model"]
            mlflow.set_tags(exp["tags"])

            # Cross-validation 5 folds pour un score robuste
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

            # Entraînement final sur tout le train
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            auc_test = roc_auc_score(y_test, y_proba)

            # Log des paramètres
            mlflow.log_params(model.get_params())

            # Log des métriques
            mlflow.log_metric("auc_cv_mean", cv_scores.mean())
            mlflow.log_metric("auc_cv_std", cv_scores.std())
            mlflow.log_metric("auc_test", auc_test)

            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_metric("precision_churn", report["1"]["precision"])
            mlflow.log_metric("recall_churn", report["1"]["recall"])
            mlflow.log_metric("f1_churn", report["1"]["f1-score"])

            print(f"  AUC test    : {auc_test:.4f}")
            print(f"  AUC CV      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Precision   : {report['1']['precision']:.4f}")
            print(f"  Recall      : {report['1']['recall']:.4f}")

            results.append({
                "name": exp["name"],
                "model": model,
                "auc_test": auc_test,
                "auc_cv": cv_scores.mean(),
            })

    # Sélection du meilleur modèle (AUC test)
    best = max(results, key=lambda r: r["auc_test"])
    print(f"\nMeilleur modele : {best['name']} (AUC = {best['auc_test']:.4f})")

    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    print(f"Modele sauvegarde : {MODEL_PATH}")

    return results


if __name__ == "__main__":
    train()
