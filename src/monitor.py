import pandas as pd
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently import Report
from sklearn.preprocessing import LabelEncoder
import os

DATA_PATH = "data/Churn_Modelling.csv"

FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    for col in ["Geography", "Gender"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df[FEATURES]


def run_report():
    df = load_data()

    reference = df.iloc[:7000]
    production = df.iloc[7000:].copy()
    production["Age"] = production["Age"] * 1.1
    production["Balance"] = production["Balance"] * 1.2

    os.makedirs("reports", exist_ok=True)

    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
    ])

    snapshot = report.run(current_data=production, reference_data=reference)
    snapshot.save_html("reports/drift_report.html")
    print("Rapport généré : reports/drift_report.html")


if __name__ == "__main__":
    run_report()