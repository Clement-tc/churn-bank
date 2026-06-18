import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Predictor - Banking", layout="wide")

MODEL_PATH = "model/xgb_churn.pkl"
ENCODER_PATH = "model/encoders.pkl"
DATA_PATH = "data/Churn_Modelling.csv"

FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    return df


def encode(df, encoders):
    df = df.copy()
    for col, le in encoders.items():
        df[col] = le.transform(df[col])
    return df


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Exploration", "🤖 Prédiction", "📈 Performance"])

model, encoders = load_model()
df_raw = load_data()

# ── Page 1 : EDA ──────────────────────────────────────────────────────────────
if page == "📊 Exploration":
    st.title("Exploration des données — Churn Bancaire")

    col1, col2, col3 = st.columns(3)
    col1.metric("Clients", f"{len(df_raw):,}")
    col2.metric("Taux de churn", f"{df_raw['Exited'].mean()*100:.1f}%")
    col3.metric("Features", len(FEATURES))

    st.subheader("Distribution du churn")
    fig = px.pie(
        df_raw, names=df_raw["Exited"].map({0: "Fidèle", 1: "Churné"}),
        color_discrete_sequence=["#2ecc71", "#e74c3c"],
        hole=0.4,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn par géographie et genre")
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.histogram(
            df_raw, x="Geography", color=df_raw["Exited"].map({0: "Fidèle", 1: "Churné"}),
            barmode="group", color_discrete_sequence=["#2ecc71", "#e74c3c"],
            labels={"color": "Statut"},
        )
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.histogram(
            df_raw, x="Gender", color=df_raw["Exited"].map({0: "Fidèle", 1: "Churné"}),
            barmode="group", color_discrete_sequence=["#2ecc71", "#e74c3c"],
            labels={"color": "Statut"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Âge vs Solde — coloré par churn")
    fig4 = px.scatter(
        df_raw, x="Age", y="Balance",
        color=df_raw["Exited"].map({0: "Fidèle", 1: "Churné"}),
        color_discrete_sequence=["#2ecc71", "#e74c3c"],
        opacity=0.5, labels={"color": "Statut"},
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Matrice de corrélation")
    df_enc = encode(df_raw, encoders)
    corr = df_enc.corr()
    fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig5, use_container_width=True)


# ── Page 2 : Prédiction ───────────────────────────────────────────────────────
elif page == "🤖 Prédiction":
    st.title("Prédiction de churn — Client individuel")
    st.markdown("Remplis le profil d'un client pour obtenir sa probabilité de churn.")

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Âge", 18, 92, 40)
        tenure = st.slider("Ancienneté (années)", 0, 10, 5)
        balance = st.number_input("Solde (€)", 0.0, 250000.0, 50000.0, step=1000.0)
        num_products = st.selectbox("Nombre de produits", [1, 2, 3, 4])
    with col2:
        geography = st.selectbox("Pays", ["France", "Germany", "Spain"])
        gender = st.selectbox("Genre", ["Male", "Female"])
        has_cr_card = st.selectbox("Carte de crédit ?", [1, 0], format_func=lambda x: "Oui" if x else "Non")
        is_active = st.selectbox("Membre actif ?", [1, 0], format_func=lambda x: "Oui" if x else "Non")
        salary = st.number_input("Salaire estimé (€)", 0.0, 200000.0, 60000.0, step=1000.0)

    if st.button("Prédire le churn", type="primary"):
        input_dict = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary,
        }

        df_input = pd.DataFrame([input_dict])
        df_enc = encode(df_input, encoders)[FEATURES]
        proba = model.predict_proba(df_enc)[0][1]

        st.divider()
        col_res, col_gauge = st.columns(2)

        with col_res:
            if proba >= 0.5:
                st.error(f"⚠️ Risque de churn élevé : **{proba*100:.1f}%**")
            else:
                st.success(f"✅ Risque de churn faible : **{proba*100:.1f}%**")

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if proba >= 0.5 else "#2ecc71"},
                    "steps": [
                        {"range": [0, 40], "color": "#d5f5e3"},
                        {"range": [40, 60], "color": "#fdebd0"},
                        {"range": [60, 100], "color": "#fadbd8"},
                    ],
                },
                title={"text": "Probabilité de churn (%)"},
                number={"suffix": "%"},
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Pourquoi ce score ? (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_enc)
        fig_shap, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=df_enc.values[0],
                feature_names=FEATURES,
            ),
            show=False,
        )
        st.pyplot(fig_shap, use_container_width=True)


# ── Page 3 : Performance ──────────────────────────────────────────────────────
elif page == "📈 Performance":
    st.title("Performance du modèle")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, confusion_matrix, roc_curve,
        precision_recall_curve, classification_report
    )

    df_enc = encode(df_raw, encoders)
    X = df_enc[FEATURES]
    y = df_enc["Exited"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", f"{auc:.4f}")
    report = classification_report(y_test, y_pred, output_dict=True)
    col2.metric("Précision (churn)", f"{report['1']['precision']:.2%}")
    col3.metric("Recall (churn)", f"{report['1']['recall']:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {auc:.3f}", fill="tozeroy"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Aléatoire"))
        fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Prédit Fidèle", "Prédit Churné"],
            y=["Réel Fidèle", "Réel Churné"],
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Importance des features")
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig_imp = px.bar(importance, orientation="h", labels={"value": "Importance", "index": "Feature"})
    st.plotly_chart(fig_imp, use_container_width=True)
