import joblib
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from utils.metrics import compute_metrics, get_confusion_and_report

st.set_page_config(page_title="Dry Bean Classifier — ML Assignment 2", layout="wide")
st.title("Dry Bean Classification — ML Assignment 2")
st.caption("Upload a test CSV. If it contains 'Class', the app will compute metrics.")

available_models = {
    "Logistic Regression": "model/Logistic_Regression.joblib",
    "Decision Tree": "model/Decision_Tree.joblib",
    "kNN": "model/kNN.joblib",
    "Naive Bayes (Gaussian)": "model/Naive_Bayes_Gaussian.joblib",
    "Random Forest": "model/Random_Forest.joblib",
    "XGBoost": "model/XGBoost.joblib",
}

model_name = st.sidebar.selectbox("Choose a model", list(available_models.keys()))
model_path = Path(available_models[model_name])
if not model_path.exists():
    st.error(f"Model not found: {model_path}. Please run training and redeploy.")
    st.stop()

pipe = joblib.load(model_path)
uploaded = st.sidebar.file_uploader("Upload Test CSV (features + optional 'Class')", type=["csv"])

def plot_cm(cm, labels, title):
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

summary_path = Path("artifacts/metrics_summary.csv")
if summary_path.exists():
    with st.expander("📊 Training Comparison Table"):
        st.dataframe(pd.read_csv(summary_path), use_container_width=True)

st.markdown("### Inference & Evaluation")

if uploaded is None:
    st.info("Upload a CSV with the 16 feature columns. Include 'Class' to see evaluation.")
else:
    df_test = pd.read_csv(uploaded)
    st.write("**Preview**"); st.dataframe(df_test.head(), use_container_width=True)

    if "Class" in df_test.columns:
        y_true = df_test["Class"].astype(str)
        X_test = df_test.drop(columns=["Class"])
    else:
        y_true = None
        X_test = df_test

    y_pred = pipe.predict(X_test)
    st.success(f"Predictions generated using **{model_name}**.")
    st.write(pd.DataFrame({"pred": y_pred[:20]}))

    if y_true is not None:
        y_proba = pipe.predict_proba(X_test) if hasattr(pipe[-1], "predict_proba") else None
        labels = sorted(pd.unique(pd.concat([y_true, pd.Series(y_pred)], ignore_index=True)).tolist())
        m = compute_metrics(y_true, y_pred, y_proba, labels=labels, average_for_multiclass="macro")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']:.4f}")
        c2.metric("Macro F1", f"{m['f1']:.4f}")
        c3.metric("MCC", f"{m['mcc']:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Macro Precision", f"{m['precision']:.4f}")
        c5.metric("Macro Recall", f"{m['recall']:.4f}")
        c6.metric("Macro AUC (OvR)", f"{m['auc']:.4f}" if isinstance(m['auc'], float) else "N/A")

        cm_rep = get_confusion_and_report(y_true, y_pred, labels=labels)
        st.markdown("#### Confusion Matrix")
        plot_cm(cm_rep["confusion_matrix"], labels, f"Confusion Matrix — {model_name}")

        st.markdown("#### Classification Report")
        st.code(cm_rep["classification_report"], language="text")
