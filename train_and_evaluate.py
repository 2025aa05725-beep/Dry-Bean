from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt, seaborn as sns

from utils.metrics import compute_metrics, get_confusion_and_report

RANDOM_STATE = 42
TEST_SIZE = 0.2

def print_block(title: str):
    print("\n" + "="*88)
    print(title)
    print("="*88)

def main():
    data_path = Path("data/Dry_Bean_Dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path.resolve()}")

    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        raise ValueError("Expected target column named 'Class'.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(str)
    class_labels = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    from collections import Counter
    print_block("Test Split — Class Distribution")
    print(Counter(y_test))

    scaler = ColumnTransformer([("num", StandardScaler(), list(X.columns))], remainder="drop")

    pipelines = {
        "Logistic Regression": Pipeline([("scale", scaler),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        "Decision Tree": Pipeline([("identity", "passthrough"),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        "kNN": Pipeline([("scale", scaler),
            ("clf", KNeighborsClassifier(n_neighbors=15))]),
        "Naive Bayes (Gaussian)": Pipeline([("identity", "passthrough"),
            ("clf", GaussianNB())]),
        "Random Forest": Pipeline([("identity", "passthrough"),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))]),
        "XGBoost": Pipeline([("identity", "passthrough"),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.8,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=RANDOM_STATE, n_jobs=-1
            ))]),
    }

    Path("model").mkdir(parents=True, exist_ok=True)
    Path("artifacts/classification_reports").mkdir(parents=True, exist_ok=True)

    le = LabelEncoder().fit(class_labels)   # used only for XGBoost
    rows = []

    for model_name, pipe in pipelines.items():
        print_block(f"Training: {model_name}")

        if model_name == "XGBoost":
            y_train_enc = le.transform(y_train)
            pipe.fit(X_train, y_train_enc)
            y_pred_enc = pipe.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)

            if hasattr(pipe[-1], "predict_proba"):
                y_proba_enc = pipe[-1].predict_proba(X_test)
                enc_order = list(le.classes_)
                idx_map = [enc_order.index(lbl) for lbl in class_labels]
                y_proba = y_proba_enc[:, idx_map]
            else:
                y_proba = None
        else:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe[-1].predict_proba(X_test) if hasattr(pipe[-1], "predict_proba") else None

        m = compute_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba,
                            labels=class_labels, average_for_multiclass="macro")

        print_block(f"RESULTS — {model_name}")
        auc_line = f"{m['auc']:.4f}" if isinstance(m['auc'], float) else "N/A"
        print(f"Accuracy         : {m['accuracy']:.4f}")
        print(f"Macro AUC (OvR)  : {auc_line}")
        print(f"Precision (macro): {m['precision']:.4f}")
        print(f"Recall (macro)   : {m['recall']:.4f}")
        print(f"F1 (macro)       : {m['f1']:.4f}")
        print(f"MCC              : {m['mcc']:.4f}")

        cm_rep = get_confusion_and_report(y_test, y_pred, labels=class_labels)

        safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, Path("model") / f"{safe}.joblib")

        with open(Path("artifacts/classification_reports") / f"{model_name}_report.txt", "w", encoding="utf-8") as f:
            f.write(cm_rep["classification_report"])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_rep["confusion_matrix"], annot=True, fmt="d",
                    xticklabels=class_labels, yticklabels=class_labels, cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix — {model_name}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout()
        fig.savefig(Path("artifacts") / f"{model_name}_confusion_matrix.png")
        plt.close(fig)

        rows.append({
            "ML Model Name": model_name,
            "Accuracy": float(m["accuracy"]),
            "AUC": float(m["auc"]) if isinstance(m["auc"], float) else np.nan,
            "Precision": float(m["precision"]),
            "Recall": float(m["recall"]),
            "F1": float(m["f1"]),
            "MCC": float(m["mcc"]),
        })

    summary = pd.DataFrame(rows).sort_values("F1", ascending=False)
    out_csv = Path("artifacts/metrics_summary.csv")
    summary.to_csv(out_csv, index=False)

    print_block("=== Metrics Summary (sorted by F1, rounded to 4 decimals) ===")
    disp = summary.copy()
    for c in ["Accuracy","AUC","Precision","Recall","F1","MCC"]:
        if c in disp: disp[c] = disp[c].astype(float).round(4)
    print(disp.to_string(index=False))

    print_block(f"Saved summary CSV → {out_csv.resolve()}")
    print("Training complete. Models saved in ./model and artifacts in ./artifacts")

if __name__ == "__main__":
    main()
