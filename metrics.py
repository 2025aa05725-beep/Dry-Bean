from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
)

def compute_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray],
    labels: List[Any],
    average_for_multiclass: str = "macro",
) -> Dict[str, float]:
    m = {}
    m["accuracy"]  = accuracy_score(y_true, y_pred)
    m["precision"] = precision_score(y_true, y_pred, average=average_for_multiclass, zero_division=0)
    m["recall"]    = recall_score(y_true, y_pred, average=average_for_multiclass, zero_division=0)
    m["f1"]        = f1_score(y_true, y_pred, average=average_for_multiclass, zero_division=0)
    m["mcc"]       = matthews_corrcoef(y_true, y_pred)

    if y_proba is not None:
        try:
            y_true = np.array(y_true)
            labels = np.array(labels)
            y_true_bin = np.zeros((len(y_true), len(labels)))
            idx = {lab: i for i, lab in enumerate(labels)}
            for i, lab in enumerate(y_true):
                y_true_bin[i, idx[lab]] = 1.0
            m["auc"] = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average=average_for_multiclass)
        except Exception:
            m["auc"] = float("nan")
    else:
        m["auc"] = float("nan")
    return m

def get_confusion_and_report(y_true, y_pred, labels: List[Any]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
    return {"confusion_matrix": cm, "classification_report": report}
