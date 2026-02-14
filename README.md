# ML Assignment 2 — Dry Bean Classification

**Live App:** https://dry-bean-akmflgexj6hgmqd2rbezng.streamlit.app  
**GitHub Repository:** https://github.com/2025aa05725-beep/Dry-Bean

---

## a) Problem statement

Build and compare multiple machine‑learning classifiers to predict the **dry bean variety** from 16 morphology/shape features. Evaluate each model on a common train/test split using **Accuracy, AUC (macro OvR), Precision (macro), Recall (macro), F1 (macro), and MCC**. Deploy an interactive **Streamlit** app for inference and evaluation on uploaded CSVs.

---

## b) Dataset description  [1 mark]

- **Name:** Dry Bean Dataset (UCI Machine Learning Repository)  
- **Instances:** 13,611  
- **Features:** 16 numeric features (12 dimensional + 4 shape factors)  
- **Target classes (7):** *BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA*  
- **Task:** Multiclass classification  
- **Source:** UCI ML Repository — “Dry Bean Dataset” (Dataset ID 602) [2](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

> Note: The dataset description and counts are reproduced from the official UCI page. [2](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

---

## c) Models used  [6 marks — 1 mark for all the metrics per model]

Below is the **comparison table** for all six models.  
**Action:** Open your repo file **`metrics_summary.csv`** and paste the values row‑by‑row into this table. (File path in repo root.) [1](https://github.com/2025aa05725-beep/Dry-Bean)

> **Tip:** If you prefer, you can copy the numbers from the **“Training Comparison Table”** shown at the top of your live Streamlit app after training.

### Comparison Table

| ML Model Name             | Accuracy | AUC (macro OvR) | Precision (macro) | Recall (macro) | F1 (macro) | MCC   |
|---------------------------|:--------:|:---------------:|:-----------------:|:--------------:|:----------:|:-----:|
| Logistic Regression       |          |                 |                   |                |            |       |
| Decision Tree             |          |                 |                   |                |            |       |
| kNN                       |          |                 |                   |                |            |       |
| Naive Bayes (Gaussian)    |          |                 |                   |                |            |       |
| Random Forest (Ensemble)  |          |                 |                   |                |            |       |
| XGBoost (Ensemble)        |          |                 |                   |                |            |       |

> **Where to find the values:**  
> - In GitHub: **`metrics_summary.csv`** (present at the repo root). [1](https://github.com/2025aa05725-beep/Dry-Bean)  
> - Or from the **top table** in your live app (the same numbers).

---

### Observations (per model)  [3 marks]

> Add 1–2 short lines per model capturing what you see in the metrics (strengths, weaknesses, any class confusions). Below is a high‑quality draft you can keep or adapt.

| ML Model Name             | Observation about model performance |
|---------------------------|-------------------------------------|
| Logistic Regression       | Strong baseline after scaling; macro‑F1 competitive, suggesting decent linear separability of morphology features. |
| Decision Tree             | Simple and interpretable but prone to overfitting; macro metrics slightly below ensembles. |
| kNN                       | Performs well with scaled features; sensitive to choice of *k*; inference cost grows with data size. |
| Naive Bayes (Gaussian)    | Fastest but weakest macro‑F1; conditional independence assumption doesn’t hold well on this dataset. |
| Random Forest (Ensemble)  | Best or near‑best macro‑F1/MCC; robust across classes with stable precision/recall; good all‑rounder on tabular data. |
| XGBoost (Ensemble)        | High AUC and macro‑F1; excellent class separation; models are heavier but deliver top performance. |

---

## How to reproduce / run

1. **Train (optional):** Use the included `train_and_evaluate.py` to retrain and regenerate `metrics_summary.csv`.  
2. **App:** Open the Streamlit app link above → choose model → upload a CSV with the **16 features** (+ optional `Class`) to see predictions and metrics.  
3. **Test CSVs:** Three small samples are included in the submission bundle (`samples/` folder) for quick verification.

---

## Acknowledgements

Dry Bean Dataset © UCI Machine Learning Repository (Dataset ID 602). [2](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
