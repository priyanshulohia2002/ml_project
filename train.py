# Training script for Assignment 2
# Dataset: Ozone Level Detection, 8-hour peak (UCI)
# Run this for the BITS lab screenshot:  python train.py

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
import joblib

DATA_PATH = "data/ozone_8hr.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 72 weather / upper-air features. Date was already removed in the UCI dump I used.
FEATURES = (
    [f"WSR{h}" for h in range(24)]
    + ["WSR_PK", "WSR_AV"]
    + [f"T{h}" for h in range(24)]
    + ["T_PK", "T_AV"]
    + [
        "T85", "RH85", "U85", "V85", "HT85",
        "T70", "RH70", "U70", "V70", "HT70",
        "T50", "RH50", "U50", "V50", "HT50",
        "KI", "TT", "SLP", "SLP_change", "Precp",
    ]
)
TARGET = "OzoneDay"
POS = "ozone"  # rare class (~6%)

df = pd.read_csv(DATA_PATH)
print("dataset shape:", df.shape)
print(df[TARGET].value_counts())
print("n features:", len(FEATURES))
print()

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("train:", X_train.shape, "test:", X_test.shape)
print("ozone days in test:", int((y_test == POS).sum()))
print("majority baseline (always normal) acc:", round((y_test != POS).mean(), 4))

test_out = X_test.copy()
test_out[TARGET] = y_test.values
test_out.to_csv("test_data.csv", index=False)
print("wrote test_data.csv")

# wind speed and geopotential height are on totally different scales
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

# ozone days are only ~6%, so balanced weights where the model supports them
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=8, class_weight="balanced", random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(
        kernel="rbf", class_weight="balanced", probability=True, random_state=42
    ),
}

file_map = {
    "Logistic Regression": "logreg.joblib",
    "Decision Tree": "dtree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "nb.joblib",
    "Support Vector Machine": "svm.joblib",
}


def get_scores(model, Xte, yte):
    pred = model.predict(Xte)
    idx = list(model.classes_).index(POS)
    proba = model.predict_proba(Xte)[:, idx]
    return {
        "Accuracy": accuracy_score(yte, pred),
        "AUC": roc_auc_score((yte == POS).astype(int), proba),
        "Precision": precision_score(yte, pred, pos_label=POS, zero_division=0),
        "Recall": recall_score(yte, pred, pos_label=POS, zero_division=0),
        "F1": f1_score(yte, pred, pos_label=POS, zero_division=0),
        "MCC": matthews_corrcoef(yte, pred),
    }


print("\n--- training ---")
results = {}
for name, clf in models.items():
    print("fitting", name, "...", flush=True)
    clf.fit(X_train_s, y_train)
    scores = get_scores(clf, X_test_s, y_test)
    results[name] = scores
    joblib.dump(clf, os.path.join(MODEL_DIR, file_map[name]))
    print("   ", {k: round(v, 4) for k, v in scores.items()})

print("\n================  TEST SET RESULTS  ================")
header = f"{'Model':<24} {'Acc':>8} {'AUC':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'MCC':>8}"
print(header)
print("-" * len(header))
for name, s in results.items():
    print(
        f"{name:<24} {s['Accuracy']:8.4f} {s['AUC']:8.4f} "
        f"{s['Precision']:8.4f} {s['Recall']:8.4f} {s['F1']:8.4f} {s['MCC']:8.4f}"
    )

best = max(results, key=lambda n: results[n]["MCC"])
print("-" * len(header))
print("best MCC:", best)

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nmodels saved in model/. screenshot this table for BITS lab.")
