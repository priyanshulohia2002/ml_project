# Streamlit app - loads the saved models, does not retrain
# Upload test_data.csv from the sidebar

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

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
POS = "ozone"

MODEL_PATHS = {
    "Logistic Regression": "model/logreg.joblib",
    "Decision Tree": "model/dtree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/nb.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

st.set_page_config(page_title="Ozone Day Classification", layout="wide")

st.title("8-Hour Ozone Day Classification")
st.write(
    "Assignment 2 – Machine Learning (M.Tech AIML). "
    "Predict whether a day in the Houston–Galveston area is an ozone day "
    "from 72 weather and upper-air measurements. "
    "Upload the test csv to score the saved models. Nothing is retrained here."
)

st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload test data (csv)", type=["csv"])
model_name = st.sidebar.selectbox("Select model", list(MODEL_PATHS.keys()))

st.sidebar.markdown("---")
st.sidebar.write("**Positive class:** ozone")
st.sidebar.caption(
    "Only 160 of 2534 days are ozone days (~6%), so accuracy alone looks high "
    "even if the model misses most ozone days."
)


@st.cache_resource
def load_models():
    scaler = joblib.load("model/scaler.joblib")
    fitted = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    return scaler, fitted


def scores_for(model, Xs, y):
    pred = model.predict(Xs)
    idx = list(model.classes_).index(POS)
    proba = model.predict_proba(Xs)[:, idx]
    y_bin = (y == POS).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y_bin, proba),
        "Precision": precision_score(y, pred, pos_label=POS, zero_division=0),
        "Recall": recall_score(y, pred, pos_label=POS, zero_division=0),
        "F1": f1_score(y, pred, pos_label=POS, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "pred": pred,
    }


scaler, models = load_models()

if uploaded is None:
    st.info(
        "Upload `test_data.csv` from the sidebar. "
        "It needs the 72 weather columns plus `OzoneDay` (normal / ozone)."
    )
    with st.expander("Expected columns"):
        st.write(FEATURES + [TARGET])
    st.stop()

df = pd.read_csv(uploaded)

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    st.stop()

y = df[TARGET].replace({
    0: "normal", 1: "ozone", "0": "normal", "1": "ozone",
    "Normal": "normal", "Ozone": "ozone",
    "2": "ozone", 2: "ozone",
})
X = df[FEATURES]
Xs = scaler.transform(X)

st.write(
    f"Uploaded rows: **{len(df)}**  |  "
    f"ozone: **{(y == POS).sum()}**  |  "
    f"normal: **{(y == 'normal').sum()}**"
)

all_scores = {}
for name, m in models.items():
    all_scores[name] = scores_for(m, Xs, y)

st.subheader("Comparison on uploaded test data")
table = pd.DataFrame({
    name: {k: v for k, v in s.items() if k != "pred"}
    for name, s in all_scores.items()
}).T
st.dataframe(table.style.format("{:.4f}"), width="stretch")

st.subheader("Selected model: " + model_name)
s = all_scores[model_name]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{s['Accuracy']:.3f}")
c2.metric("AUC", f"{s['AUC']:.3f}")
c3.metric("Precision", f"{s['Precision']:.3f}")
c4.metric("Recall", f"{s['Recall']:.3f}")
c5.metric("F1", f"{s['F1']:.3f}")
c6.metric("MCC", f"{s['MCC']:.3f}")

col_a, col_b = st.columns(2)

with col_a:
    st.write("Confusion matrix")
    cm = confusion_matrix(y, s["pred"], labels=["normal", "ozone"])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["normal", "ozone"],
        yticklabels=["normal", "ozone"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with col_b:
    st.write("Classification report")
    report = classification_report(
        y, s["pred"], labels=["normal", "ozone"],
        target_names=["normal", "ozone"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), width="stretch")

st.caption("Precision / Recall / F1 are for the ozone class.")
