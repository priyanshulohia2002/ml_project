# 8-Hour Ozone Day Classification

ML Assignment 2, M.Tech AIML

## a. Problem statement

Ground-level ozone is a health issue in the Houston–Galveston–Brazoria area. The UCI ozone dataset records one day at a time with weather and upper-air measurements. The label is whether that day was an **8-hour ozone day** or a normal day.

This is binary classification:

- `normal` – 2374 days
- `ozone` – 160 days (~6%)

A model that always predicts normal already gets about 93.7% accuracy, so I treat Accuracy as a weak metric here. Precision / Recall / F1 are for the ozone class. I also report AUC and MCC.

## b. Dataset description

**Ozone Level Detection (eight-hour peak set)** from UCI (Zhang, Fan, Yuan; Houston area, 1998–2004).

- Source: https://archive.ics.uci.edu/dataset/172/ozone+level+detection
- Instances: **2534** days
- Features: **72** numeric
  - WSR0–WSR23, WSR_PK, WSR_AV (resultant wind speed through the day, peak, average)
  - T0–T23, T_PK, T_AV (temperature through the day, peak, average)
  - T / RH / U / V / HT at 850, 700 and 500 hPa
  - KI, TT, SLP, SLP_change, Precp
- Target: `OzoneDay` ∈ {normal, ozone}
- Missing values: none in the copy I used (OpenML `ozone-level-8hr`; the raw UCI file marks some values as `?`, this version is already filled)

I did not keep a Date column. Train/test: 80/20 stratified, random_state=42 → 2027 train, 507 test (32 ozone days in test). Same hold-out is `test_data.csv`.

Features were standardised with `StandardScaler` fitted on train only (`model/scaler.joblib`). Wind speed, temperature and geopotential height are on very different scales, so this matters for logreg / KNN / SVM.

## c. Github Repository Link

**GitHub:** (https://github.com/priyanshulohia2002/ml_project)

**Streamlit app:** ()

```
app.py
train.py
requirements.txt
README.md
test_data.csv
data/ozone_8hr.csv
model/   (logreg, dtree, knn, nb, svm, scaler)
```

```
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Upload `test_data.csv` in the app.

## d. Models used

Same split for all five. Precision / Recall / F1 are for **ozone**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8462 | 0.9120 | 0.2604 | 0.7812 | 0.3906 | 0.3921 |
| Decision Tree | 0.8166 | 0.7440 | 0.1980 | 0.6250 | 0.3008 | 0.2767 |
| KNN | 0.9467 | 0.8692 | 0.6667 | 0.3125 | 0.4255 | 0.4334 |
| Naive Bayes | 0.6706 | 0.8079 | 0.1390 | 0.8125 | 0.2374 | 0.2387 |
| Support Vector Machine | 0.8659 | 0.9365 | 0.2955 | 0.8125 | 0.4333 | 0.4379 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy 0.85 is *below* the 0.94 dummy, because `class_weight='balanced'` fires a lot of ozone alarms (precision only 0.26). Recall 0.78 and AUC 0.91 are actually good – it ranks ozone days well, the default threshold is just aggressive. |
| Decision Tree | Weakest ranker (AUC 0.74). Depth 6 on 72 correlated hourly readings is not enough, and it still over-predicts ozone (precision 0.20). |
| KNN | Highest accuracy (0.947), even a bit above the dummy. Precision 0.67 looks nice but recall is only 0.31 – it only catches about 10 of the 32 ozone days. In 72-D space a rare ozone day’s 5 neighbours are usually normal days, so majority vote stays “normal”. This is the accuracy trap on this set. |
| Naive Bayes | Lowest accuracy (0.67) and precision (0.14). T0–T23 and WSR0–WSR23 are strongly correlated, so the independence assumption fails and it flags way too many ozone days. Recall 0.81 is high for the same reason. |
| Support Vector Machine | Best MCC (0.44) and best AUC (0.94). Recall 0.81 with slightly better precision than logreg. RBF + balanced weights. Still not a precise filter (precision 0.30) – useful as a screening model, not as a final “this is definitely an ozone day” call. |
| **Overall winner?** | **SVM**, by MCC and AUC. KNN “wins” accuracy but that is not the point of this dataset. If the goal is catching ozone days, SVM or logreg. I would not use NB or the tree. |

---

App: upload `test_data.csv`, pick a model. Comparison table for all five, then confusion matrix and classification report for the selected one. `app.py` only loads the `.joblib` files; it does not call `fit()`.
