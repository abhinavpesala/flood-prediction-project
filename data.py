import pandas as pd
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ============================
# LOAD DATA
# ============================
df = pd.read_csv("flood.csv")

# ============================
# FEATURE ENGINEERING
# ============================
df["Climate_Risk"] = df["MonsoonIntensity"] + df["ClimateChange"] + df["CoastalVulnerability"]

df["Infrastructure_Risk"] = df["DeterioratingInfrastructure"] + df["DrainageSystems"] + df["DamsQuality"]

df["Human_Impact"] = df["Urbanization"] + df["Deforestation"] + df["Encroachments"] + df["PopulationScore"]

df["Environmental_Risk"] = df["WetlandLoss"] + df["Siltation"] + df["AgriculturalPractices"]

df["Disaster_Management"] = df["IneffectiveDisasterPreparedness"] + df["InadequatePlanning"] + df["PoliticalFactors"]

# ============================
# TARGET
# ============================
df["Flood"] = (df["FloodProbability"] >= 0.5).astype(int)

X = df.drop(columns=["FloodProbability", "Flood"])
y = df["Flood"]

# ============================
# CROSS VALIDATION (BEST PRACTICE)
# ============================
model_cv = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model_cv,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("\n===== CROSS VALIDATION (ROC AUC) =====")
print("Scores:", cv_scores)
print("Mean ROC AUC:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# ============================
# TRAIN / TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# FINAL MODEL TRAINING
# ============================
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ============================
# TEST EVALUATION
# ============================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n===== TEST SET EVALUATION =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_proba))

# ============================
# SAVE MODEL
# ============================
with open("flood_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "columns": X.columns.tolist()
    }, f) 

print("\n✅ Model saved successfully")
