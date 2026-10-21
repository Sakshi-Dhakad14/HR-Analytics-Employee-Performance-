import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, roc_curve)

DATA_PATH = "data/IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv"

df = pd.read_csv(DATA_PATH)
y = (df["Attrition"] == "Yes").astype(int)

X = df.drop(columns=["Attrition"])
cat_cols = X.select_dtypes(include="object").columns
X = pd.get_dummies(X, columns=cat_cols, dtype=int)
feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = {}
best_name, best_auc, best_model = None, 0, None

for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    step = max(1, len(fpr) // 60)  # chart ke liye points kam karo

    auc = roc_auc_score(y_test, proba)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred, zero_division=0), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "roc_auc": round(auc, 3),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "roc_curve": {"fpr": fpr[::step].round(3).tolist(),
                      "tpr": tpr[::step].round(3).tolist()},
    }
    print(f"{name}: AUC={auc:.3f}")
    if auc > best_auc:
        best_name, best_auc, best_model = name, auc, model

# Feature importance (Random Forest se)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=feature_columns)
imp = imp.sort_values(ascending=False).head(10)

output = {
    "best_model": best_name,
    "models": results,
    "feature_importance": {"labels": imp.index.tolist(),
                           "values": imp.round(4).tolist()},
}

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")
with open("models/metrics.json", "w") as f:
    json.dump(output, f, indent=2)

print("Best model:", best_name)