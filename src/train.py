"""
Trains and compares several churn models, logs every run to MLflow,
and saves the best model + feature list to disk for the API to load.

Run:
    python src/train.py
Then view results:
    mlflow ui   (opens http://localhost:5000)
"""
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from features import engineer_features, get_feature_columns

MODEL_DIR = "models"


def load_data():
    df = pd.read_csv("data/customers.csv")
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["churned"]
    return X, y, feature_cols


def evaluate(y_true, y_pred, y_prob):
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def main():
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=42),
        "lightgbm": LGBMClassifier(random_state=42, verbose=-1),
        "catboost": CatBoostClassifier(verbose=0, random_state=42),
    }

    mlflow.set_experiment("churn-intelligence-platform")

    results = {}
    best_name, best_score, best_model = None, -1, None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            if name == "logistic_regression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            metrics = evaluate(y_test, y_pred, y_prob)
            mlflow.log_params({"model_type": name})
            mlflow.log_metrics(metrics)

            results[name] = metrics
            print(f"{name:20s} ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")

            if metrics["roc_auc"] > best_score:
                best_score, best_name, best_model = metrics["roc_auc"], name, model

    print(f"\nBest model: {best_name} (ROC-AUC={best_score:.4f})")

    # Persist best model + supporting artifacts for the API to use
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    with open(f"{MODEL_DIR}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(f"{MODEL_DIR}/best_model_name.json", "w") as f:
        json.dump({"name": best_name}, f)
    with open(f"{MODEL_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved best model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
