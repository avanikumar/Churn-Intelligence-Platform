"""
Wraps SHAP so the API can return "why did this customer get this score"
as a ranked list of {feature, impact} instead of a raw model number.
"""
import json
import joblib
import shap
import pandas as pd

MODEL_DIR = "models"


class ChurnExplainer:
    def __init__(self):
        self.model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
        with open(f"{MODEL_DIR}/feature_cols.json") as f:
            self.feature_cols = json.load(f)
        with open(f"{MODEL_DIR}/best_model_name.json") as f:
            self.model_name = json.load(f)["name"]

        # TreeExplainer works for xgboost/lightgbm/catboost/random_forest.
        # Logistic regression falls back to a linear explainer.
        if self.model_name == "logistic_regression":
            self.scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
            self.explainer = shap.LinearExplainer(self.model, masker=shap.maskers.Independent(
                pd.DataFrame([[0] * len(self.feature_cols)], columns=self.feature_cols)
            ))
        else:
            self.scaler = None
            self.explainer = shap.TreeExplainer(self.model)

    def explain_one(self, row: pd.DataFrame, top_k: int = 5):
        X = row[self.feature_cols]
        if self.scaler is not None:
            X_input = self.scaler.transform(X)
        else:
            X_input = X

        shap_values = self.explainer.shap_values(X_input)
        # Some models return a list per class; normalize to a single array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        contributions = list(zip(self.feature_cols, shap_values[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return [
            {"feature": feat, "impact": round(float(val), 4)}
            for feat, val in contributions[:top_k]
        ]
