"""
FastAPI service for the Churn Intelligence Platform.

Run:
    uvicorn src.api:app --reload --port 8000
Then open:
    http://localhost:8000/docs   (interactive Swagger UI)
"""
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from features import engineer_features
from explain import ChurnExplainer
from recommend import generate_explanation

MODEL_DIR = "models"

app = FastAPI(
    title="Churn Intelligence Platform API",
    description="Predicts churn, explains why, and recommends a retention action.",
    version="1.0.0",
)

model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
with open(f"{MODEL_DIR}/feature_cols.json") as f:
    feature_cols = json.load(f)
with open(f"{MODEL_DIR}/best_model_name.json") as f:
    model_name = json.load(f)["name"]
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl") if model_name == "logistic_regression" else None
explainer = ChurnExplainer()


class Customer(BaseModel):
    customer_id: str
    tenure_months: int
    monthly_spend: float
    contract_type: str  # "month-to-month" | "one-year" | "two-year"
    logins_last_30d: int
    usage_drop_pct: float
    support_tickets_90d: int
    nps_score: int
    days_since_last_login: int
    discount_pct: int
    payment_failures_90d: int
    is_premium: int


@app.get("/")
def root():
    return {"status": "ok", "model": model_name}


@app.post("/predict")
def predict(customer: Customer):
    try:
        raw = pd.DataFrame([customer.dict()])
        engineered = engineer_features(raw)

        # Make sure all training columns exist (one-hot cols may be missing
        # for a single row), fill any gaps with 0
        for col in feature_cols:
            if col not in engineered.columns:
                engineered[col] = 0
        X = engineered[feature_cols]

        X_input = scaler.transform(X) if scaler is not None else X
        churn_prob = float(model.predict_proba(X_input)[0][1])

        shap_contributions = explainer.explain_one(engineered)
        explanation = generate_explanation(
            customer.customer_id, churn_prob, shap_contributions, customer.is_premium
        )

        return {
            "customer_id": customer.customer_id,
            "churn_probability": round(churn_prob, 4),
            "shap_contributions": shap_contributions,
            **explanation,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model-info")
def model_info():
    with open(f"{MODEL_DIR}/results.json") as f:
        results = json.load(f)
    return {"active_model": model_name, "all_model_results": results}
