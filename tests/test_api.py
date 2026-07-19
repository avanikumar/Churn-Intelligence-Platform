"""
Basic smoke tests. Run with: pytest tests/
Requires model artifacts to exist first (python src/train.py).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict():
    payload = {
        "customer_id": "TEST001",
        "tenure_months": 3,
        "monthly_spend": 99.0,
        "contract_type": "month-to-month",
        "logins_last_30d": 1,
        "usage_drop_pct": 60,
        "support_tickets_90d": 4,
        "nps_score": 2,
        "days_since_last_login": 30,
        "discount_pct": 0,
        "payment_failures_90d": 1,
        "is_premium": 0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["churn_probability"] <= 1
    assert "recommended_action" in body
