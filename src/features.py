"""
Feature engineering: turns raw customer columns into model-ready features.
Kept as its own module so training and serving (the API) both call the
exact same transformation -> no train/serve skew.
"""
import pandas as pd

CATEGORICAL_COLS = ["contract_type"]
NUMERIC_COLS = [
    "tenure_months",
    "monthly_spend",
    "logins_last_30d",
    "usage_drop_pct",
    "support_tickets_90d",
    "nps_score",
    "days_since_last_login",
    "discount_pct",
    "payment_failures_90d",
    "is_premium",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # A few derived features that tend to carry real churn signal
    df["spend_per_login"] = df["monthly_spend"] / (df["logins_last_30d"] + 1)
    df["is_inactive"] = (df["days_since_last_login"] > 14).astype(int)
    df["low_nps"] = (df["nps_score"] <= 6).astype(int)
    df["high_risk_contract"] = (df["contract_type"] == "month-to-month").astype(int)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    drop_cols = {"customer_id", "churned"}
    return [c for c in df.columns if c not in drop_cols]
