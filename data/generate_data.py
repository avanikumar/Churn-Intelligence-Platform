"""
Generates a realistic synthetic B2B/SaaS customer churn dataset.

Why synthetic data? So you can clone this repo and run the whole
pipeline immediately, with zero setup, before plugging in a real
dataset (e.g. from Kaggle's Telco Customer Churn, or your own
company data later).

Run:
    python data/generate_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000


def generate():
    customer_id = [f"CUST{i:05d}" for i in range(N)]

    tenure_months = np.random.gamma(shape=2.0, scale=12, size=N).clip(1, 72).astype(int)
    monthly_spend = np.random.normal(120, 45, N).clip(10, 500)
    contract_type = np.random.choice(
        ["month-to-month", "one-year", "two-year"], N, p=[0.55, 0.3, 0.15]
    )
    logins_last_30d = np.random.poisson(8, N)
    usage_drop_pct = np.random.normal(10, 25, N).clip(-50, 100)
    support_tickets_90d = np.random.poisson(1.2, N)
    nps_score = np.random.randint(0, 11, N)
    days_since_last_login = np.random.exponential(7, N).clip(0, 120).astype(int)
    discount_pct = np.random.choice([0, 5, 10, 15, 20], N, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    payment_failures_90d = np.random.poisson(0.3, N)
    is_premium = np.random.choice([0, 1], N, p=[0.7, 0.3])

    # Build churn probability from a believable business logic mix,
    # then sample actual churn from it (so the model has real signal to learn).
    logit = (
        -2.2
        + 0.9 * (contract_type == "month-to-month")
        - 0.6 * (contract_type == "two-year")
        + 0.03 * usage_drop_pct
        + 0.05 * days_since_last_login
        + 0.35 * support_tickets_90d
        - 0.25 * nps_score / 10 * 5
        - 0.02 * tenure_months
        + 0.4 * payment_failures_90d
        - 0.3 * is_premium
        - 0.015 * logins_last_30d
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = np.random.binomial(1, prob_churn)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend.round(2),
            "contract_type": contract_type,
            "logins_last_30d": logins_last_30d,
            "usage_drop_pct": usage_drop_pct.round(1),
            "support_tickets_90d": support_tickets_90d,
            "nps_score": nps_score,
            "days_since_last_login": days_since_last_login,
            "discount_pct": discount_pct,
            "payment_failures_90d": payment_failures_90d,
            "is_premium": is_premium,
            "churned": churn,
        }
    )
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/customers.csv", index=False)
    print(f"Generated {len(df)} rows -> data/customers.csv")
    print(f"Churn rate: {df['churned'].mean():.2%}")
