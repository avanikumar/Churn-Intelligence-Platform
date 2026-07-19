"""
Business-facing dashboard for the Churn Intelligence Platform.

This uses Streamlit instead of React/Next.js so you can build a working
UI without learning frontend frameworks first. Once you're comfortable,
see the README for how to port this to a Next.js app calling the same
FastAPI backend.

Run (with the API already running in another terminal):
    streamlit run dashboard/app.py
"""
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")
st.title("📊 Enterprise Customer Churn Intelligence Platform")

tab1, tab2 = st.tabs(["Portfolio Overview", "Single Customer Lookup"])

with tab1:
    st.subheader("Customer Portfolio Risk")
    try:
        df = pd.read_csv("data/customers.csv")
        st.metric("Total customers", len(df))
        st.metric("Historical churn rate", f"{df['churned'].mean():.1%}")

        fig = px.histogram(df, x="tenure_months", color="churned",
                            title="Tenure distribution by churn outcome", barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(df, x="contract_type", y="usage_drop_pct", color="churned",
                       title="Usage drop by contract type")
        st.plotly_chart(fig2, use_container_width=True)
    except FileNotFoundError:
        st.warning("Run `python data/generate_data.py` first to create the dataset.")

with tab2:
    st.subheader("Score a Customer")
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_id = st.text_input("Customer ID", "CUST00001")
        tenure_months = st.slider("Tenure (months)", 1, 72, 12)
        monthly_spend = st.number_input("Monthly spend ($)", 10.0, 500.0, 120.0)
        contract_type = st.selectbox("Contract type", ["month-to-month", "one-year", "two-year"])
    with col2:
        logins_last_30d = st.slider("Logins (last 30 days)", 0, 60, 8)
        usage_drop_pct = st.slider("Usage drop (%)", -50, 100, 10)
        support_tickets_90d = st.slider("Support tickets (90 days)", 0, 15, 1)
        nps_score = st.slider("NPS score", 0, 10, 7)
    with col3:
        days_since_last_login = st.slider("Days since last login", 0, 120, 5)
        discount_pct = st.selectbox("Current discount (%)", [0, 5, 10, 15, 20])
        payment_failures_90d = st.slider("Payment failures (90 days)", 0, 5, 0)
        is_premium = st.selectbox("Premium plan?", [0, 1])

    if st.button("Predict Churn Risk", type="primary"):
        payload = {
            "customer_id": customer_id,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend,
            "contract_type": contract_type,
            "logins_last_30d": logins_last_30d,
            "usage_drop_pct": usage_drop_pct,
            "support_tickets_90d": support_tickets_90d,
            "nps_score": nps_score,
            "days_since_last_login": days_since_last_login,
            "discount_pct": discount_pct,
            "payment_failures_90d": payment_failures_90d,
            "is_premium": is_premium,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            risk = result["churn_probability"]
            st.metric("Churn Probability", f"{risk:.1%}")
            st.progress(min(risk, 1.0))
            st.write(f"**Urgency:** {result['urgency'].upper()}")
            st.info(result["summary"])
            st.success(f"Recommended action: {result['recommended_action']}")

            st.subheader("Why this score? (SHAP)")
            shap_df = pd.DataFrame(result["shap_contributions"])
            fig = px.bar(shap_df, x="impact", y="feature", orientation="h",
                         title="Top feature contributions to churn risk")
            st.plotly_chart(fig, use_container_width=True)
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the API. Make sure it's running: `uvicorn src.api:app --reload --port 8000`")
