"""
Turns a churn score + SHAP reasons into a human-readable explanation and
a recommended retention action.

Ships as free rule-based logic (no API key needed) so the whole project
runs for $0. There's an optional LLM mode if you want to add a real
OpenAI/Anthropic call later — see `generate_with_llm()`.
"""
import os

FEATURE_LABELS = {
    "usage_drop_pct": "a drop in product usage",
    "days_since_last_login": "long inactivity (no recent logins)",
    "support_tickets_90d": "a high number of recent support tickets",
    "nps_score": "a low satisfaction (NPS) score",
    "payment_failures_90d": "recent payment failures",
    "high_risk_contract": "being on a flexible month-to-month contract",
    "tenure_months": "short tenure with the company",
    "is_premium": "not being on a premium plan",
    "spend_per_login": "low value gained per visit",
    "low_nps": "low satisfaction score",
    "is_inactive": "recent inactivity",
}


def humanize_reasons(shap_contributions, top_n=3):
    risky = [c for c in shap_contributions if c["impact"] > 0][:top_n]
    reasons = [FEATURE_LABELS.get(c["feature"], c["feature"]) for c in risky]
    return reasons


def recommend_action(churn_prob: float, reasons: list, is_premium: int):
    if churn_prob >= 0.75:
        urgency = "high"
        if "payment failures" in " ".join(reasons):
            action = "Reach out personally to resolve the billing issue and offer a payment plan."
        elif "contract" in " ".join(reasons):
            action = "Offer a discounted annual contract to lock in retention."
        else:
            action = "Offer a premium upgrade with a 20% loyalty discount and a personal check-in call."
    elif churn_prob >= 0.4:
        urgency = "medium"
        action = "Send a re-engagement email highlighting underused features, with a 10% incentive."
    else:
        urgency = "low"
        action = "No urgent action needed — include in the standard quarterly newsletter."

    return {"urgency": urgency, "recommended_action": action}


def generate_explanation(customer_id, churn_prob, shap_contributions, is_premium):
    reasons = humanize_reasons(shap_contributions)
    reasons_text = ", ".join(reasons) if reasons else "a mix of moderate risk factors"
    rec = recommend_action(churn_prob, reasons, is_premium)

    summary = (
        f"Customer {customer_id} has a {churn_prob:.0%} chance of churning, "
        f"driven mainly by {reasons_text}."
    )
    return {
        "summary": summary,
        "top_reasons": reasons,
        "urgency": rec["urgency"],
        "recommended_action": rec["recommended_action"],
    }


def generate_with_llm(customer_id, churn_prob, shap_contributions, is_premium):
    """
    Optional upgrade: replace the rule-based summary above with a real LLM
    call for richer, more natural language. Requires ANTHROPIC_API_KEY set
    as an environment variable. Left here as a clearly-marked extension
    point — not required to run the project.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    reasons = humanize_reasons(shap_contributions)
    prompt = (
        f"A customer has a {churn_prob:.0%} churn risk. Top risk factors: "
        f"{', '.join(reasons)}. Premium customer: {bool(is_premium)}. "
        f"In 2-3 sentences, explain why they're at risk and recommend one "
        f"concrete retention action."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text
