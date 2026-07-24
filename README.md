# Enterprise Customer Churn Intelligence Platform

An end-to-end machine learning system that predicts customer churn, explains the drivers behind each prediction, quantifies customer value, and recommends the retention action with the highest expected return -rather than stopping at a risk score.

Most churn projects answer one question: *who is going to leave?* This platform is built to answer four:

- **Who** is likely to churn
- **Why** they're likely to churn
- **What** action should be taken
- **Which** retention strategy delivers the best ROI

## Architecture

```
Customer Data
    │
    ▼
Feature Engineering
    │
    ▼
Churn Prediction (LightGBM / XGBoost / CatBoost)
    │
    ▼
Explainability (SHAP)
    │
    ▼
Customer Segmentation
    │
    ▼
Customer Lifetime Value
    │
    ▼
Retention Strategy Generator (LLM)
    │
    ▼
Dashboard (React)
    │
    ▼
REST API (FastAPI)
    │
    ▼
Docker + AWS
```

## Design rationale

A churn probability alone isn't actionable. A customer flagged as "high risk" could be a lost cause no offer will retain, a customer who was never going to leave, or a customer whose lifetime value doesn't justify the cost of a retention offer in the first place. Treating all high-risk customers identically wastes retention budget on the wrong people.

This platform addresses that by combining four signals before recommending any action:

1. **Churn probability** -from gradient-boosted models trained on engineered customer features
2. **Explanation** - SHAP values showing which factors are driving each prediction, at both the population and individual customer level
3. **Customer lifetime value** - so retention effort is prioritized by value, not risk alone
4. **Treatment effect / uplift** - to separate customers who can actually be influenced by a retention offer from those who can't

The retention strategy layer is grounded in these outputs rather than generating recommendations from an LLM in isolation, which avoids unsupported or hallucinated numbers in the final recommendation.

## Tech stack

| Layer | Technology |
|---|---|
| Modeling | Python, LightGBM, XGBoost, CatBoost |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Backend / API | FastAPI |
| Database | PostgreSQL |
| Frontend | React |
| Deployment | Docker, AWS EC2 |

## Project status

| Module | Status |
|---|---|
| Data cleaning & feature engineering | Complete |
| Churn prediction models (LightGBM, XGBoost, CatBoost) | Complete |
| SHAP explainability | Complete |
| Customer segmentation | In progress |
| Customer lifetime value | Planned |
| Uplift / retention ROI modeling | Planned |
| LLM-based retention strategy generator | Planned |
| FastAPI backend | Planned |
| React dashboard | Planned |
| Docker + AWS deployment | Planned |

## Repository structure

```
├── data/                  # raw and processed data (excluded via .gitignore)
├── notebooks/             # exploratory analysis and experiments
├── src/
│   ├── features/          # feature engineering pipeline
│   ├── models/            # LightGBM / XGBoost / CatBoost training
│   ├── explainability/    # SHAP analysis
│   ├── segmentation/      # RFM and clustering
│   ├── clv/               # customer lifetime value estimation
│   ├── retention/         # uplift modeling and strategy generation
│   └── api/               # FastAPI application
├── frontend/              # React dashboard
├── docker/                # Dockerfiles and docker-compose configuration
└── docs/                  # additional documentation and references
```

## Setup

```bash
git clone https://github.com/<your-username>/churn-intelligence-platform.git
cd churn-intelligence-platform

python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows

pip install -r requirements.txt

python src/models/train.py
```

Docker Compose instructions will be added once the API and frontend reach a runnable state.

## Data

Development uses a public churn dataset (details in `data/README.md`), since the objective is to demonstrate the modeling and system-design approach rather than report results on proprietary enterprise data. The architecture is designed to generalize to any tabular customer dataset with usage, billing, and support-interaction history.

## Contributing / feedback

Issues and pull requests are welcome, particularly around the modules still in progress.

## License

MIT
