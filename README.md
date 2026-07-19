# Enterprise Customer Churn Intelligence Platform

An end-to-end ML system that doesn't just predict churn — it explains *why*
a customer is at risk and recommends *what to do about it*.

```
Customer Data (CSV / synthetic generator)
        │
        ▼
Feature Engineering (src/features.py)
        │
        ▼
Model Training + Comparison (src/train.py)
   Logistic Regression | Random Forest | XGBoost | LightGBM | CatBoost
   → tracked with MLflow
        │
        ▼
Explainability (src/explain.py — SHAP)
        │
        ▼
Retention Recommendation (src/recommend.py)
        │
        ▼
REST API (src/api.py — FastAPI)
        │
        ▼
Dashboard (dashboard/app.py — Streamlit)
```

## What this demonstrates

- Comparing multiple ML models on the same problem, not just picking one
- Explainability (SHAP) instead of a black-box score
- Turning a model output into a business recommendation
- A real API, not just a notebook
- Experiment tracking (MLflow)
- Containerization (Docker)
- Basic CI (GitHub Actions)
- Tests

## Project structure

```
churn-intelligence-platform/
├── data/
│   └── generate_data.py      # creates a synthetic dataset
├── src/
│   ├── features.py           # feature engineering (shared by train + API)
│   ├── train.py               # trains & compares 5 models, logs to MLflow
│   ├── explain.py             # SHAP explainability wrapper
│   ├── recommend.py           # turns predictions into recommended actions
│   └── api.py                  # FastAPI service
├── dashboard/
│   └── app.py                  # Streamlit business dashboard
├── tests/
│   └── test_api.py
├── .github/workflows/ci.yml   # CI: installs, trains, tests on every push
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quickstart (run locally, no Docker)

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the synthetic dataset
python data/generate_data.py

# 4. Train and compare models (saves the best one to models/)
python src/train.py

# 5. Start the API
uvicorn src.api:app --reload --port 8000
# → open http://localhost:8000/docs to try it interactively

# 6. In a second terminal, start the dashboard
streamlit run dashboard/app.py
```

## Quickstart (Docker)

```bash
docker compose up --build
# API available at http://localhost:8000
```

## Swapping in real data

Replace `data/customers.csv` with your own dataset (e.g. the
[Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)),
matching or adapting the column names in `src/features.py`, then re-run
`python src/train.py`.

---

# Step-by-step: turning this into your own GitHub repo

I can't create a repo directly on your GitHub account (I don't have access
to it) — but here's exactly how to do it in about 5 minutes:

1. **Download the project** using the link I'll share after this message.
2. **Create a new repo on GitHub**: go to github.com → the "+" icon (top
   right) → "New repository" → name it e.g. `churn-intelligence-platform`
   → don't initialize with a README (you already have one) → Create.
3. **Push your local code** to it:
   ```bash
   cd churn-intelligence-platform
   git init
   git add .
   git commit -m "Initial commit: churn intelligence platform"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/churn-intelligence-platform.git
   git push -u origin main
   ```
4. **Verify CI runs**: go to the "Actions" tab on your GitHub repo — you
   should see the workflow from `.github/workflows/ci.yml` run
   automatically and pass.
5. **Polish for recruiters**:
   - Add a couple of screenshots of the dashboard to the README
   - Fill in the "About" section on the repo page (one-line description + topics like `machine-learning`, `fastapi`, `mlops`)
   - Pin the repo on your GitHub profile

---

# Learning resources (in the order I'd learn them)

You don't need to master everything before starting — run the project
first (steps above), then come back to understand each piece.

### 1. Python fundamentals (if shaky)
- [Python official tutorial](https://docs.python.org/3/tutorial/)
- [Corey Schafer's Python YouTube series](https://www.youtube.com/playlist?list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU)

### 2. Pandas (data manipulation — used everywhere in this repo)
- [Pandas official 10-minute intro](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Kaggle's free Pandas course](https://www.kaggle.com/learn/pandas)

### 3. Scikit-learn (the ML basics: train/test split, models, metrics)
- [Scikit-learn "Getting Started"](https://scikit-learn.org/stable/getting_started.html)
- [Kaggle's Intro to Machine Learning course](https://www.kaggle.com/learn/intro-to-machine-learning)

### 4. Gradient boosting models (XGBoost / LightGBM / CatBoost)
- [XGBoost official docs — "Introduction to Boosted Trees"](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
- [StatQuest's XGBoost video series](https://www.youtube.com/watch?v=OtD8wVaFm6E) (great intuition, no heavy math required)
- [LightGBM docs](https://lightgbm.readthedocs.io/)
- [CatBoost docs](https://catboost.ai/docs/)

### 5. SHAP (explainability)
- [SHAP official docs + examples](https://shap.readthedocs.io/en/latest/)
- [StatQuest: SHAP values explained](https://www.youtube.com/watch?v=9haIOplEIGM)

### 6. MLflow (experiment tracking)
- [MLflow "Getting Started" tutorial](https://mlflow.org/docs/latest/getting-started/index.html)

### 7. FastAPI (the backend/API layer)
- [Official FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/) — genuinely one of the best-written docs of any framework, do this one properly
- [FreeCodeCamp FastAPI course (video)](https://www.youtube.com/watch?v=0sOvCWFmrtA)

### 8. Streamlit (the dashboard used here — easiest way into building UIs)
- [Streamlit "Get Started"](https://docs.streamlit.io/get-started)

### 9. Docker (containerization)
- [Docker's official "Getting Started" guide](https://docs.docker.com/get-started/)
- [FreeCodeCamp Docker crash course (video)](https://www.youtube.com/watch?v=fqMOX6JJhGo)

### 10. Git & GitHub (if new to this)
- [GitHub's own Git handbook](https://guides.github.com/introduction/git-handbook/)
- [Learn Git Branching (interactive)](https://learngitbranching.js.org/)

### 11. (Later) React / Next.js — to replace the Streamlit dashboard
Once the rest feels comfortable, this is the natural upgrade path to make
the project even stronger for frontend-adjacent roles:
- [Next.js official "Learn" course](https://nextjs.org/learn)
- Point it at the same FastAPI endpoints (`/predict`, `/model-info`) — no backend changes needed.

### 12. (Later) AWS EC2 deployment
- [AWS's own EC2 "Get Started" guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)
- Deploy this repo's `Dockerfile` — search "deploy Docker container to EC2" once you're at this stage.

---

## Suggested order to actually build this (so it's not overwhelming)

1. Get the repo running locally exactly as-is (Quickstart above). Don't change anything yet — just get it working.
2. Read `src/features.py` and `src/train.py` line by line until you understand every line.
3. Swap in a real dataset (Telco churn from Kaggle) instead of the synthetic one.
4. Read `src/explain.py`, then `src/api.py`.
5. Customize `src/recommend.py` — this is the easiest place to add your own "product thinking" (different rules, different messaging).
6. Push to GitHub, get CI passing.
7. Write the README screenshots + polish.
8. (Stretch) Rebuild the dashboard in Next.js.
9. (Stretch) Deploy to AWS EC2 or Render/Railway (much simpler than EC2 if you want a live demo link fast).
