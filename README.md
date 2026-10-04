# Falabella seller reactivation model (simulation)

A simulation of **Initiative 2** of our data strategy for Falabella's Sellers Management area (IE University, Data Management & Governance, Assignment 6).

The model scores every inactive seller by their probability of becoming active in the next 3 months, so account managers know who to contact first. It learns from the history of when sellers went from inactive to active, the history that the star schema built in Initiative 1 records.

> **All data in this repository is synthetic.** It is generated to match the figures in the assignment annex (20,000 sellers, about 40% active). It does not describe real Falabella sellers, and results on real data would differ.

## Results

| Metric | Value |
|---|---|
| Training data | 146,550 monthly snapshots of inactive sellers (months 12–30) |
| Out-of-time test | 34,043 snapshots (months 32–34) |
| AUC on the test months | 0.83 |
| Base 3-month activation rate | 10.5% |
| Activation rate in the top 10% of scores | 43.3% (4.1× the base rate) |
| Share of activations captured by the top 30% | 74.7% |
| Inactive sellers scored today | 11,826 |
| Expected activations, top 3,000 by score | ~781 |
| Expected activations, 3,000 at random | ~277 (the model gives 2.8× more) |

The features that raise the score most are products listed, onboarding steps completed, portal logins and having sold before. Months inactive, self sign-up onboarding, being a micro-sized seller and time since registration lower it.

## Repository structure

```
data/
  sellers.csv                    one row per seller: category, country, business unit,
                                 onboarding channel, size band, registration month,
                                 onboarding steps, contact completeness
  monthly_snapshots.csv.gz       one row per seller per month: tenure, logins, listings,
                                 ever sold, months inactive, active flag, 3-month label
model/
  reactivation_model.joblib      trained scikit-learn pipeline (scaler + one-hot + logistic regression)
  metrics.json                   AUC, decile activation rates, coefficients
outputs/
  inactive_sellers_scored.csv    today's 11,826 inactive sellers with their score
  logistic_curve.mp4             26-second text-free animation used in the presentation:
                                 the logistic curve is fitted, then today's sellers are
                                 mapped onto it to read off their score
src/
  generate_data.py               builds the synthetic data
  train_model.py                 trains, tests and scores
  make_curve_animation.py        renders the animation
  bigquery_ml.sql                how the same model would run in BigQuery ML
fonts/                           Public Sans (SIL Open Font License), used by the animation
```

## Run it

Python 3.10+ and, for the animation, ffmpeg.

```bash
pip install -r requirements.txt
python src/generate_data.py      # writes data/
python src/train_model.py        # writes model/ and outputs/inactive_sellers_scored.csv
python src/make_curve_animation.py   # writes outputs/logistic_curve.mp4
```

Run the scripts from the repository root. Generation uses a fixed random seed (42), so the results above are reproducible.

## Use the trained model

```python
import joblib, pandas as pd

model = joblib.load("model/reactivation_model.joblib")
features = ["tenure_months", "logins_last_3m", "log_listings", "ever_sold",
            "months_inactive", "onboarding_steps",
            "category", "country", "business_unit", "onboarding_channel", "size_band"]
# df must contain these columns; log_listings = log(1 + listings)
scores = model.predict_proba(df[features])[:, 1]
```

## How the data is generated

- 20,000 sellers register at random over 36 months; month 36 is "today".
- Each seller has observed features and a hidden engagement level the model never sees, so the model cannot be perfect.
- Each month, an inactive seller becomes active with a probability that depends on their features, listings, logins, whether they sold before and how long they have been inactive. Active sellers can also stop selling.
- The label for a snapshot is 1 if an inactive seller becomes active within the next 3 months.

## Context

This model belongs to a one-year data strategy built on one cloud stack: Datastream to bring data in, BigQuery as the warehouse, dbt to build the golden record and star schema (with SCD Type 2 history), Dataplex for catalog and data quality, BigQuery ML for this model, and Looker to give account managers the ranked list.
