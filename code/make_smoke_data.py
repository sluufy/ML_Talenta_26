"""SMOKE TEST ONLY: random data with the Kaggle promotion schema, used to check that the code runs.
Never used for any reported result."""
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
rating = rng.choice([1, 2, 3, 4, 5, np.nan], n, p=[.1, .1, .3, .2, .22, .08])
score = rng.normal(63, 13, n).round()
kpi = rng.binomial(1, 0.35, n)
logit = -4 + 0.04 * (score - 63) + 1.2 * kpi + 0.4 * np.nan_to_num(rating - 3)
df = pd.DataFrame({
    "employee_id": np.arange(n),
    "department": rng.choice(["Sales & Marketing", "Operations", "Technology", "Analytics", "R&D"], n),
    "region": rng.choice([f"region_{i}" for i in range(1, 12)], n),
    "education": rng.choice(["Bachelor's", "Master's & above", "Below Secondary", None], n, p=[.65, .27, .03, .05]),
    "gender": rng.choice(["m", "f"], n, p=[.7, .3]),
    "recruitment_channel": rng.choice(["other", "sourcing", "referred"], n),
    "no_of_trainings": rng.integers(1, 5, n),
    "age": rng.integers(20, 60, n),
    "previous_year_rating": rating,
    "length_of_service": rng.integers(1, 30, n),
    "KPIs_met >80%": kpi,
    "awards_won?": rng.binomial(1, 0.03, n),
    "avg_training_score": score,
    "is_promoted": rng.binomial(1, 1 / (1 + np.exp(-logit))),
})
df.to_csv(sys.argv[1], index=False)
print(df.shape, df["is_promoted"].mean())
