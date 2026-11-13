"""Shared data loading, preprocessing, model zoo and metrics for the HR-ML study."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
# HRML_DATA / HRML_RESULTS let the pipeline run on another dataset (e.g. ASN records) or write elsewhere
RESULTS = Path(os.environ.get("HRML_RESULTS", ROOT / "results"))
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
for _d in (TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

# ---------------------------------------------------------------- data

RENAME = {"KPIs_met >80%": "kpis_met", "awards_won?": "awards_won"}
CATEGORICAL = ["department", "region", "education", "recruitment_channel"]
NUMERIC = ["no_of_trainings", "age", "previous_year_rating", "length_of_service", "kpis_met", "awards_won",
           "avg_training_score"]
# Performance indicators: excluded from the "potential" model so the 9-box axes stay distinct.
PERFORMANCE_FEATURES = ["previous_year_rating", "kpis_met"]


def find_promotion_csv() -> Path:
    """Locate the labelled Kaggle/WNS promotion file (any CSV in data/raw with an is_promoted column)."""
    for p in sorted(RAW.glob("*.csv")):
        if "is_promoted" in pd.read_csv(p, nrows=1).columns:
            return p
    raise FileNotFoundError(f"No CSV with an 'is_promoted' column found in {RAW}. "
                            "Download train.csv from https://www.kaggle.com/datasets/arashnic/hr-ana")


def load_promotion(path: Path | None = None) -> pd.DataFrame:
    path = path or os.environ.get("HRML_DATA") or find_promotion_csv()
    df = pd.read_csv(path).rename(columns=RENAME)
    df = df.drop_duplicates(subset="employee_id").reset_index(drop=True)
    df["age_group"] = pd.cut(df["age"], bins=[0, 29, 39, 200], labels=["<30", "30-39", ">=40"]).astype(str)
    df["gender"] = df["gender"].map({"m": "Male", "f": "Female"}).fillna(df["gender"])
    return df


def task_frame(df: pd.DataFrame, task: str):
    """Return X, y, sensitive attributes and feature lists for a task.

    talent    : y = is_promoted, all job-related features (gender excluded: fairness-through-unawareness)
    potential : y = is_promoted, performance indicators removed (potential axis of the 9-box)
    performance: y = previous_year_rating >= 4 (high performer), rows with a rating only;
                 is_promoted excluded because promotion is decided after the rating (temporal leakage)
    """
    # Some redistributions of the dataset (e.g. Kaggle arashnic/hr-ana) omit the KPI column
    num = [c for c in NUMERIC if c in df.columns]
    if task == "talent":
        data, y = df, df["is_promoted"].astype(int)
    elif task == "potential":
        data, y = df, df["is_promoted"].astype(int)
        num = [c for c in num if c not in PERFORMANCE_FEATURES]
    elif task == "performance":
        data = df[df["previous_year_rating"].notna()].reset_index(drop=True)
        y = (data["previous_year_rating"] >= 4).astype(int)
        num = [c for c in num if c != "previous_year_rating"]
    else:
        raise ValueError(task)
    X = data[CATEGORICAL + num].copy()
    sens = data[["gender", "age_group"]].copy()
    return X, y.reset_index(drop=True), sens, CATEGORICAL, num


def dev_holdout(X, y, sens):
    """Set aside a fixed 20% development partition used ONLY for hyper-parameter tuning, so that no row used to
    choose hyper-parameters is ever used for evaluation (avoids selection bias; Cawley & Talbot, 2010)."""
    dev, ev = train_test_split(np.arange(len(y)), test_size=0.8, stratify=y, random_state=SEED)
    take = lambda ix: (X.iloc[ix].reset_index(drop=True), y.iloc[ix].reset_index(drop=True),
                       sens.iloc[ix].reset_index(drop=True))
    return take(dev), take(ev)


def split(X, y, sens, seed: int):
    """Stratified 60/10/10/20 split of the EVALUATION partition (rows outside the tuning partition):
    train / validation (threshold) / calibration (conformal) / test."""
    _, (X, y, sens) = dev_holdout(X, y, sens)
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, rest = train_test_split(rest, test_size=0.75, stratify=y.iloc[rest], random_state=seed)
    ca, te = train_test_split(rest, test_size=2 / 3, stratify=y.iloc[rest], random_state=seed)
    take = lambda ix: (X.iloc[ix].reset_index(drop=True), y.iloc[ix].reset_index(drop=True),
                       sens.iloc[ix].reset_index(drop=True))
    return {"train": take(tr), "val": take(va), "cal": take(ca), "test": take(te)}


# ---------------------------------------------------------------- models

def preprocessor(cat, num, scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], verbose_feature_names_out=False)


def model_zoo(pos_weight: float):
    """Estimators and randomized-search spaces. Default: no class re-weighting (keeps probabilities calibrated);
    imbalance is handled by threshold selection. LightGBM-CW is the cost-sensitive ablation."""
    return {
        "LR": (LogisticRegression(max_iter=2000), True,
               {"clf__C": np.logspace(-3, 2, 20)}),
        "RF": (RandomForestClassifier(n_estimators=400, n_jobs=N_JOBS, random_state=SEED), False,
               {"clf__max_depth": [None, 8, 12, 16, 24], "clf__min_samples_leaf": [1, 2, 5, 10],
                "clf__max_features": ["sqrt", 0.3, 0.5]}),
        "XGBoost": (XGBClassifier(n_estimators=400, tree_method="hist", n_jobs=N_JOBS, random_state=SEED,
                                  eval_metric="logloss"), False,
                    {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_depth": [3, 4, 6, 8],
                     "clf__subsample": [0.7, 0.85, 1.0], "clf__colsample_bytree": [0.6, 0.8, 1.0],
                     "clf__min_child_weight": [1, 3, 5]}),
        "LightGBM": (LGBMClassifier(n_estimators=400, n_jobs=N_JOBS, random_state=SEED, verbose=-1), False,
                     {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__num_leaves": [15, 31, 63],
                      "clf__min_child_samples": [10, 20, 50], "clf__subsample": [0.7, 0.85, 1.0],
                      "clf__subsample_freq": [1], "clf__colsample_bytree": [0.6, 0.8, 1.0]}),
        "LightGBM-CW": (LGBMClassifier(n_estimators=400, n_jobs=N_JOBS, random_state=SEED, verbose=-1,
                                       scale_pos_weight=pos_weight), False,
                        {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__num_leaves": [15, 31, 63],
                         "clf__min_child_samples": [10, 20, 50], "clf__colsample_bytree": [0.6, 0.8, 1.0]}),
        "CatBoost": (CatBoostClassifier(iterations=500, verbose=0, thread_count=N_JOBS, random_seed=SEED), False,
                     {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__depth": [4, 6, 8],
                      "clf__l2_leaf_reg": [1, 3, 5, 10]}),
        "MLP": (MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=300,
                              random_state=SEED), True,
                {"clf__alpha": np.logspace(-5, -1, 10), "clf__learning_rate_init": [1e-3, 3e-3, 1e-2]}),
    }


def build_pipeline(estimator, scale: bool, cat, num) -> Pipeline:
    return Pipeline([("pre", preprocessor(cat, num, scale)), ("clf", estimator)])


# ---------------------------------------------------------------- metrics

def best_f1_threshold(y, p) -> float:
    grid = np.linspace(0.02, 0.98, 97)
    return float(grid[np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid])])


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[ids == b].mean() - p[ids == b].mean()) * (ids == b).mean()
                     for b in range(bins) if (ids == b).any()))


def scores(y, p, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Accuracy": accuracy_score(y, yhat), "MacroF1": f1_score(y, yhat, average="macro", zero_division=0),
            "F1": f1_score(y, yhat, zero_division=0), "Precision": precision_score(y, yhat, zero_division=0),
            "Recall": recall_score(y, yhat), "MCC": matthews_corrcoef(y, yhat),
            "BalAcc": balanced_accuracy_score(y, yhat), "Brier": brier_score_loss(y, p), "ECE": ece(y, p),
            "threshold": thr}
