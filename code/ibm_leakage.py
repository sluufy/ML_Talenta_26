"""Experiment 6: target-leakage audit of the IBM HR 'PerformanceRating' task (OpenML 43897).

Shows that PerformanceRating is a deterministic function of PercentSalaryHike, so models that include it
reach near-perfect scores, while without it performance is at chance level (checked with a label-permutation test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from common import RAW, SEED, TABLES, build_pipeline, model_zoo

df = pd.read_csv(RAW / "openml_43897_ibm-employee-performance.csv")
y = (df["PerformanceRating"].astype(int) == 4).astype(int)
constant = [c for c in df.columns if df[c].nunique() == 1] + ["EmployeeNumber"]
X = df.drop(columns=["PerformanceRating"] + constant)
cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
num = [c for c in X.columns if c not in cat]

# Deterministic rule check
rule = (df["PercentSalaryHike"] >= 20).astype(int)
rule_acc = accuracy_score(y, rule)
print(f"Rule 'rating=4 iff PercentSalaryHike>=20' accuracy = {rule_acc:.4f}; positive rate = {y.mean():.4f}")

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
zoo = model_zoo(float((y == 0).sum() / (y == 1).sum()))
rows = []
for setting, drop in [("with PercentSalaryHike", []), ("without PercentSalaryHike", ["PercentSalaryHike"])]:
    n_ = [c for c in num if c not in drop]
    for name in ["LR", "RF", "XGBoost", "LightGBM", "CatBoost"]:
        est, scale, _ = zoo[name]
        pipe = build_pipeline(clone(est), scale, cat, n_)
        p = cross_val_predict(pipe, X[cat + n_], y, cv=cv, method="predict_proba")[:, 1]
        rows.append({"setting": setting, "model": name, "ROC-AUC": roc_auc_score(y, p),
                     "PR-AUC": average_precision_score(y, p), "Accuracy@0.5": accuracy_score(y, p >= 0.5),
                     "F1@0.5": f1_score(y, p >= 0.5, zero_division=0)})
        print(rows[-1])

res = pd.DataFrame(rows)
res.to_csv(TABLES / "ibm_leakage.csv", index=False)

# Permutation test for the leakage-free setting (is AUC distinguishable from chance?)
est, scale, _ = zoo["LightGBM"]
pipe = build_pipeline(clone(est), scale, cat, [c for c in num if c != "PercentSalaryHike"])
score, perm, pval = permutation_test_score(pipe, X.drop(columns=["PercentSalaryHike"]), y, cv=cv,
                                           scoring="roc_auc", n_permutations=100, random_state=SEED, n_jobs=-1)
pd.DataFrame([{"model": "LightGBM", "AUC": score, "perm_mean": np.mean(perm), "perm_sd": np.std(perm),
               "p_value": pval, "majority_class_accuracy": 1 - y.mean(), "rule_accuracy": rule_acc}]
             ).to_csv(TABLES / "ibm_permutation_test.csv", index=False)
print(f"Permutation test: AUC={score:.3f}, null mean={np.mean(perm):.3f}±{np.std(perm):.3f}, p={pval:.3f}")
