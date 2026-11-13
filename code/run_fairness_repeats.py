"""Fairness audit and threshold post-processing over the 10 benchmark splits (mean ± SD).

Talent task: F1-optimal threshold. Performance task: MCC-optimal threshold, because with a 43% prevalence and weak
signal the F1-optimal threshold degenerates to labelling almost everyone positive.
"""
import json

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from common import TABLES, best_f1_threshold, build_pipeline, load_promotion, model_zoo, split, task_frame


def best_mcc_threshold(y, p) -> float:
    grid = np.linspace(0.02, 0.98, 97)
    return float(grid[np.argmax([matthews_corrcoef(y, p >= t) for t in grid])])


df = load_promotion()
chosen = json.loads((TABLES / "chosen_models.json").read_text())
rows = []
for task in ["talent", "performance"]:
    X, y, sens, cat, num = task_frame(df, task)
    name = chosen[task]
    params = json.loads((TABLES / f"best_params_{task}.json").read_text())[name]
    for r in range(10):
        parts = split(X, y, sens, seed=1000 + r)
        (Xtr, ytr, _), (Xva, yva, sva), (Xte, yte, ste) = parts["train"], parts["val"], parts["test"]
        est, scale, _ = model_zoo(float((ytr == 0).sum() / (ytr == 1).sum()))[name]
        pipe = build_pipeline(est, scale, cat, num).set_params(**params).fit(Xtr, ytr)
        pva, pte = pipe.predict_proba(Xva)[:, 1], pipe.predict_proba(Xte)[:, 1]
        thr = best_f1_threshold(yva, pva) if task == "talent" else best_mcc_threshold(yva, pva)
        yhat = (pte >= thr).astype(int)
        for attr in ["gender", "age_group"]:
            preds = {"Unmitigated": yhat}
            to = ThresholdOptimizer(estimator=pipe, constraints="equalized_odds", objective="balanced_accuracy_score",
                                    prefit=True, predict_method="predict_proba")
            to.fit(Xva, yva, sensitive_features=sva[attr])
            preds["ThresholdOptimizer (equalized odds)"] = to.predict(Xte, sensitive_features=ste[attr],
                                                                      random_state=r)
            for method, yp in preds.items():
                rows.append({"task": task, "repeat": r, "attribute": attr, "method": method, "threshold": thr,
                             "F1": f1_score(yte, yp), "MCC": matthews_corrcoef(yte, yp),
                             "BalAcc": balanced_accuracy_score(yte, yp), "PosRate": np.mean(yp),
                             "DPD": demographic_parity_difference(yte, yp, sensitive_features=ste[attr]),
                             "EOD": equalized_odds_difference(yte, yp, sensitive_features=ste[attr])})
        print(task, "repeat", r, "done")

res = pd.DataFrame(rows)
res.to_csv(TABLES / "fairness_repeats_runs.csv", index=False)
summ = res.drop(columns="repeat").groupby(["task", "attribute", "method"]).agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.to_csv(TABLES / "fairness_repeats_summary.csv")
pd.set_option("display.width", 250)
print(summ[[c for c in summ.columns if c.split("_")[0] in ("F1", "MCC", "BalAcc", "PosRate", "DPD", "EOD", "threshold")]]
      .round(3).to_string())
