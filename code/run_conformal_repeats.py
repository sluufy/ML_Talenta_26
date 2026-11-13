"""Conformal coverage of the chosen model over the 10 benchmark splits (mean ± SD), marginal vs Mondrian."""
import json

import numpy as np
import pandas as pd

from common import TABLES, build_pipeline, load_promotion, model_zoo, split, task_frame
from run_trust import conformal_sets

df = load_promotion()
chosen = json.loads((TABLES / "chosen_models.json").read_text())
rows = []
for task in ["talent", "performance"]:
    X, y, sens, cat, num = task_frame(df, task)
    name = chosen[task]
    best = json.loads((TABLES / f"best_params_{task}.json").read_text())[name]
    for r in range(10):
        parts = split(X, y, sens, seed=1000 + r)
        (Xtr, ytr, _), (Xca, yca, _), (Xte, yte, ste) = parts["train"], parts["cal"], parts["test"]
        est, scale, _ = model_zoo(float((ytr == 0).sum() / (ytr == 1).sum()))[name]
        pipe = build_pipeline(est, scale, cat, num).set_params(**best).fit(Xtr, ytr)
        pc, pt = pipe.predict_proba(Xca)[:, 1], pipe.predict_proba(Xte)[:, 1]
        yt = yte.to_numpy()
        for mondrian in (False, True):
            for alpha in (0.2, 0.1, 0.05):
                S = conformal_sets(pc, yca.to_numpy(), pt, alpha, mondrian)
                cov = S[np.arange(len(yt)), yt]
                size = S.sum(1)
                row = {"task": task, "model": name, "repeat": r, "alpha": alpha,
                       "method": "Mondrian" if mondrian else "Marginal", "coverage": cov.mean(),
                       "coverage_class0": cov[yt == 0].mean(), "coverage_class1": cov[yt == 1].mean(),
                       "ambiguous": (size == 2).mean(), "empty": (size == 0).mean(),
                       "singleton_acc": cov[size == 1].mean()}
                for g in ["Female", "Male"]:
                    m = (ste["gender"] == g).to_numpy()
                    row[f"cov1_{g}"] = cov[m & (yt == 1)].mean()
                    row[f"defer_{g}"] = (size[m] != 1).mean()
                rows.append(row)
        print(task, "repeat", r, "done")

res = pd.DataFrame(rows)
res.to_csv(TABLES / "conformal_repeats_runs.csv", index=False)
summ = res.drop(columns=["model", "repeat"]).groupby(["task", "method", "alpha"]).agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.to_csv(TABLES / "conformal_repeats_summary.csv")
cols = ["coverage", "coverage_class0", "coverage_class1", "ambiguous", "singleton_acc", "cov1_Female", "cov1_Male"]
print(summ[[f"{c}_mean" for c in cols] + ["coverage_class1_std"]].round(3).to_string())
