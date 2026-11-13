"""Experiment 8: robustness of the nine-box results and bootstrap confidence intervals.

Variants (age-blind ML vs conventional):
  base        layout A (box 8 = above/medium), performance 5 = above, potential tertiles
  perf45      performance 4-5 = above, 3 = as expected
  layoutB     box 8 = as expected/high potential
  quartiles   potential cut at quartiles (low < Q1, medium Q1-Q3, high >= Q3)
For each variant: box 9 and boxes 8-9 precision/recall, paired Wilcoxon over 10 splits, and prioritisation precision
inside the conventional boxes 8-9 pool. For the base variant, 95% bootstrap CIs (1,000 employee resamples per split,
averaged over splits) for the main effects.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone

from common import SEED, TABLES, build_pipeline, load_promotion, model_zoo, split, task_frame

B = 1000
rng = np.random.default_rng(SEED)
VARIANTS = {
    "base": dict(perf=(5, 3), layout="A", cuts=(1 / 3, 2 / 3)),
    "perf45": dict(perf=(4, 3), layout="A", cuts=(1 / 3, 2 / 3)),
    "layoutB": dict(perf=(5, 3), layout="B", cuts=(1 / 3, 2 / 3)),
    "quartiles": dict(perf=(5, 3), layout="A", cuts=(0.25, 0.75)),
}
BOX89 = {"A": {("High", "High"): 9, ("High", "Medium"): 8}, "B": {("High", "High"): 9, ("Medium", "High"): 8}}


def levels(v, ref, cuts):
    q = np.quantile(ref, cuts)
    return np.select([v < q[0], v < q[1]], ["Low", "Medium"], "High")


def dept_z(score, dept, ref_score, ref_dept):
    st = pd.DataFrame({"s": ref_score, "d": ref_dept}).groupby("d")["s"].agg(["mean", "std"])
    return (np.asarray(score) - pd.Series(dept).map(st["mean"]).to_numpy()) / pd.Series(dept).map(st["std"]).to_numpy()


def box_no(perf, pot, layout):
    return np.array([BOX89[layout].get((a, b), 0) for a, b in zip(perf, pot)])


df = load_promotion()
Xp, y, sens, cat_p, num_p = task_frame(df, "potential")
Xt, _, _, cat_t, num_t = task_frame(df, "talent")
extra = sens.assign(rating=df["previous_year_rating"].to_numpy(), score=df["avg_training_score"].to_numpy(),
                    dept=df["department"].to_numpy(), idx=np.arange(len(df)))
bp = {t: json.loads((TABLES / f"best_params_{t}.json").read_text())["CatBoost"] for t in ["potential", "talent"]}
num_p_nb, num_t_nb = [c for c in num_p if c != "age"], [c for c in num_t if c != "age"]

rows, boot = [], {k: [] for k in ["box9_recall_diff", "box9_precision_diff", "box89_recall_diff",
                                  "top10_ml_minus_raw", "top10_ml_minus_dept", "top25_ml_minus_dept"]}
for r in range(10):
    parts = split(Xp, y, extra, seed=1000 + r)
    (Xtr, ytr, etr), (Xca, yca, eca), (Xte, yte, ete) = parts["train"], parts["cal"], parts["test"]
    est = model_zoo(1.0)["CatBoost"][0]
    pot = build_pipeline(clone(est), False, cat_p, num_p_nb).set_params(**bp["potential"]).fit(Xtr, ytr)
    tal = build_pipeline(clone(est), False, cat_t, num_t_nb).set_params(**bp["talent"]).fit(Xt.iloc[etr["idx"]], ytr)
    pot_cal, pot_te = pot.predict_proba(Xca)[:, 1], pot.predict_proba(Xte)[:, 1]
    tal_te = tal.predict_proba(Xt.iloc[ete["idx"]])[:, 1]

    rated = ete["rating"].notna().to_numpy()
    yt = yte.to_numpy()[rated]
    rating, score, dept = ete["rating"].to_numpy()[rated], ete["score"].to_numpy()[rated], ete["dept"].to_numpy()[rated]
    z = dept_z(score, dept, etr["score"], etr["dept"])
    pml, ptal = pot_te[rated], tal_te[rated]

    for name, v in VARIANTS.items():
        hi, mid = v["perf"]
        perf = np.select([rating >= hi, rating >= mid], ["High", "Medium"], "Low")
        conv = box_no(perf, levels(score, etr["score"], v["cuts"]), v["layout"])
        ml = box_no(perf, levels(pml, pot_cal, v["cuts"]), v["layout"])
        row = {"repeat": r, "variant": name}
        for m, bx in [("conv", conv), ("ml", ml)]:
            for st, sel in [("box9", bx == 9), ("box89", bx >= 8)]:
                row[f"{st}_{m}_share"] = sel.mean()
                row[f"{st}_{m}_precision"] = yt[sel].mean()
                row[f"{st}_{m}_recall"] = yt[sel].sum() / yt.sum()
        pool = np.where(conv >= 8)[0]
        for frac in (0.10, 0.25):
            k = max(1, int(round(frac * len(pool))))
            for m, sc in [("raw", score), ("dept", z), ("ml", ptal)]:
                row[f"top{int(frac * 100)}_{m}"] = yt[pool[np.argsort(-sc[pool], kind="stable")[:k]]].mean()
        rows.append(row)

        if name == "base":  # bootstrap over employees within this split
            n = len(yt)
            k10 = max(1, int(round(0.10 * len(pool))))
            k25 = max(1, int(round(0.25 * len(pool))))
            res = {k: [] for k in boot}
            for _ in range(B):
                i = rng.integers(0, n, n)
                yb, cb, mb = yt[i], conv[i], ml[i]
                tot = max(1, yb.sum())
                res["box9_recall_diff"].append((yb[mb == 9].sum() - yb[cb == 9].sum()) / tot)
                res["box9_precision_diff"].append(yb[mb == 9].mean() - yb[cb == 9].mean())
                res["box89_recall_diff"].append((yb[mb >= 8].sum() - yb[cb >= 8].sum()) / tot)
                pb = pool[rng.integers(0, len(pool), len(pool))]
                top = lambda sc, k: yt[pb[np.argsort(-sc[pb], kind="stable")[:k]]].mean()
                t_ml10, t_raw10, t_dept10 = top(ptal, k10), top(score, k10), top(z, k10)
                res["top10_ml_minus_raw"].append(t_ml10 - t_raw10)
                res["top10_ml_minus_dept"].append(t_ml10 - t_dept10)
                res["top25_ml_minus_dept"].append(top(ptal, k25) - top(z, k25))
            for k in boot:
                boot[k].append(res[k])
    print("repeat", r, "done")

R = pd.DataFrame(rows)
R.to_csv(TABLES / "robustness_runs.csv", index=False)
summ = []
for name in VARIANTS:
    s = R[R.variant == name]
    for st in ["box9", "box89"]:
        for metric in ["precision", "recall"]:
            a, c = s[f"{st}_ml_{metric}"], s[f"{st}_conv_{metric}"]
            summ.append({"variant": name, "comparison": f"{st} {metric}: ML age-blind vs conventional",
                         "conventional": c.mean(), "ml": a.mean(), "rel_gain_%": 100 * (a.mean() / c.mean() - 1),
                         "ml_better_in": int((a > c).sum()), "p_wilcoxon": wilcoxon(a, c).pvalue,
                         "share_conv": s[f"{st}_conv_share"].mean(), "share_ml": s[f"{st}_ml_share"].mean()})
    for frac in (10, 25):
        for other in ("raw", "dept"):
            a, c = s[f"top{frac}_ml"], s[f"top{frac}_{other}"]
            summ.append({"variant": name, "comparison": f"top {frac}% of boxes 8-9: ML vs {other}",
                         "conventional": c.mean(), "ml": a.mean(), "rel_gain_%": 100 * (a.mean() / c.mean() - 1),
                         "ml_better_in": int((a > c).sum()), "p_wilcoxon": wilcoxon(a, c).pvalue})
S = pd.DataFrame(summ)
S.to_csv(TABLES / "robustness_summary.csv", index=False)

ci = []
for k, per_split in boot.items():
    arr = np.mean(np.array(per_split), axis=0)  # average over splits for each bootstrap iteration
    ci.append({"effect": k, "mean": arr.mean(), "ci_low": np.percentile(arr, 2.5), "ci_high": np.percentile(arr, 97.5)})
C = pd.DataFrame(ci)
C.to_csv(TABLES / "bootstrap_ci.csv", index=False)
pd.set_option("display.width", 250)
print(S.round(4).to_string())
print(C.round(4).to_string())
