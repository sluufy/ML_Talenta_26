"""Experiment 6: regulation-compliant ML for nine-box talent placement (boxes 8-9) and promotion prioritisation.

Nine-box layout (PermenPANRB 3/2020): performance axis x potential axis, three levels each.
  Box 9 = performance above expectation & high potential; Box 8 = performance above expectation & medium potential.
Performance axis (observed appraisal, never predicted): rating 5 = above expectation, 3-4 = as expected, 1-2 = below.

Stage 1 - placement. Potential axis from (a) conventional: tertiles of the assessment proxy (average training score);
          (b) department-normalised: tertiles of the training score standardised within department;
          (c) ML-assisted: tertiles of a CatBoost potential model trained without the performance rating.
Stage 2 - prioritisation inside boxes 8-9 for limited vacancies: rank by training score, department-normalised score,
          logistic regression, or the CatBoost talent model; precision/recall at the top k%.
Stage 3 - accountability: Mondrian conformal deferral inside boxes 8-9 and fairness of box membership.
Validation target: actual promotion (held-out test sets, 10 repeated splits, paired Wilcoxon tests with Holm correction).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score

from common import SEED, TABLES, build_pipeline, load_promotion, model_zoo, split, task_frame
from run_trust import conformal_sets

LEVELS = ["Low", "Medium", "High"]
# box number by (performance, potential) - PermenPANRB 3/2020 layout
BOX_NO = {("Low", "Low"): 1, ("Low", "Medium"): 2, ("Medium", "Low"): 3, ("Low", "High"): 4, ("Medium", "Medium"): 5,
          ("High", "Low"): 6, ("Medium", "High"): 7, ("High", "Medium"): 8, ("High", "High"): 9}
FRACS = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def tertile(v, ref):
    q = np.quantile(ref, [1 / 3, 2 / 3])
    return np.select([v < q[0], v < q[1]], ["Low", "Medium"], "High")


def dept_z(score, dept, ref_score, ref_dept):
    stats = pd.DataFrame({"s": ref_score, "d": ref_dept}).groupby("d")["s"].agg(["mean", "std"])
    m = pd.Series(dept).map(stats["mean"]).to_numpy()
    s = pd.Series(dept).map(stats["std"]).to_numpy()
    return (np.asarray(score) - m) / s


def holm(p):
    keys = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


df = load_promotion()
Xp, y, sens, cat_p, num_p = task_frame(df, "potential")
Xt, _, _, cat_t, num_t = task_frame(df, "talent")
extra = sens.assign(rating=df["previous_year_rating"].to_numpy(), score=df["avg_training_score"].to_numpy(),
                    dept=df["department"].to_numpy(), idx=np.arange(len(df)))
bp = {t: json.loads((TABLES / f"best_params_{t}.json").read_text()) for t in ["potential", "talent"]}

place, prio, defer, fair, cells, agree = [], [], [], [], [], []
for r in range(10):
    parts = split(Xp, y, extra, seed=1000 + r)
    (Xtr, ytr, etr), (Xca, yca, eca), (Xte, yte, ete) = parts["train"], parts["cal"], parts["test"]
    zoo = model_zoo(float((ytr == 0).sum() / (ytr == 1).sum()))
    pot = build_pipeline(clone(zoo["CatBoost"][0]), False, cat_p, num_p).set_params(**bp["potential"]["CatBoost"]).fit(Xtr, ytr)
    tal = build_pipeline(clone(zoo["CatBoost"][0]), False, cat_t, num_t).set_params(**bp["talent"]["CatBoost"]).fit(
        Xt.iloc[etr["idx"]], ytr)
    lr = build_pipeline(clone(zoo["LR"][0]), True, cat_t, num_t).set_params(**bp["talent"]["LR"]).fit(Xt.iloc[etr["idx"]], ytr)
    # age-blind variants (fairness ablation): age removed from the feature set
    pot_nb = build_pipeline(clone(zoo["CatBoost"][0]), False, cat_p, [c for c in num_p if c != "age"]).set_params(
        **bp["potential"]["CatBoost"]).fit(Xtr, ytr)
    tal_nb = build_pipeline(clone(zoo["CatBoost"][0]), False, cat_t, [c for c in num_t if c != "age"]).set_params(
        **bp["talent"]["CatBoost"]).fit(Xt.iloc[etr["idx"]], ytr)

    pot_cal, pot_te = pot.predict_proba(Xca)[:, 1], pot.predict_proba(Xte)[:, 1]
    tal_cal = tal.predict_proba(Xt.iloc[eca["idx"]])[:, 1]
    tal_te = tal.predict_proba(Xt.iloc[ete["idx"]])[:, 1]
    lr_te = lr.predict_proba(Xt.iloc[ete["idx"]])[:, 1]
    potnb_cal, potnb_te = pot_nb.predict_proba(Xca)[:, 1], pot_nb.predict_proba(Xte)[:, 1]
    talnb_te = tal_nb.predict_proba(Xt.iloc[ete["idx"]])[:, 1]
    S = conformal_sets(tal_cal, yca.to_numpy(), tal_te, alpha=0.10, mondrian=True)

    rated = ete["rating"].notna().to_numpy()
    yt = yte.to_numpy()[rated]
    rating = ete["rating"].to_numpy()[rated]
    perf = np.select([rating >= 5, rating >= 3], ["High", "Medium"], "Low")
    score, dept = ete["score"].to_numpy()[rated], ete["dept"].to_numpy()[rated]
    z = dept_z(score, dept, etr["score"], etr["dept"])
    z_ref = dept_z(etr["score"], etr["dept"], etr["score"], etr["dept"])
    axes = {"Conventional": tertile(score, etr["score"]),
            "Dept-normalised": tertile(z, z_ref),
            "ML-assisted": tertile(pot_te[rated], pot_cal),
            "ML-assisted (age-blind)": tertile(potnb_te[rated], potnb_cal)}
    g = ete["gender"].to_numpy()[rated]
    a = ete["age_group"].to_numpy()[rated]
    boxes = {}
    for m, pa in axes.items():
        box = np.array([BOX_NO[(p, q)] for p, q in zip(perf, pa)])
        boxes[m] = box
        for name, sel in [("Box 9", box == 9), ("Boxes 8-9", box >= 8)]:
            tp_ = yt[sel].sum()
            place.append({"repeat": r, "method": m, "set": name, "share": sel.mean(), "precision": yt[sel].mean(),
                          "recall": tp_ / yt.sum(), "lift": yt[sel].mean() / yt.mean()})
            for attr, vals in [("gender", g), ("age_group", a)]:
                rates = {v: sel[vals == v].mean() for v in np.unique(vals)}
                tprs = {v: sel[(vals == v) & (yt == 1)].mean() for v in np.unique(vals)}
                fair.append({"repeat": r, "method": m, "set": name, "attribute": attr,
                             "selection_rate_diff": max(rates.values()) - min(rates.values()),
                             "tpr_diff": max(tprs.values()) - min(tprs.values()),
                             **{f"rate_{k}": v for k, v in rates.items()}})
        for b in range(1, 10):
            sel = box == b
            cells.append({"repeat": r, "method": m, "box": b, "n": sel.sum(), "promotion_rate": yt[sel].mean()})
    for m in ["Dept-normalised", "ML-assisted", "ML-assisted (age-blind)"]:
        agree.append({"repeat": r, "method": m, "kappa_vs_conventional": cohen_kappa_score(boxes["Conventional"], boxes[m]),
                      "share_changed_box": (boxes["Conventional"] != boxes[m]).mean(),
                      "share_changed_89_membership": ((boxes["Conventional"] >= 8) != (boxes[m] >= 8)).mean()})

    # Stage 2: prioritisation inside the (conventional) boxes 8-9 pool
    pool = np.where(boxes["Conventional"] >= 8)[0]
    rankers = {"Training score": score, "Dept-normalised score": z, "Logistic regression": lr_te[rated],
               "CatBoost (ML)": tal_te[rated], "CatBoost (age-blind)": talnb_te[rated], "Random": np.random.default_rng(SEED + r).random(len(score))}
    for frac in FRACS:
        k = max(1, int(round(frac * len(pool))))
        for name, sc in rankers.items():
            top = pool[np.argsort(-sc[pool], kind="stable")[:k]]
            prio.append({"repeat": r, "ranker": name, "top_frac": frac, "k": k, "precision": yt[top].mean(),
                         "recall_in_pool": yt[top].sum() / max(1, yt[pool].sum()),
                         "female_share": (g[top] == "Female").mean(), "age40_share": (a[top] == ">=40").mean()})

    # Stage 3: conformal deferral inside ML-assisted boxes 8-9
    Sr = S[rated]
    in89 = boxes["ML-assisted"] >= 8
    ready = in89 & Sr[:, 1] & ~Sr[:, 0]
    notready = in89 & Sr[:, 0] & ~Sr[:, 1]
    amb = in89 & Sr[:, 0] & Sr[:, 1]
    defer.append({"repeat": r, "pool": in89.sum(), "confident_ready": ready.sum(), "confident_not_ready": notready.sum(),
                  "deferred": amb.sum(), "share_deferred": amb.sum() / in89.sum(),
                  "precision_confident_ready": yt[ready].mean() if ready.any() else np.nan,
                  "promotion_rate_not_ready": yt[notready].mean() if notready.any() else np.nan,
                  "promotion_rate_deferred": yt[amb].mean() if amb.any() else np.nan,
                  "share_promoted_caught_ready_or_deferred": yt[ready | amb].sum() / max(1, yt[in89].sum())})
    print("repeat", r, "done")

P = pd.DataFrame(place); P.to_csv(TABLES / "box_placement_runs.csv", index=False)
Q = pd.DataFrame(prio); Q.to_csv(TABLES / "box_prioritisation_runs.csv", index=False)
D = pd.DataFrame(defer); D.to_csv(TABLES / "box_deferral_runs.csv", index=False)
F = pd.DataFrame(fair); F.to_csv(TABLES / "box_fairness_runs.csv", index=False)
C = pd.DataFrame(cells); C.to_csv(TABLES / "box_cells_runs.csv", index=False)
A = pd.DataFrame(agree); A.to_csv(TABLES / "box_agreement_runs.csv", index=False)

pd.set_option("display.width", 250)
ps = P.groupby(["set", "method"])[["share", "precision", "recall", "lift"]].agg(["mean", "std"]); ps.columns = [f"{x}_{y}" for x, y in ps.columns]
ps.to_csv(TABLES / "box_placement_summary.csv"); print(ps.round(3).to_string())

tests = []
for st in ["Box 9", "Boxes 8-9"]:
    for metric in ["precision", "recall"]:
        w = P[P.set == st].pivot(index="repeat", columns="method", values=metric)
        for ml in ["ML-assisted", "ML-assisted (age-blind)"]:
            raw = {m: wilcoxon(w[ml], w[m]).pvalue for m in ["Conventional", "Dept-normalised"]}
            for m, padj in holm(raw).items():
                tests.append({"comparison": f"{ml} vs {m}", "set": st, "metric": metric,
                              "mean_ml": w[ml].mean(), "mean_other": w[m].mean(),
                              "rel_gain_%": 100 * (w[ml].mean() / w[m].mean() - 1),
                              "ml_better_in": int((w[ml] > w[m]).sum()), "p_raw": raw[m], "p_holm": padj})
qs = Q.groupby(["top_frac", "ranker"])[["precision", "recall_in_pool", "female_share", "age40_share"]].agg(["mean", "std"])
qs.columns = [f"{x}_{y}" for x, y in qs.columns]; qs.to_csv(TABLES / "box_prioritisation_summary.csv")
for frac in [0.10, 0.25, 0.50]:
    w = Q[Q.top_frac == frac].pivot(index="repeat", columns="ranker", values="precision")
    for ml in ["CatBoost (ML)", "CatBoost (age-blind)"]:
        raw = {m: wilcoxon(w[ml], w[m]).pvalue for m in ["Training score", "Dept-normalised score", "Logistic regression"]}
        for m, padj in holm(raw).items():
            tests.append({"comparison": f"{ml} vs {m}", "set": f"Top {int(frac * 100)}% of boxes 8-9", "metric": "precision",
                          "mean_ml": w[ml].mean(), "mean_other": w[m].mean(),
                          "rel_gain_%": 100 * (w[ml].mean() / w[m].mean() - 1),
                          "ml_better_in": int((w[ml] > w[m]).sum()), "p_raw": raw[m], "p_holm": padj})
T = pd.DataFrame(tests); T.to_csv(TABLES / "box_tests.csv", index=False)
print(T.round(4).to_string())
print(qs.loc[[0.1, 0.25, 0.5]][["precision_mean", "precision_std", "recall_in_pool_mean", "female_share_mean", "age40_share_mean"]].round(3).to_string())
print(D.mean(numeric_only=True).round(3).to_string())
fs = F.groupby(["set", "method", "attribute"])[["selection_rate_diff", "tpr_diff"]].agg(["mean", "std"])
fs.columns = [f"{x}_{y}" for x, y in fs.columns]; fs.to_csv(TABLES / "box_fairness_summary.csv"); print(fs.round(3).to_string())
cs = C.groupby(["method", "box"])[["n", "promotion_rate"]].mean(); cs.to_csv(TABLES / "box_cells_summary.csv")
print(A.groupby("method").mean(numeric_only=True).round(3).to_string())
