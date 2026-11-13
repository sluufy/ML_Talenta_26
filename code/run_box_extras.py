"""Age-blind models: discrimination over 10 splits and Mondrian conformal deferral inside age-blind boxes 8-9."""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from common import TABLES, build_pipeline, load_promotion, model_zoo, split, task_frame
from run_trust import conformal_sets

df = load_promotion()
Xp, y, sens, cat_p, num_p = task_frame(df, "potential")
Xt, _, _, cat_t, num_t = task_frame(df, "talent")
extra = sens.assign(rating=df["previous_year_rating"].to_numpy(), idx=np.arange(len(df)))
# (performance, potential) -> box, Annex of PermenPANRB 20/2025
BOX_NO = {("Low", "Low"): 1, ("Medium", "Low"): 2, ("Low", "Medium"): 3, ("High", "Low"): 4, ("Medium", "Medium"): 5,
          ("Low", "High"): 6, ("High", "Medium"): 7, ("Medium", "High"): 8, ("High", "High"): 9}


def tertile(v, ref):
    q = np.quantile(ref, [1 / 3, 2 / 3])
    return np.select([v < q[0], v < q[1]], ["Low", "Medium"], "High")


bp = {t: json.loads((TABLES / f"best_params_{t}.json").read_text())["CatBoost"] for t in ["potential", "talent"]}
num_p_nb, num_t_nb = [c for c in num_p if c != "age"], [c for c in num_t if c != "age"]

perf_rows, defer_rows = [], []
for r in range(10):
    parts = split(Xp, y, extra, seed=1000 + r)
    (Xtr, ytr, etr), (Xca, yca, eca), (Xte, yte, ete) = parts["train"], parts["cal"], parts["test"]
    est = model_zoo(1.0)["CatBoost"][0]
    pot = build_pipeline(clone(est), False, cat_p, num_p_nb).set_params(**bp["potential"]).fit(Xtr, ytr)
    tal = build_pipeline(clone(est), False, cat_t, num_t_nb).set_params(**bp["talent"]).fit(Xt.iloc[etr["idx"]], ytr)
    p_pot_cal, p_pot = pot.predict_proba(Xca)[:, 1], pot.predict_proba(Xte)[:, 1]
    p_tal_cal, p_tal = tal.predict_proba(Xt.iloc[eca["idx"]])[:, 1], tal.predict_proba(Xt.iloc[ete["idx"]])[:, 1]
    yt_all = yte.to_numpy()
    perf_rows += [{"repeat": r, "model": "Potential (age-blind)", "ROC-AUC": roc_auc_score(yt_all, p_pot),
                   "PR-AUC": average_precision_score(yt_all, p_pot)},
                  {"repeat": r, "model": "Talent (age-blind)", "ROC-AUC": roc_auc_score(yt_all, p_tal),
                   "PR-AUC": average_precision_score(yt_all, p_tal)}]
    rated = ete["rating"].notna().to_numpy()
    yt, rating = yt_all[rated], ete["rating"].to_numpy()[rated]
    perf = np.select([rating >= 5, rating >= 3], ["High", "Medium"], "Low")
    box = np.array([BOX_NO[(a, b)] for a, b in zip(perf, tertile(p_pot[rated], p_pot_cal))])
    in89 = box >= 8
    g = ete["gender"].to_numpy()[rated]
    for alpha in (0.20, 0.10):
        S = conformal_sets(p_tal_cal, yca.to_numpy(), p_tal, alpha=alpha, mondrian=True)[rated]
        ready, amb = in89 & S[:, 1] & ~S[:, 0], in89 & S[:, 0] & S[:, 1]
        notready = in89 & S[:, 0] & ~S[:, 1]
        defer_rows.append({"repeat": r, "alpha": alpha, "pool": in89.sum(), "share_ready": ready.sum() / in89.sum(),
                           "share_deferred": amb.sum() / in89.sum(), "share_not_ready": notready.sum() / in89.sum(),
                           "promo_rate_ready": yt[ready].mean(), "promo_rate_deferred": yt[amb].mean() if amb.any() else np.nan,
                           "promo_rate_pool": yt[in89].mean(),
                           "promoted_missed": int(yt[notready].sum()),
                           "deferred_female": amb[g == "Female"].sum() / max(1, in89[g == "Female"].sum()),
                           "deferred_male": amb[g == "Male"].sum() / max(1, in89[g == "Male"].sum())})
    print("repeat", r, "done")

P = pd.DataFrame(perf_rows); P.to_csv(TABLES / "agebind_models_runs.csv", index=False)
D = pd.DataFrame(defer_rows); D.to_csv(TABLES / "box_deferral_ageblind_runs.csv", index=False)
print(P.groupby("model")[["ROC-AUC", "PR-AUC"]].agg(["mean", "std"]).round(3).to_string())
print(D.drop(columns="repeat").groupby("alpha").agg(["mean", "std"]).round(3).T.to_string())
