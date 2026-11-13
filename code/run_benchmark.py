"""Experiment 1: tuned model benchmark over repeated stratified splits + statistical tests.

Usage: python run_benchmark.py [--repeats 10] [--n-iter 12] [--tasks talent performance potential]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from common import (SEED, TABLES, best_f1_threshold, build_pipeline, dev_holdout, load_promotion, model_zoo,
                    scores, split, task_frame)


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=pvals.get)
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def tune(task, X, y, sens, cat, num, n_iter):
    (Xtr, ytr, _), _ = dev_holdout(X, y, sens)
    zoo = model_zoo(float((ytr == 0).sum() / (ytr == 1).sum()))
    best = {}
    for name, (est, scale, space) in zoo.items():
        t0 = time.time()
        search = RandomizedSearchCV(build_pipeline(est, scale, cat, num), space, n_iter=n_iter,
                                    scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                    random_state=SEED, n_jobs=1, refit=False)
        search.fit(Xtr, ytr)
        best[name] = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        print(f"  [{task}] tuned {name:12s} cv PR-AUC={search.best_score_:.4f} ({time.time() - t0:.0f}s) {best[name]}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--tasks", nargs="+", default=["talent", "performance", "potential"])
    args = ap.parse_args()

    df = load_promotion()
    print(f"Loaded {len(df)} employees; promotion rate={df['is_promoted'].mean():.4f}")
    for task in args.tasks:
        X, y, sens, cat, num = task_frame(df, task)
        print(f"\n=== Task {task}: n={len(y)}, positive rate={y.mean():.4f}, features={cat + num}")
        best = tune(task, X, y, sens, cat, num, args.n_iter)
        (TABLES / f"best_params_{task}.json").write_text(json.dumps(best, indent=2))

        rows = []
        for r in range(args.repeats):
            parts = split(X, y, sens, seed=1000 + r)
            Xtr, ytr, _ = parts["train"]
            Xva, yva, _ = parts["val"]
            Xte, yte, _ = parts["test"]
            zoo = model_zoo(float((ytr == 0).sum() / (ytr == 1).sum()))
            for name, (est, scale, _) in zoo.items():
                pipe = build_pipeline(clone(est), scale, cat, num).set_params(**best[name])
                t0 = time.time()
                pipe.fit(Xtr, ytr)
                fit_s = time.time() - t0
                thr = best_f1_threshold(yva, pipe.predict_proba(Xva)[:, 1])
                t0 = time.perf_counter()
                pte = pipe.predict_proba(Xte)[:, 1]  # batch inference incl. preprocessing
                latency_ms = (time.perf_counter() - t0) / len(Xte) * 1000 * 1000  # ms per 1,000 employees
                rows.append({"task": task, "repeat": r, "model": name, "fit_seconds": fit_s,
                             "latency_ms_per_1k": latency_ms, **scores(yte, pte, thr)})
            print(f"  repeat {r + 1}/{args.repeats} done")
        res = pd.DataFrame(rows)
        res.to_csv(TABLES / f"benchmark_runs_{task}.csv", index=False)

        metrics = ["ROC-AUC", "PR-AUC", "Accuracy", "MacroF1", "F1", "Precision", "Recall", "MCC", "BalAcc", "Brier",
                   "ECE", "fit_seconds", "latency_ms_per_1k"]
        summary = res.groupby("model")[metrics].agg(["mean", "std"])
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary = summary.sort_values("PR-AUC_mean", ascending=False)
        summary.to_csv(TABLES / f"benchmark_summary_{task}.csv")
        print(summary[[c for c in summary.columns if c.endswith("_mean")]].round(4).to_string())

        # Friedman test across models + Wilcoxon signed-rank (Holm) best-vs-rest, per metric
        stats_rows = []
        for metric in ["PR-AUC", "ROC-AUC", "MCC", "F1"]:
            wide = res.pivot(index="repeat", columns="model", values=metric)
            fr = friedmanchisquare(*[wide[c] for c in wide.columns])
            top = wide.mean().idxmax()
            raw = {c: wilcoxon(wide[top], wide[c]).pvalue for c in wide.columns if c != top}
            adj = holm(raw)
            for c in raw:
                stats_rows.append({"task": task, "metric": metric, "best": top, "vs": c,
                                   "mean_diff": wide[top].mean() - wide[c].mean(), "p_raw": raw[c], "p_holm": adj[c],
                                   "friedman_chi2": fr.statistic, "friedman_p": fr.pvalue,
                                   "avg_rank_best": wide.rank(axis=1, ascending=False).mean()[top]})
        pd.DataFrame(stats_rows).to_csv(TABLES / f"stats_tests_{task}.csv", index=False)
        ranks = {m: res.pivot(index="repeat", columns="model", values=m).rank(axis=1, ascending=False).mean()
                 for m in ["PR-AUC", "MCC"]}
        pd.DataFrame(ranks).to_csv(TABLES / f"avg_ranks_{task}.csv")


if __name__ == "__main__":
    main()
