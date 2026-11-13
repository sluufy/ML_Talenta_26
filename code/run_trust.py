"""Experiments 2-5 on the reference split: calibration, SHAP explanations, fairness audit + mitigation,
Mondrian conformal prediction, and the ML-assisted 9-box talent grid.

Run after run_benchmark.py (it reads the tuned hyper-parameters and the benchmark ranking).
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from fairlearn.metrics import (MetricFrame, count, demographic_parity_difference, equalized_odds_difference,
                               false_positive_rate, selection_rate, true_positive_rate)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score

from common import (FIGURES, SEED, TABLES, best_f1_threshold, build_pipeline, load_promotion, model_zoo, scores,
                    split, task_frame)

# Validated categorical palette (light mode, print) + text inks
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 2})

TREE_MODELS = ["LightGBM", "XGBoost", "CatBoost", "RF"]
NICE = {"avg_training_score": "Avg. training score", "previous_year_rating": "Previous-year rating",
        "kpis_met": "KPIs met >80%", "awards_won": "Awards won", "length_of_service": "Length of service",
        "no_of_trainings": "No. of trainings", "age": "Age"}


def nice(name: str) -> str:
    if name in NICE:
        return NICE[name]
    if name.startswith("missingindicator_"):
        return NICE.get(name[17:], name[17:]) + " (missing)"
    for p in ["department_", "region_", "education_", "recruitment_channel_"]:
        if name.startswith(p):
            return f"{p[:-1].replace('_', ' ').capitalize()} = {name[len(p):]}"
    return name


def chosen_model(task: str) -> str:
    summ = pd.read_csv(TABLES / f"benchmark_summary_{task}.csv", index_col=0)
    return summ.loc[summ.index.isin(TREE_MODELS), "PR-AUC_mean"].idxmax()


def fit_on_reference_split(task: str, df: pd.DataFrame, name: str | None = None):
    X, y, sens, cat, num = task_frame(df, task)
    parts = split(X, y, sens, SEED)
    Xtr, ytr, _ = parts["train"]
    name = name or chosen_model(task)
    best = json.loads((TABLES / f"best_params_{task}.json").read_text())
    est, scale, _ = model_zoo(float((ytr == 0).sum() / (ytr == 1).sum()))[name]
    pipe = build_pipeline(est, scale, cat, num).set_params(**best[name]).fit(Xtr, ytr)
    Xva, yva, _ = parts["val"]
    thr = best_f1_threshold(yva, pipe.predict_proba(Xva)[:, 1])
    return name, pipe, thr, parts


# ---------------------------------------------------------------- calibration

def calibration_plot(task, df):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, label="Perfect calibration")
    rows = []
    for color, name in zip([C1, C2, C3], [chosen_model(task), "LR", "LightGBM-CW"]):
        _, pipe, thr, parts = fit_on_reference_split(task, df, name)
        Xte, yte, _ = parts["test"]
        p = pipe.predict_proba(Xte)[:, 1]
        fy, fx = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        ax.plot(fx, fy, marker="o", ms=4, color=color, label=name)
        rows.append({"model": name, **scores(yte, p, thr)})
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / f"fig_calibration_{task}.png")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(TABLES / f"calibration_{task}.csv", index=False)


# ---------------------------------------------------------------- convergence, confusion matrix, errors

def convergence_plot(task, df):
    """Training vs validation log-loss per boosting iteration for the chosen model
    (LightGBM is used as the boosting reference when the chosen model is not a booster)."""
    name, pipe, _, parts = fit_on_reference_split(task, df)
    if name not in ("LightGBM", "XGBoost", "CatBoost"):
        name, pipe, _, parts = fit_on_reference_split(task, df, "LightGBM")
    pre, clf = pipe.named_steps["pre"], clone(pipe.named_steps["clf"])
    (Xtr, ytr, _), (Xva, yva, _) = parts["train"], parts["val"]
    A, B = pre.transform(Xtr), pre.transform(Xva)
    if name == "LightGBM":
        clf.fit(A, ytr, eval_set=[(A, ytr), (B, yva)], eval_metric="binary_logloss")
        r = clf.evals_result_
        tr, va = r["training"]["binary_logloss"], r["valid_1"]["binary_logloss"]
    elif name == "XGBoost":
        clf.fit(A, ytr, eval_set=[(A, ytr), (B, yva)], verbose=False)
        r = clf.evals_result()
        tr, va = r["validation_0"]["logloss"], r["validation_1"]["logloss"]
    else:
        clf.fit(A, ytr, eval_set=(B, yva))
        r = clf.get_evals_result()
        tr, va = r["learn"]["Logloss"], r["validation"]["Logloss"]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(range(1, len(tr) + 1), tr, color=C1, label="Training")
    ax.plot(range(1, len(va) + 1), va, color=C2, label="Validation")
    best = int(np.argmin(va)) + 1
    ax.axvline(best, color=INK2, ls="--", lw=1)
    ax.annotate(f"min val. loss @ {best}", (best, max(va)), fontsize=7, color=INK2, ha="left", va="top",
                xytext=(4, 0), textcoords="offset points")
    ax.set(xlabel="Boosting iteration", ylabel="Log-loss (binary cross-entropy)", title=f"{task} – {name}")
    ax.title.set_fontsize(9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / f"fig_convergence_{task}.png")
    plt.close(fig)
    pd.DataFrame({"iteration": range(1, len(tr) + 1), "train_logloss": tr, "val_logloss": va}).to_csv(
        TABLES / f"convergence_{task}.csv", index=False)


def confusion_and_errors(task, df):
    name, pipe, thr, parts = fit_on_reference_split(task, df)
    Xca, yca, _ = parts["cal"]
    Xte, yte, ste = parts["test"]
    p = pipe.predict_proba(Xte)[:, 1]
    yhat = (p >= thr).astype(int)
    yt = yte.to_numpy()

    cm = confusion_matrix(yt, yhat)
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    labels = ["Not promoted", "Promoted"] if task == "talent" else ["Not high", "High performer"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels, rotation=90, va="center")
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"{task} – {name} (threshold = {thr:.2f})")
    ax.title.set_fontsize(9)
    ax.grid(False)
    fig.savefig(FIGURES / f"fig_confusion_{task}.png")
    plt.close(fig)
    pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
        TABLES / f"confusion_{task}.csv")

    # Error profile: who are the false positives / false negatives?
    S = conformal_sets(pipe.predict_proba(Xca)[:, 1], yca.to_numpy(), p, alpha=0.10, mondrian=True)
    outcome = np.select([(yt == 1) & (yhat == 1), (yt == 0) & (yhat == 1), (yt == 1) & (yhat == 0)],
                        ["TP", "FP", "FN"], "TN")
    prof = Xte.copy()
    prof["outcome"] = outcome
    prof["p_hat"] = p
    prof["near_threshold(|p-thr|<0.1)"] = np.abs(p - thr) < 0.1
    prof["deferred_by_conformal"] = S.sum(1) != 1
    prof["gender_female"] = (ste["gender"] == "Female").to_numpy()
    num_cols = [c for c in prof.columns if pd.api.types.is_numeric_dtype(prof[c]) and c != "outcome"]
    agg = prof.groupby("outcome")[num_cols].mean()
    agg.insert(0, "n", prof.groupby("outcome").size())
    if "previous_year_rating" in Xte:
        agg["rating_missing"] = prof.groupby("outcome")["previous_year_rating"].apply(lambda s: s.isna().mean())
    agg.T.to_csv(TABLES / f"error_profile_{task}.csv")
    top_dept = (prof.groupby("outcome")["department"].agg(lambda s: s.value_counts(normalize=True).head(3).round(3)
                                                          .to_dict()))
    top_dept.to_csv(TABLES / f"error_departments_{task}.csv")


# ---------------------------------------------------------------- SHAP

def shap_analysis(task, df):
    name, pipe, _, parts = fit_on_reference_split(task, df)
    Xte, _, _ = parts["test"]
    Xs = Xte.sample(n=min(3000, len(Xte)), random_state=SEED)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pd.DataFrame(pre.transform(Xs), columns=[nice(c) for c in pre.get_feature_names_out()])
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = pd.Series(np.abs(sv).mean(0), index=Xt.columns).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(TABLES / f"shap_importance_{task}.csv")

    plt.figure()
    shap.summary_plot(sv, Xt, max_display=12, show=False, plot_size=(6, 4.2), cmap="coolwarm")
    plt.title(f"SHAP summary – {task} ({name})", fontsize=9, color=INK)
    plt.savefig(FIGURES / f"fig_shap_beeswarm_{task}.png")
    plt.close("all")

    top = [c for c in imp.index if not c.endswith("(missing)")][:2]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for ax, feat in zip(axes, top):
        j = list(Xt.columns).index(feat)
        ax.scatter(Xt[feat], sv[:, j], s=8, alpha=0.35, color=C1, edgecolors="none")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set(xlabel=feat, ylabel="SHAP value (log-odds)")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig_shap_dependence_{task}.png")
    plt.close(fig)
    return name


# ---------------------------------------------------------------- fairness

def group_report(y, yhat, sf):
    mf = MetricFrame(metrics={"n": count, "selection_rate": selection_rate, "TPR": true_positive_rate,
                              "FPR": false_positive_rate,
                              "precision": lambda a, b: precision_score(a, b, zero_division=0)},
                     y_true=y, y_pred=yhat, sensitive_features=sf)
    return mf.by_group


def fairness_analysis(task, df):
    name, pipe, thr, parts = fit_on_reference_split(task, df)
    Xva, yva, sva = parts["val"]
    Xte, yte, ste = parts["test"]
    p = pipe.predict_proba(Xte)[:, 1]
    yhat = (p >= thr).astype(int)
    by_group, summary = [], []
    for attr in ["gender", "age_group"]:
        g = group_report(yte, yhat, ste[attr]).reset_index().rename(columns={attr: "group"})
        g.insert(0, "attribute", attr)
        g.insert(0, "method", "Unmitigated")
        by_group.append(g)
        summary.append({"method": "Unmitigated", "attribute": attr, "F1": f1_score(yte, yhat),
                        "BalAcc": balanced_accuracy_score(yte, yhat),
                        "DP_diff": demographic_parity_difference(yte, yhat, sensitive_features=ste[attr]),
                        "EOdds_diff": equalized_odds_difference(yte, yhat, sensitive_features=ste[attr])})
        for constraint in ["equalized_odds", "true_positive_rate_parity", "demographic_parity"]:
            to = ThresholdOptimizer(estimator=pipe, constraints=constraint, objective="balanced_accuracy_score",
                                    prefit=True, predict_method="predict_proba")
            to.fit(Xva, yva, sensitive_features=sva[attr])
            ym = to.predict(Xte, sensitive_features=ste[attr], random_state=SEED)
            label = f"ThresholdOptimizer({constraint})"
            g = group_report(yte, ym, ste[attr]).reset_index().rename(columns={attr: "group"})
            g.insert(0, "attribute", attr)
            g.insert(0, "method", label)
            by_group.append(g)
            summary.append({"method": label, "attribute": attr, "F1": f1_score(yte, ym),
                            "BalAcc": balanced_accuracy_score(yte, ym),
                            "DP_diff": demographic_parity_difference(yte, ym, sensitive_features=ste[attr]),
                            "EOdds_diff": equalized_odds_difference(yte, ym, sensitive_features=ste[attr])})
    pd.concat(by_group).to_csv(TABLES / f"fairness_by_group_{task}.csv", index=False)
    pd.DataFrame(summary).to_csv(TABLES / f"fairness_summary_{task}.csv", index=False)

    # Observed base rates (is the disparity in the labels or in the model?)
    base = pd.DataFrame({"y": yte, "yhat": yhat, "gender": ste["gender"], "age_group": ste["age_group"]})
    rates = pd.concat([base.groupby(a)[["y", "yhat"]].mean().assign(attribute=a) for a in ["gender", "age_group"]])
    rates.rename(columns={"y": "observed_rate", "yhat": "predicted_rate"}).to_csv(TABLES / f"base_rates_{task}.csv")


# ---------------------------------------------------------------- conformal

def conformal_sets(p_cal, y_cal, p_test, alpha, mondrian=True):
    """Split conformal for binary classification with score s = 1 - p_hat(y|x).
    Mondrian = class-conditional quantiles -> coverage guaranteed within each class."""
    P_cal = np.column_stack([1 - p_cal, p_cal])
    P_te = np.column_stack([1 - p_test, p_test])
    s = 1 - P_cal[np.arange(len(y_cal)), y_cal]
    q = {}
    for c in (0, 1):
        sc = s[y_cal == c] if mondrian else s
        n = len(sc)
        q[c] = np.quantile(sc, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    return np.column_stack([1 - P_te[:, c] <= q[c] for c in (0, 1)])


def conformal_analysis(task, df):
    name, pipe, thr, parts = fit_on_reference_split(task, df)
    Xca, yca, _ = parts["cal"]
    Xte, yte, ste = parts["test"]
    pc, pt = pipe.predict_proba(Xca)[:, 1], pipe.predict_proba(Xte)[:, 1]
    yt = yte.to_numpy()
    rows = []
    for mondrian in (False, True):
        for alpha in (0.20, 0.10, 0.05):
            S = conformal_sets(pc, yca.to_numpy(), pt, alpha, mondrian)
            cover = S[np.arange(len(yt)), yt]
            size = S.sum(1)
            single = size == 1
            row = {"method": "Mondrian (class-conditional)" if mondrian else "Marginal split",
                   "alpha": alpha, "target_coverage": 1 - alpha, "coverage": cover.mean(),
                   "coverage_class0": cover[yt == 0].mean(), "coverage_class1": cover[yt == 1].mean(),
                   "share_singleton": single.mean(), "share_ambiguous_{0,1}": (size == 2).mean(),
                   "share_empty": (size == 0).mean(), "singleton_accuracy": cover[single].mean(),
                   "avg_set_size": size.mean()}
            for attr in ["gender", "age_group"]:
                for g in sorted(ste[attr].unique()):
                    m = (ste[attr] == g).to_numpy()
                    row[f"cov_{attr}={g}"] = cover[m].mean()
                    row[f"cov1_{attr}={g}"] = cover[m & (yt == 1)].mean()
                    row[f"defer_{attr}={g}"] = (size[m] == 2).mean()
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / f"conformal_{task}.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.6, 3))
    x = np.arange(3)
    w = 0.38
    for k, (meth, color) in enumerate([("Marginal split", C2), ("Mondrian (class-conditional)", C1)]):
        sub = res[res["method"] == meth].sort_values("alpha", ascending=False)
        bars = ax.bar(x + (k - 0.5) * w, sub["coverage_class1"], width=w - 0.03, color=color, label=meth)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK2, padding=2)
    ax.grid(axis="x", visible=False)
    for xi, t in zip(x, [0.8, 0.9, 0.95]):
        ax.hlines(t, xi - 0.42, xi + 0.42, colors=INK, linestyles="--", lw=1)
    ax.set_xticks(x, ["α = 0.20", "α = 0.10", "α = 0.05"])
    ax.set(ylabel="Coverage on positive class", ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(FIGURES / f"fig_conformal_coverage_{task}.png")
    plt.close(fig)
    return res


# ---------------------------------------------------------------- 9-box

def nine_box(df):
    """Performance axis = observed previous-year rating (SKP analogue); potential axis = conformal-aware
    predicted promotion readiness from the 'potential' model (performance indicators excluded)."""
    name, pipe, _, parts = fit_on_reference_split("potential", df)
    Xca, yca, _ = parts["cal"]
    Xte, yte, _ = parts["test"]
    pc, pt = pipe.predict_proba(Xca)[:, 1], pipe.predict_proba(Xte)[:, 1]
    S = conformal_sets(pc, yca.to_numpy(), pt, alpha=0.10, mondrian=True)
    q1, q2 = np.quantile(pc, [1 / 3, 2 / 3])
    pot = np.select([pt < q1, pt < q2], ["Low", "Medium"], "High")
    # The potential model never sees the rating; recover it for the same test rows (split depends on y + seed only)
    X, y, sens, _, _ = task_frame(df, "potential")
    rating = split(X, y, sens.assign(rating=df["previous_year_rating"].to_numpy()), SEED)["test"][2]["rating"]
    perf = np.select([rating <= 2, rating == 3, rating >= 4], ["Low", "Medium", "High"], "Unrated")
    box = pd.DataFrame({"performance": perf, "potential": pot, "promoted": yte, "ambiguous": S.sum(1) == 2})
    box = box[box["performance"] != "Unrated"]
    order = ["Low", "Medium", "High"]
    agg = box.groupby(["performance", "potential"]).agg(n=("promoted", "size"), promotion_rate=("promoted", "mean"),
                                                        deferral_rate=("ambiguous", "mean")).reset_index()
    agg.to_csv(TABLES / "nine_box.csv", index=False)

    grid = agg.pivot(index="performance", columns="potential", values="promotion_rate").reindex(order[::-1])[order]
    n = agg.pivot(index="performance", columns="potential", values="n").reindex(order[::-1])[order]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    vmax = max(0.3, np.nanmax(grid.values))
    im = ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=vmax)
    for i in range(3):
        for j in range(3):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:.1%}\n(n={int(n.values[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * vmax else INK)
    ax.set_xticks(range(3), order)
    ax.set_yticks(range(3), order[::-1])
    ax.set(xlabel="Predicted potential (tertile)", ylabel="Observed performance (rating)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Observed promotion rate", shrink=0.8)
    fig.savefig(FIGURES / "fig_nine_box.png")
    plt.close(fig)
    return name


def main():
    df = load_promotion()
    chosen = {}
    for task in ["talent", "performance"]:
        print(f"== {task}")
        calibration_plot(task, df)
        convergence_plot(task, df)
        confusion_and_errors(task, df)
        chosen[task] = shap_analysis(task, df)
        fairness_analysis(task, df)
        print(conformal_analysis(task, df)[["method", "alpha", "coverage", "coverage_class1",
                                            "share_ambiguous_{0,1}"]].round(3).to_string())
    chosen["potential"] = nine_box(df)
    (TABLES / "chosen_models.json").write_text(json.dumps(chosen, indent=2))
    print("chosen models:", chosen)


if __name__ == "__main__":
    main()
