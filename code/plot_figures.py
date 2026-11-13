"""Redraw the result figures from saved CSV tables in English and Indonesian (no model refitting).

Outputs results/figures/<name>_en.png and <name>_id.png for: confusion, convergence, conformal coverage, nine-box.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, TABLES
from run_trust import C1, C2, INK, INK2

L = {
    "en": {"pred": "Predicted", "act": "Actual", "talent": ["Not promoted", "Promoted"],
           "performance": ["Not high", "High performer"], "thr": "threshold", "iter": "Boosting iteration",
           "loss": "Log-loss (binary cross-entropy)", "train": "Training", "val": "Validation",
           "minval": "min. validation loss", "cov": "Coverage on positive class",
           "defer": "Share deferred (ambiguous sets)", "marg": "Marginal split",
           "mond": "Mondrian (class-conditional)", "pot": "Predicted potential (tertile)",
           "perf": "Observed performance (rating)", "rate": "Observed promotion rate", "levels": ["Low", "Medium", "High"],
           "deferred": "deferred", "task": {"talent": "Talent", "performance": "Performance"}},
    "id": {"pred": "Prediksi", "act": "Aktual", "talent": ["Tidak promosi", "Promosi"],
           "performance": ["Bukan tinggi", "Kinerja tinggi"], "thr": "ambang", "iter": "Iterasi boosting",
           "loss": "Log-loss (binary cross-entropy)", "train": "Pelatihan", "val": "Validasi",
           "minval": "loss validasi minimum", "cov": "Cakupan pada kelas positif",
           "defer": "Proporsi dirujuk (himpunan ambigu)", "marg": "Split marginal",
           "mond": "Mondrian (bersyarat kelas)", "pot": "Potensi terprediksi (tertil)",
           "perf": "Kinerja teramati (nilai)", "rate": "Tingkat promosi teramati", "levels": ["Rendah", "Sedang", "Tinggi"],
           "deferred": "dirujuk", "task": {"talent": "Talenta", "performance": "Kinerja"}},
}
def loc(s: str, lang: str) -> str:
    """Indonesian number format: 1,234.5 -> 1.234,5."""
    return s.replace(",", "X").replace(".", ",").replace("X", ".") if lang == "id" else s


chosen = pd.read_json(TABLES / "chosen_models.json", typ="series")
cal = {t: pd.read_csv(TABLES / f"calibration_{t}.csv").set_index("model") for t in ["talent", "performance"]}

for lang, T in L.items():
    for task in ["talent", "performance"]:
        name = chosen[task]
        # confusion matrix
        cm = pd.read_csv(TABLES / f"confusion_{task}.csv", index_col=0).to_numpy()
        thr = cal[task].loc[name, "threshold"]
        fig, ax = plt.subplots(figsize=(3.3, 2.9))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, loc(f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})", lang),
                        ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() * 0.55 else INK)
        ax.set_xticks([0, 1], T[task]); ax.set_yticks([0, 1], T[task], rotation=90, va="center")
        ax.set(xlabel=T["pred"], ylabel=T["act"], title=loc(f"{T['task'][task]} – {name} ({T['thr']} = {thr:.2f})", lang))
        ax.title.set_fontsize(9); ax.grid(False)
        fig.savefig(FIGURES / f"cm_{task[:4]}_{lang}.png"); plt.close(fig)

        # convergence
        cv = pd.read_csv(TABLES / f"convergence_{task}.csv")
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        ax.plot(cv.iteration, cv.train_logloss, color=C1, label=T["train"])
        ax.plot(cv.iteration, cv.val_logloss, color=C2, label=T["val"])
        best = int(cv.val_logloss.idxmin()) + 1
        ax.axvline(best, color=INK2, ls="--", lw=1)
        ax.text(best, cv.val_logloss.max(), f"{T['minval']}\n@ {best}", fontsize=7, color=INK2, ha="right", va="top")
        ax.set(xlabel=T["iter"], ylabel=T["loss"], title=f"{T['task'][task]} – {name}")
        ax.title.set_fontsize(9); ax.legend(frameon=False, fontsize=8, loc="center right")
        fig.savefig(FIGURES / f"conv_{task[:4]}_{lang}.png"); plt.close(fig)

        # conformal (10 repeats)
        res = pd.read_csv(TABLES / "conformal_repeats_runs.csv")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
        alphas, x, w = [0.20, 0.10, 0.05], np.arange(3), 0.38
        for ax, metric, ylabel in [(axes[0], "coverage_class1", T["cov"]), (axes[1], "ambiguous", T["defer"])]:
            for k, (meth, color, label) in enumerate([("Marginal", C2, T["marg"]), ("Mondrian", C1, T["mond"])]):
                g = res[(res.task == task) & (res.method == meth)].groupby("alpha")[metric]
                m, s = g.mean().reindex(alphas), g.std().reindex(alphas)
                bars = ax.bar(x + (k - 0.5) * w, m, w - 0.03, yerr=s, capsize=2, color=color, label=label,
                              error_kw={"ecolor": INK2, "lw": 0.8})
                ax.bar_label(bars, labels=[loc(f"{v:.2f}", lang) for v in m], fontsize=7, color=INK2, padding=3)
            if metric == "coverage_class1":
                for xi, t in zip(x, [0.8, 0.9, 0.95]):
                    ax.hlines(t, xi - 0.42, xi + 0.42, colors=INK, linestyles="--", lw=1)
            ax.set_xticks(x, ["α = 0.20", "α = 0.10", "α = 0.05"]); ax.set(ylabel=ylabel, ylim=(0, 1.12))
            ax.grid(axis="x", visible=False)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(FIGURES / f"conf_{task[:4]}_{lang}.png"); plt.close(fig)

    # nine-box with deferral share
    nb = pd.read_csv(TABLES / "nine_box.csv")
    order = ["Low", "Medium", "High"]
    grid = nb.pivot(index="performance", columns="potential", values="promotion_rate").reindex(order[::-1])[order]
    n = nb.pivot(index="performance", columns="potential", values="n").reindex(order[::-1])[order]
    d = nb.pivot(index="performance", columns="potential", values="deferral_rate").reindex(order[::-1])[order]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    vmax = 0.3
    im = ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=vmax)
    for i in range(3):
        for j in range(3):
            v = grid.values[i, j]
            txt = f"{v:.1%}\nn={int(n.values[i, j]):,}\n{d.values[i, j]:.0%} {T['deferred']}"
            if lang == "id":
                txt = txt.replace(",", "X").replace(".", ",").replace("X", ".")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="white" if v > 0.55 * vmax else INK)
    ax.set_xticks(range(3), T["levels"]); ax.set_yticks(range(3), T["levels"][::-1])
    ax.set(xlabel=T["pot"], ylabel=T["perf"]); ax.grid(False)
    fig.colorbar(im, ax=ax, label=T["rate"], shrink=0.8)
    fig.savefig(FIGURES / f"ninebox_{lang}.png"); plt.close(fig)
    print("saved", lang)
