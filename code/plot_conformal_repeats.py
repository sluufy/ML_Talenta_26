"""Figure: positive-class coverage and deferral share (mean ± SD over 10 splits), marginal vs Mondrian."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, TABLES
from run_trust import C1, C2, INK, INK2

res = pd.read_csv(TABLES / "conformal_repeats_runs.csv")
for task in ["talent", "performance"]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    alphas = [0.20, 0.10, 0.05]
    x, w = np.arange(3), 0.38
    for ax, metric, ylabel in [(axes[0], "coverage_class1", "Coverage on positive class"),
                               (axes[1], "ambiguous", "Share deferred ({0,1} sets)")]:
        for k, (meth, color, label) in enumerate([("Marginal", C2, "Marginal split"),
                                                  ("Mondrian", C1, "Mondrian (class-conditional)")]):
            g = res[(res.task == task) & (res.method == meth)].groupby("alpha")[metric]
            m, s = g.mean().reindex(alphas), g.std().reindex(alphas)
            bars = ax.bar(x + (k - 0.5) * w, m, w - 0.03, yerr=s, capsize=2, color=color, label=label,
                          error_kw={"ecolor": INK2, "lw": 0.8})
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in m], fontsize=7, color=INK2, padding=3)
        if metric == "coverage_class1":
            for xi, t in zip(x, [0.8, 0.9, 0.95]):
                ax.hlines(t, xi - 0.42, xi + 0.42, colors=INK, linestyles="--", lw=1)
        ax.set_xticks(x, ["α = 0.20", "α = 0.10", "α = 0.05"])
        ax.set(ylabel=ylabel, ylim=(0, 1.12))
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig_conformal_coverage_{task}.png")
    plt.close(fig)
    print("saved", task)
