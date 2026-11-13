"""Figures for the talent-box experiment (EN and ID): nine-box comparison and prioritisation curve."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, TABLES
from plot_figures import loc
from run_trust import C1, C2, C3, INK, INK2

L = {
    "en": {"conv": "(a) Conventional nine-box", "ml": "(b) ML-assisted nine-box (age-blind)",
           "perf": "Performance (appraisal)", "pot": "Potential", "lv": ["Low", "Medium", "High"],
           "perf_lv": ["Below", "As expected", "Above"], "rate": "Observed promotion rate", "box": "Box",
           "top": "Top share of boxes 8–9 pool selected", "prec": "Precision (share actually promoted)",
           "rank": {"Training score": "Assessment score (raw)", "Dept-normalised score": "Assessment score (dept-normalised)",
                    "Logistic regression": "Logistic regression", "CatBoost (age-blind)": "CatBoost, age-blind (proposed)",
                    "Random": "Random order"}},
    "id": {"conv": "(a) Kotak sembilan konvensional", "ml": "(b) Kotak sembilan berbantuan ML (buta-usia)",
           "perf": "Kinerja (penilaian)", "pot": "Potensi", "lv": ["Rendah", "Menengah", "Tinggi"],
           "perf_lv": ["Di bawah", "Sesuai", "Di atas"], "rate": "Tingkat promosi teramati", "box": "Kotak",
           "top": "Proporsi teratas kumpulan kotak 8–9 yang dipilih", "prec": "Presisi (proporsi yang benar dipromosikan)",
           "rank": {"Training score": "Skor asesmen (mentah)", "Dept-normalised score": "Skor asesmen (ternormalisasi unit)",
                    "Logistic regression": "Regresi logistik", "CatBoost (age-blind)": "CatBoost buta-usia (usulan)",
                    "Random": "Urutan acak"}},
}
# (performance, potential) -> box, Annex of PermenPANRB 20/2025
BOX_NO = {("Low", "Low"): 1, ("Medium", "Low"): 2, ("Low", "Medium"): 3, ("High", "Low"): 4, ("Medium", "Medium"): 5,
          ("Low", "High"): 6, ("High", "Medium"): 7, ("Medium", "High"): 8, ("High", "High"): 9}
cells = pd.read_csv(TABLES / "box_cells_summary.csv")
prio = pd.read_csv(TABLES / "box_prioritisation_runs.csv")
prio = prio[prio["pool"] == "8-9"]

for lang, T in L.items():
    # nine-box comparison: x = performance, y = potential (ASN layout)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"wspace": 0.18})
    vmax = 0.32
    for ax, (method, title) in zip(axes, [("Conventional", T["conv"]), ("ML-assisted (age-blind)", T["ml"])]):
        sub = cells[cells.method == method].set_index("box")
        grid = np.zeros((3, 3))
        for i, pot in enumerate(["High", "Medium", "Low"]):
            for j, perf in enumerate(["Low", "Medium", "High"]):
                b = BOX_NO[(perf, pot)]
                v, n = sub.loc[b, "promotion_rate"], sub.loc[b, "n"]
                grid[i, j] = v
                ax.text(j, i - 0.22, f"{T['box']} {b}", ha="center", va="center", fontsize=8.5, weight="bold",
                        color="white" if v > 0.55 * vmax else INK)
                ax.text(j, i + 0.12, loc(f"{v:.1%}\nn = {n:,.0f}", lang), ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 * vmax else INK)
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=vmax)
        for (i, j) in [(0, 1), (0, 2)]:  # boxes 8 and 9 (high potential, promotion-eligible)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2.4))
        # box 7 completes the agency succession pool
        ax.add_patch(plt.Rectangle((1.5, 0.5), 1, 1, fill=False, ec="#eb6834", lw=1.4, ls="--"))
        ax.set_xticks(range(3), T["perf_lv"])
        ax.set_yticks(range(3), T["lv"][::-1] if method == "Conventional" else [""] * 3)
        ax.set(xlabel=T["perf"], ylabel=T["pot"] if method == "Conventional" else "", title=title)
        ax.title.set_fontsize(9.5); ax.grid(False)
    fig.colorbar(im, ax=axes, label=T["rate"], shrink=0.8)
    fig.savefig(FIGURES / f"box_{lang}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # prioritisation curve
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    styles = {"CatBoost (age-blind)": (C1, "-", 2.2), "Logistic regression": (C3, "-", 1.4),
              "Dept-normalised score": ("#9085e9", "--", 1.6), "Training score": (C2, "-", 1.6),
              "Random": (INK2, ":", 1.4)}
    for name, (color, ls, lw) in styles.items():
        g = prio[prio.ranker == name].groupby("top_frac")["precision"]
        m, s = g.mean(), g.std()
        ax.plot(m.index * 100, m.values, color=color, ls=ls, lw=lw, label=T["rank"][name])
        ax.fill_between(m.index * 100, m - s, m + s, color=color, alpha=0.12, lw=0)
    ax.set(xlabel=T["top"], ylabel=T["prec"], ylim=(0, 1.02), xlim=(5, 100))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    fig.savefig(FIGURES / f"prio_{lang}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved", lang)
