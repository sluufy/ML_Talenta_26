"""Graphical abstract (Elsevier: >= 1328 x 531 px, 5:2 ratio)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIGURES, TABLES

INK, INK2 = "#0b0b0b", "#52514e"
# grayscale only: dark = ML / emphasis, light = conventional
C1, C2, C3, C4 = "#2b2b2b", "#a6a6a6", "#2b2b2b", "#2b2b2b"
HI_FILL, LO_FILL, HI_EDGE, LO_EDGE = "#cfcfcf", "#f2f2f2", "#0b0b0b", "#6b6b6b"

place = pd.read_csv(TABLES / "box_placement_runs.csv")
prio = pd.read_csv(TABLES / "box_prioritisation_runs.csv")
prio = prio[prio["pool"] == "8-9"]
rec = place[place.set == "Boxes 8-9"].groupby("method").recall.mean()
p10 = prio[prio.top_frac == 0.10].groupby("ranker").precision.mean()

fig = plt.figure(figsize=(13.28, 5.31))
fig.patch.set_facecolor("white")
fig.text(0.5, 0.94, "Regulation-compliant ML for civil-service nine-box placement and promotion (boxes 8–9)",
         ha="center", fontsize=15, weight="bold", color=INK)


def panel(x, title, color):
    ax = fig.add_axes([x, 0.06, 0.29, 0.80])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec=color, lw=2))
    ax.text(0.5, 0.93, title, ha="center", va="center", fontsize=13, weight="bold", color=INK)
    return ax


# Panel 1: regulatory nine-box
a = panel(0.02, "1  Regulatory nine-box", C1)
labels = [["6", "8", "9"], ["3", "5", "7"], ["1", "2", "4"]]  # PermenPANRB 20/2025 Annex
for i in range(3):
    for j in range(3):
        hi = labels[i][j] in ("8", "9")
        a.add_patch(plt.Rectangle((0.2 + j * 0.22, 0.56 - i * 0.16), 0.2, 0.14, fc=HI_FILL if hi else LO_FILL,
                                  ec=HI_EDGE if hi else LO_EDGE, lw=2.2 if hi else 1))
        a.text(0.3 + j * 0.22, 0.63 - i * 0.16, f"Box {labels[i][j]}", ha="center", va="center", fontsize=9.5,
               weight="bold" if hi else "normal")
a.text(0.53, 0.18, "Performance = official appraisal (kept)", ha="center", fontsize=9.5, color=INK)
a.text(0.08, 0.48, "Potential", rotation=90, ha="center", va="center", fontsize=9.5, color=INK)
a.text(0.53, 0.09, "Boxes 8–9 = high-potential pool (7–9 = agency pool)", ha="center", fontsize=9.5, color=INK, weight="bold")

# Panel 2: two-stage ML
b = panel(0.355, "2  Two-stage ML + accountability", C3)
steps = ["Stage 1: age-blind CatBoost calibrates potential", "Stage 2: rank boxes 8–9 for limited vacancies",
         "SHAP explanation per candidate", "Fairness audit (gender, age)",
         "Mondrian conformal deferral", "Talent committee decides"]
for i, s in enumerate(steps):
    y = 0.78 - i * 0.125
    b.add_patch(FancyBboxPatch((0.06, y - 0.045), 0.88, 0.09, boxstyle="round,pad=0.005,rounding_size=0.02",
                               fc=HI_FILL if i < 2 else LO_FILL, ec="#333333", lw=1))
    b.text(0.5, y, s, ha="center", va="center", fontsize=9.6, color=INK)
    if i < len(steps) - 1:
        b.add_patch(FancyArrowPatch((0.5, y - 0.047), (0.5, y - 0.078), arrowstyle="-|>", mutation_scale=8, color=INK2))

# Panel 3: results
c = panel(0.69, "3  Evidence (10 held-out splits)", C4)
for k, (title, vals, ypos) in enumerate([
        ("Boxes 8–9 recall of promoted employees", [("Conventional", rec["Conventional"], C2),
                                               ("ML, age-blind", rec["ML-assisted (age-blind)"], C1)], 0.55),
        ("Top-10% precision inside boxes 8–9", [("Raw score", p10["Training score"], C2),
                                               ("ML, age-blind", p10["CatBoost (age-blind)"], C1)], 0.21)]):
    ax = fig.add_axes([0.745, ypos, 0.2, 0.17])
    ax.barh([v[0] for v in vals], [v[1] for v in vals], color=[v[2] for v in vals], height=0.55)
    for i, v in enumerate(vals):
        ax.text(v[1] + 0.02, i, f"{v[1]:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.05); ax.set_title(title, fontsize=9.5, color=INK2)
    ax.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
c.text(0.5, 0.05, "Smaller age gap; unsure cases deferred", ha="center", fontsize=9.6, weight="bold",
       color=C4)

for x0 in (0.315, 0.65):
    fig.add_artist(FancyArrowPatch((x0, 0.46), (x0 + 0.035, 0.46), arrowstyle="-|>", mutation_scale=18, color=INK2,
                                   transform=fig.transFigure))
fig.savefig(FIGURES / "graphical_abstract.png", dpi=200)
print("saved graphical_abstract.png")
