"""Figure 1: end-to-end protocol diagram (English and Indonesian versions)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIGURES

INK, INK2 = "#0b0b0b", "#52514e"
FILL = {"data": "#e8eef7", "eval": "#fdebe2", "trust": "#e3f4ec", "decision": "#fff4d6"}
EDGE = {"data": "#2a78d6", "eval": "#eb6834", "trust": "#1baf7a", "decision": "#c98500"}

TEXT = {
    "en": [
        ("data", "1. Data\nPromotion data (n = 54,808)\nIBM HR (n = 1,470)"),
        ("data", "2. Leakage audit\nDeterministic-rule check\nwith/without ablation\npermutation test"),
        ("data", "3. Partitioning\n20% tuning partition\n80% evaluation partition"),
        ("eval", "4. Preprocessing\nfitted in-fold only\nmedian/mode imputation\n+ missing indicator, one-hot"),
        ("eval", "5. Models & tuning\nLR, RF, XGBoost, LightGBM,\nCatBoost, MLP, LightGBM-CW\nrandomised search (PR-AUC)"),
        ("eval", "6. Evaluation (×10)\ntrain 60/val 10/cal 10/test 20\nPR-AUC, MCC, F1, ECE, latency\nFriedman + Wilcoxon–Holm"),
        ("decision", "7. Stage 1: placement\nperformance = appraisal\npotential = age-blind ML\n→ nine-box, boxes 8–9"),
        ("decision", "8. Stage 2: prioritise\nrank boxes 8–9 pool\nfor limited vacancies\n(precision@k)"),
        ("trust", "9. Stage 3: accountability\nconformal deferral, SHAP,\nfairness audit\n(gender, age)"),
        ("trust", "10. Talent committee\nconfirms or overrides;\nfinal promotion decision\nremains human"),
    ],
    "id": [
        ("data", "1. Data\nData promosi (n = 54.808)\nIBM HR (n = 1.470)"),
        ("data", "2. Audit kebocoran\nCek aturan deterministik\nablasi dengan/tanpa fitur\nuji permutasi"),
        ("data", "3. Partisi data\n20% partisi tuning\n80% partisi evaluasi"),
        ("eval", "4. Praproses\nhanya di dalam fold\nimputasi median/modus\n+ indikator hilang, one-hot"),
        ("eval", "5. Model & tuning\nLR, RF, XGBoost, LightGBM,\nCatBoost, MLP, LightGBM-CW\nrandomised search (PR-AUC)"),
        ("eval", "6. Evaluasi (×10)\nlatih 60/val 10/kal 10/uji 20\nPR-AUC, MCC, F1, ECE, latensi\nFriedman + Wilcoxon–Holm"),
        ("decision", "7. Tahap 1: penempatan\nkinerja = penilaian SKP\npotensi = ML buta-usia\n→ 9-box, kotak 8–9"),
        ("decision", "8. Tahap 2: prioritas\nurutkan kumpulan kotak 8–9\nuntuk formasi terbatas\n(presisi@k)"),
        ("trust", "9. Tahap 3: akuntabilitas\nrujukan konformal, SHAP,\naudit keadilan\n(gender, usia)"),
        ("trust", "10. Komite talenta\nmengonfirmasi/mengoreksi;\nkeputusan promosi akhir\ntetap oleh manusia"),
    ],
}

# 2 rows of 5 boxes, snake order
POS = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0)]

for lang, items in TEXT.items():
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(-0.1, 5.0); ax.set_ylim(-0.15, 1.62); ax.axis("off")
    W, H = 0.9, 0.62
    centers = []
    for (kind, label), (cx, cy) in zip(items, POS):
        x, y = cx * 0.98 + 0.02, cy * 0.85 + 0.1
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=FILL[kind], ec=EDGE[kind], lw=1.4))
        title, *rest = label.split("\n")
        ax.text(x + W / 2, y + H - 0.1, title, ha="center", va="top", fontsize=8, weight="bold", color=INK)
        ax.text(x + W / 2, y + H - 0.24, "\n".join(rest), ha="center", va="top", fontsize=7, color=INK2,
                linespacing=1.25)
        centers.append((x, y))
    for k in range(len(centers) - 1):
        (x0, y0), (x1, y1) = centers[k], centers[k + 1]
        if y0 == y1:
            start = (x0 + W, y0 + H / 2) if x1 > x0 else (x0, y0 + H / 2)
            end = (x1, y1 + H / 2) if x1 > x0 else (x1 + W, y1 + H / 2)
        else:
            start, end = (x0 + W / 2, y0), (x1 + W / 2, y1 + H)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2))
    legend = {"en": ["Data & leakage control", "Model selection", "Accountability", "Regulation-compliant talent placement"],
              "id": ["Data & kontrol kebocoran", "Pemilihan model", "Akuntabilitas", "Penempatan talenta patuh regulasi"]}[lang]
    for j, (kind, name) in enumerate(zip(FILL, legend)):
        ax.add_patch(FancyBboxPatch((0.05 + j * 1.25, -0.12), 0.12, 0.08, boxstyle="round,pad=0.01",
                                    fc=FILL[kind], ec=EDGE[kind]))
        ax.text(0.22 + j * 1.25, -0.08, name, va="center", fontsize=8, color=INK)
    fig.savefig(FIGURES / f"fig_pipeline_{lang}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", lang)
