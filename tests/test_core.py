"""Unit tests for the methodological guarantees the paper relies on.

Run from the repository root:  python -m unittest discover -s tests -v   (or: pytest tests)
No real data are needed: the tests use synthetic records with the Kaggle promotion schema.
"""
import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from common import dev_holdout, ece, load_promotion, split, task_frame  # noqa: E402
from run_trust import conformal_sets  # noqa: E402

# Annex of PermenPANRB 20/2025: (performance, potential) -> box
OFFICIAL = {("Low", "Low"): 1, ("Medium", "Low"): 2, ("Low", "Medium"): 3, ("High", "Low"): 4,
            ("Medium", "Medium"): 5, ("Low", "High"): 6, ("High", "Medium"): 7, ("Medium", "High"): 8,
            ("High", "High"): 9}


def box_no_in(script: str) -> dict:
    """Read the BOX_NO literal from a script without executing it (the experiment scripts run on import)."""
    tree = ast.parse((CODE / script).read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "BOX_NO" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"BOX_NO not found in {script}")


class SyntheticData:
    _df = None

    @classmethod
    def df(cls):
        if cls._df is None:
            tmp = Path(tempfile.mkdtemp()) / "smoke.csv"
            subprocess.run([sys.executable, str(CODE / "make_smoke_data.py"), str(tmp), "6000"], check=True,
                           capture_output=True)
            cls._df = load_promotion(tmp)
        return cls._df


class TestNineBoxLayout(unittest.TestCase):
    def test_scripts_use_official_annex(self):
        for script in ("run_talent_box.py", "run_box_extras.py", "plot_talent_box.py"):
            with self.subTest(script=script):
                self.assertEqual(box_no_in(script), OFFICIAL)

    def test_succession_pools(self):
        inv = {v: k for k, v in OFFICIAL.items()}
        self.assertEqual(inv[9], ("High", "High"))      # above expectation, high potential
        self.assertEqual(inv[8], ("Medium", "High"))    # as expected, high potential
        self.assertEqual(inv[7], ("High", "Medium"))    # above expectation, medium potential
        # boxes 8-9 are exactly the high-potential boxes with at least as-expected performance
        self.assertEqual({b for (perf, pot), b in OFFICIAL.items() if pot == "High" and perf != "Low"}, {8, 9})
        self.assertEqual(sorted(OFFICIAL.values()), list(range(1, 10)))


class TestDataProtocol(unittest.TestCase):
    def setUp(self):
        df = SyntheticData.df()
        X, y, sens, _, _ = task_frame(df, "talent")
        self.X = X.assign(_row=np.arange(len(X)))
        self.y, self.sens = y, sens

    def test_tuning_partition_never_used_for_evaluation(self):
        (Xd, _, _), (Xe, _, _) = dev_holdout(self.X, self.y, self.sens)
        dev, ev = set(Xd["_row"]), set(Xe["_row"])
        self.assertFalse(dev & ev)
        self.assertAlmostEqual(len(dev) / len(self.X), 0.2, delta=0.01)
        for seed in (1000, 1009):
            parts = split(self.X, self.y, self.sens, seed)
            rows = [set(parts[k][0]["_row"]) for k in ("train", "val", "cal", "test")]
            with self.subTest(seed=seed):
                self.assertFalse(dev & set().union(*rows))
                for i in range(4):
                    for j in range(i + 1, 4):
                        self.assertFalse(rows[i] & rows[j])
                self.assertEqual(set().union(*rows), ev)

    def test_split_proportions_and_stratification(self):
        parts = split(self.X, self.y, self.sens, 1000)
        n = sum(len(parts[k][1]) for k in parts)
        for k, share in {"train": 0.6, "val": 0.1, "cal": 0.1, "test": 0.2}.items():
            with self.subTest(part=k):
                self.assertAlmostEqual(len(parts[k][1]) / n, share, delta=0.01)
                self.assertAlmostEqual(parts[k][1].mean(), self.y.mean(), delta=0.02)

    def test_seeds_are_reproducible(self):
        a = split(self.X, self.y, self.sens, 1003)["test"][0]["_row"].tolist()
        b = split(self.X, self.y, self.sens, 1003)["test"][0]["_row"].tolist()
        self.assertEqual(a, b)


class TestFeatureSets(unittest.TestCase):
    def test_sensitive_and_leaking_features_are_excluded(self):
        df = SyntheticData.df()
        for task in ("talent", "potential", "performance"):
            X, _, sens, _, _ = task_frame(df, task)
            with self.subTest(task=task):
                self.assertNotIn("gender", X.columns)          # fairness through unawareness
                self.assertNotIn("is_promoted", X.columns)     # target never an input
                self.assertIn("gender", sens.columns)          # kept only for the fairness audit
        Xp, *_ = task_frame(df, "potential")
        self.assertNotIn("previous_year_rating", Xp.columns)   # potential axis is independent of the rating
        self.assertNotIn("kpis_met", Xp.columns)
        Xf, *_ = task_frame(df, "performance")
        self.assertNotIn("previous_year_rating", Xf.columns)   # label of the performance task


class TestConformal(unittest.TestCase):
    def test_mondrian_guarantees_coverage_within_each_class(self):
        rng = np.random.default_rng(0)
        n, prev, alpha = 40000, 0.085, 0.10
        y = (rng.random(n) < prev).astype(int)
        p = np.clip(0.08 + 0.25 * y + rng.normal(0, 0.12, n), 0.001, 0.999)   # weak, imbalanced classifier
        cal, te = np.arange(n) < n // 2, np.arange(n) >= n // 2
        for mondrian in (True, False):
            sets = conformal_sets(p[cal], y[cal], p[te], alpha, mondrian=mondrian)
            cov1 = sets[y[te] == 1, 1].mean()
            cov0 = sets[y[te] == 0, 0].mean()
            if mondrian:
                self.assertGreaterEqual(cov1, 1 - alpha - 0.02)
                self.assertGreaterEqual(cov0, 1 - alpha - 0.02)
            else:   # marginal coverage hides under-coverage of the rare class
                self.assertLess(cov1, 1 - alpha - 0.05)


class TestMetrics(unittest.TestCase):
    def test_ece(self):
        y = np.array([0, 1] * 500)
        self.assertAlmostEqual(ece(y, y.astype(float)), 0.0, places=6)
        self.assertAlmostEqual(ece(y, np.full(len(y), 0.9)), 0.4, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
