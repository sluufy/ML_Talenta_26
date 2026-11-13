"""End-to-end smoke test: run the main experiment scripts on SYNTHETIC data and check that they finish and write
their tables. Results go to a temporary directory; nothing in results/ is touched. Never used for reported results.

Usage (from the repository root):  python tests/smoke_pipeline.py [n_rows]
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
n = sys.argv[1] if len(sys.argv) > 1 else "4000"

tmp = Path(tempfile.mkdtemp(prefix="hrml_smoke_"))
(tmp / "tables").mkdir()
(tmp / "figures").mkdir()
data = tmp / "smoke.csv"
subprocess.run([sys.executable, str(CODE / "make_smoke_data.py"), str(data), n], check=True)
# the experiment scripts reuse the tuned hyper-parameters of the benchmark
# and the model ranking chosen by the benchmark (read by the plotting helpers)
for name in [f"best_params_{t}.json" for t in ("potential", "talent", "performance")] + ["chosen_models.json"]:
    src = ROOT / "results" / "tables" / name
    if src.exists():
        shutil.copy(src, tmp / "tables" / name)

env = {**os.environ, "HRML_DATA": str(data), "HRML_RESULTS": str(tmp), "MPLBACKEND": "Agg"}
steps = [
    ("run_talent_box.py", ["box_placement_summary.csv", "box_prioritisation_summary.csv", "box_tests.csv",
                           "box_cells_summary.csv"]),
    ("run_box_extras.py", ["box_deferral_ageblind_runs.csv"]),
]
# (plotting scripts need the outputs of every experiment and are therefore not part of the smoke test)
failed = False
for script, outputs in steps:
    t0 = time.time()
    r = subprocess.run([sys.executable, script], cwd=CODE, env=env, capture_output=True, text=True)
    missing = [o for o in outputs if not (tmp / "tables" / o).exists()]
    ok = r.returncode == 0 and not missing
    failed |= not ok
    print(f"{'PASS' if ok else 'FAIL'}  {script:<22} {time.time() - t0:6.1f} s" + (f"  missing: {missing}" if missing else ""))
    if r.returncode != 0:
        print(r.stderr[-2000:])
        break

print("temporary results:", tmp)
sys.exit(1 if failed else 0)
