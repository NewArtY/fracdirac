"""Regenerate all verification data, tables and figures of the manuscript."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = [
    ("tests", "verification.py"),
    ("analysis", "tables.py"),
    ("figures", "fig1_kinematics.py"),
    ("figures", "fig2_markov.py"),
    ("figures", "fig3_fractional.py"),
    ("figures", "fig4_models.py"),
    ("figures", "fig5_phase_diagram.py"),
    ("figures", "fig6_adiabatic.py"),
    ("figures", "fig7_lzs_maps.py"),
    ("figures", "fig8_fringes.py"),
    ("figures", "graphical_abstract.py"),
]
(ROOT / "output").mkdir(exist_ok=True)
for folder, script in JOBS:
    t = time.time()
    print(f"[run_all] {folder}/{script} ...", flush=True)
    with open(ROOT / "output" / (Path(script).stem + ".log"), "w", encoding="utf-8") as log:
        args = sys.argv[1:] if folder == "figures" else []      # e.g. --plot-only
        r = subprocess.run([sys.executable, script, *args], cwd=ROOT / folder,
                           stdout=log, stderr=subprocess.STDOUT)
    print(f"[run_all]   exit code {r.returncode}, {time.time() - t:.0f} s", flush=True)
    if r.returncode:
        sys.exit(r.returncode)
print("[run_all] done")
