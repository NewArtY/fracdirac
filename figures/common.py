"""Parameter sets and helpers shared by the figure scripts."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

import numpy as np  # noqa: E402
from fracdirac import model as md, solvers as sv, style  # noqa: E402,F401

# Set A: Figs. 1-6 and the tables (single trajectory k = 0 unless stated otherwise)
SET_A = dict(s_p=3.0, g=1.5, T1=40.0, T2=20.0, tau_m=1.0)
XA = (-15.0, 16.0)          # omega t in [x0, x_obs]
NA = 6200                   # time steps (delta x = 0.005)
# Set B: Figs. 7-8 (Landau-Zener-Stueckelberg interference)
SET_B = dict(s_p=3.7, g=0.65, T1=80.0, T2=45.0, tau_m=1.0)
KAPPA_B = 0.35
XB = (-18.0, 16.0)
NB = 3400                   # delta x = 0.01


def n_res(model, alpha, prm, xr, N, scheme="bdf2"):
    """Residual conduction-band population n_c(x_obs) for all parameter points."""
    _, St = sv.population(model, alpha, prm, xr[0], xr[1], N, scheme)
    return 0.5 * (1.0 + St[-1, :, 2]), float(np.linalg.norm(St, axis=-1).max())


def n_res_rich(model, alpha, prm, xr, N, scheme="bdf2"):
    """Residual population Richardson-extrapolated from N and N/2 steps (second-order scheme).

    Returns the extrapolated values, the largest estimated error of the un-extrapolated
    N-step result, max |n(N) - n(N/2)| / 3, and the largest Bloch-vector norm."""
    a, nrm = n_res(model, alpha, prm, xr, N, scheme)
    if alpha >= 1.0:
        return a, 0.0, nrm
    b, nrm2 = n_res(model, alpha, prm, xr, N // 2, scheme)
    return (4.0 * a - b) / 3.0, float(np.abs(a - b).max() / 3.0), max(nrm, nrm2)


def write_meta(name, meta):
    (DATA / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")


def plot_only():
    return "--plot-only" in sys.argv
