# fracdirac — code and data for "Memory-Induced Geometric Coupling in the Nonlinear Fractional Dynamics of Laser-Driven Massive Dirac Fermions"

Authors of the manuscript: N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Q.-H. Qin.

This archive contains everything needed to regenerate every number, table and figure of the
manuscript: the solver library, the verification suite, the figure scripts, and the data
files they produce.

## Contents

| Path | Purpose |
|---|---|
| `fracdirac/model.py` | dimensionless laser-driven massive Dirac model: pulse, kinematics, Bloch generators in the fixed and instantaneous bases |
| `fracdirac/solvers.py` | Caputo time stepping (L1 and second-order BDF2 convolution quadrature) for the fixed-basis (covariant) model and the reduced models N1 and N2; explicit two-time (covariant) scheme; rephased N1; scalar adiabatic-limit equation; Markovian reference |
| `fracdirac/mittag_leffler.py` | Mittag-Leffler function and exact constant-generator solution (reference values) |
| `fracdirac/style.py` | Matplotlib style shared by the figures |
| `tests/verification.py` | verification suite, tests 1–8 of the verification table of the manuscript → `data/verification.json` |
| `analysis/tables.py` | momentum-integrated excitation, Markovian surrogate fits, sensitivity to the memory time and to the observation time, gauge dependence of N1, residual populations versus the fractional order → `data/tables.json` |
| `figures/fig1_kinematics.py` … `fig8_fringes.py` | one script per figure: computes the data, writes `data/figN_*.csv` and `data/figN_*_meta.json`, draws `output/FigN.pdf` and `output/FigN.png` |
| `figures/graphical_abstract.py` | graphical abstract (uses `data/fig6_adiabatic_gap.csv`) |
| `figures/common.py` | parameter sets A and B, grids, helpers |
| `data/` | data produced by the scripts (CSV, JSON), as used in the manuscript |
| `output/` | figures and run logs |
| `run_all.py` | runs everything in the required order |

Dependencies between scripts: `figures/fig6_adiabatic.py` reads the surrogate dephasing time
from `data/tables.json` (run `analysis/tables.py` first); `figures/graphical_abstract.py` reads
`data/fig6_adiabatic_gap.csv`. `run_all.py` respects this order.

## Model

Dimensionless variables: time `x = omega t`, kinetic momentum `q = v_F Pi_k / Delta = sinh(theta)`,
`q(x) = kappa + a_D exp(-x^2 / 2 s_p^2) cos(x + phi)`. Fixed-basis Bloch equation

    tau_m^(alpha-1) D^alpha s = B(x) s + b(x),
    B s = Omega x s - (s.h)h / T1 - (s - (s.h)h) / T2,   b = -h / T1,
    Omega = 2 g (q, 0, 1),   h = (q, 0, 1) / sqrt(1 + q^2),   g = Delta / (hbar omega).

Models (`fracdirac.solvers.population(model, ...)`):

| name | manuscript | description |
|---|---|---|
| `fixed` | fixed-basis model (Section 3) | Caputo equation in the fixed pseudospin basis |
| `covariant` | covariant model (Section 4) and its discretization (Section 6) | the same dynamics in the instantaneous basis with the explicit two-time transporter |
| `naive` | reduced model N1 | Caputo derivative in the instantaneous basis plus the local connection |
| `notransport` | reduced model N2 | covariant integrand with the transporter replaced by the identity |

Further solvers: `solve_naive_gauge` (N1 with rephased eigenvectors), `solve_adiabatic` and
`adiabatic_first_order` (scalar adiabatic-limit equation and its first-order solution),
`solve_markov` (alpha = 1).

## Requirements

Python ≥ 3.10 with the packages in `requirements.txt` (tested with Python 3.14.3, NumPy 2.4.2,
SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.11.1, mpmath 1.3.0).

    pip install -r requirements.txt

## Usage

    python run_all.py               # verification, tables and all figures (about 1.5 hours on a laptop, < 1 GB RAM)
    python tests/verification.py    # verification suite only
    python figures/fig4_models.py   # one figure: compute + plot
    python figures/fig4_models.py --plot-only   # redraw from the stored CSV
    python run_all.py --plot-only   # recompute verification and tables, redraw all figures from stored data

All calculations are deterministic (no random numbers). Double precision is used throughout.

## Numerical parameters

Set A (Figs. 1–6 and the tables): `omega tau_p = 3`, `Delta/(hbar omega) = 1.5`, `omega T1 = 40`,
`omega T2 = 20`, `omega tau_m = 1`, `k = 0`, `phi = 0`; `omega t` from −15 to 16; 6200 steps.
Set B (Figs. 7–8): `omega tau_p = 3.7`, `Delta/(hbar omega) = 0.65`, `hbar v_F k/Delta = 0.35`,
`omega T1 = 80`, `omega T2 = 45`, `omega tau_m = 1`; `omega t` from −18 to 16; 3400 steps.
Zero temperature (`w_eq = -1`) everywhere. Tabulated values and the maps of Figs. 5 and 7 are
Richardson-extrapolated in the time step (N and N/2 steps); error estimates are recorded in the
`*_meta.json` files and in `data/verification.json` (test 3).

## Citation

Please cite the archived release (Zenodo; the DOI is given in the article) and the article
itself. Citation metadata are in `CITATION.cff`; the Zenodo record is described by
`.zenodo.json`.

## Licence

Code: MIT licence (see `LICENSE`). Data and figures (folders `data/` and `output/`): CC BY 4.0.
