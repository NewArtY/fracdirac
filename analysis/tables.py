"""Auxiliary calculations quoted in the tables and text of the manuscript
(results -> data/tables.json).  Set A, a_D = 2, k = 0 unless stated otherwise.

  A1  momentum-integrated residual excitation for the covariant and reduced models
  A2  Markovian surrogate of the fractional covariant dynamics: one-parameter (T2) and
      two-parameter (T1, T2) fits, and the Markovian excitation as a function of T2
  A3  sensitivity of the residual population to the memory time tau_m
  A4  gauge dependence of the reduced model N1
  A5  residual population of the three fractional models versus alpha (incl. alpha -> 1)
  A6  dependence of the model comparison on the observation time
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fracdirac import model as md, solvers as sv  # noqa: E402

SET_A = dict(s_p=3.0, g=1.5, T1=40.0, T2=20.0, tau_m=1.0)
X0, XF, N = -15.0, 16.0, 6200
MODELS = (("fixed", "cov"), ("naive", "N1"), ("notransport", "N2"))
out = {}


def nres(model, alpha, prm, N=N, xf=XF):
    _, St = sv.population(model, alpha, prm, X0, xf, N)
    return 0.5 * (1 + St[-1, :, 2])


def nres_rich(model, alpha, prm, N=N):
    """Richardson extrapolation of the second-order scheme (N and N/2 steps)."""
    if alpha >= 1.0:
        return nres("fixed", 1.0, prm, N)
    return (4.0 * nres(model, alpha, prm, N) - nres(model, alpha, prm, N // 2)) / 3.0


# ---------------------------------------------------------------- A1
# bar n = integral d(kappa) n_res(kappa), kappa = hbar v_F k / Delta (trapezoidal rule)
def kint(model, alpha, kmax, nk, rich=True):
    kap = np.linspace(-kmax, kmax, nk)
    prm = md.Params(a_D=2.0, kappa=kap, **SET_A)
    f = nres_rich if rich else nres
    return float(np.trapezoid(f(model, alpha, prm), kap))


a1 = {"definition": "integral over kappa of n_res(kappa), a_D = 2, set A; Richardson in time",
      "rows": {}}
for alpha in (1.0, 0.99, 0.95, 0.9, 0.8, 0.7):
    row = {tag: kint(m, alpha, 8.0, 321) for m, tag in MODELS}
    a1["rows"][str(alpha)] = row
    print("A1", alpha, row, flush=True)
a1["convergence_alpha0.8_cov"] = dict(
    kmax8_nk321=kint("fixed", 0.8, 8.0, 321, rich=False),
    kmax8_nk161=kint("fixed", 0.8, 8.0, 161, rich=False),
    kmax12_nk481=kint("fixed", 0.8, 12.0, 481, rich=False))
out["A1_momentum_integrated"] = a1
print("A1 conv", a1["convergence_alpha0.8_cov"], flush=True)

# ---------------------------------------------------------------- A2
prm = md.Params(a_D=2.0, **SET_A)
NS = 3100


def obs(St):
    return 0.5 * (1 + St[:, 0, 2]), 0.5 * np.hypot(St[:, 0, 0], St[:, 0, 1])


def markov_obs(T1, T2):
    p = md.Params(a_D=2.0, **{**SET_A, "T1": T1, "T2": T2})
    x, S = sv.solve_markov(p, X0, XF, NS, rtol=1e-9, atol=1e-11)
    return x, obs(sv.inst_from_fixed(x, S, p))


# Markovian excitation as a function of the dephasing time (T1 fixed)
T2grid = np.geomspace(0.05, 80.0, 65)
scan = [markov_obs(SET_A["T1"], T2)[1][0] for T2 in T2grid]
n_res_T2 = np.array([n[-1] for n in scan])
n_peak_T2 = np.array([n.max() for n in scan])
a2 = {"mismatch": "Z = (1/2) [ int (n_f - n_M)^2 / int n_f^2 + int (|p_f| - |p_M|)^2 / int |p_f|^2 ]"
                  " on omega t in [-15, 16]; single-observable fits use the corresponding term",
      "markov_scan": dict(omega_T2=T2grid.tolist(), n_res=n_res_T2.tolist(),
                          n_peak=n_peak_T2.tolist(),
                          T2_of_max_n_res=float(T2grid[n_res_T2.argmax()]),
                          max_n_res=float(n_res_T2.max()),
                          T2_of_max_n_peak=float(T2grid[n_peak_T2.argmax()]),
                          max_n_peak=float(n_peak_T2.max())),
      "rows": {}}
for alpha in (0.95, 0.9, 0.8, 0.7):
    x, Sc = sv.population("fixed", alpha, prm, X0, XF, NS)
    _, Sf = sv.population("fixed", alpha, prm, X0, XF, 2 * NS)
    St = Sf[::2] + (Sf[::2] - Sc) / 3.0        # Richardson extrapolation on the grid of NS steps
    n_f, p_f = obs(St)

    def terms(T1, T2):
        _, (n_m, p_m) = markov_obs(T1, T2)
        zn = np.trapezoid((n_f - n_m) ** 2, x) / np.trapezoid(n_f ** 2, x)
        zp = np.trapezoid((p_f - p_m) ** 2, x) / np.trapezoid(p_f ** 2, x)
        return zn, zp, n_m

    def cost1(logT2, which):
        zn, zp, _ = terms(SET_A["T1"], np.exp(logT2))
        return {"both": 0.5 * (zn + zp), "n": zn, "p": zp}[which]

    row = {}
    for which in ("both", "n", "p"):
        r = minimize_scalar(cost1, bounds=(np.log(0.05), np.log(80.0)), args=(which,),
                            method="bounded", options=dict(xatol=1e-3))
        zn, zp, n_m = terms(SET_A["T1"], np.exp(r.x))
        row[which] = dict(T2_eff=float(np.exp(r.x)), mismatch=float(r.fun), Z_n=float(zn),
                          Z_p=float(zp), n_peak_surrogate=float(n_m.max()),
                          n_res_surrogate=float(n_m[-1]))

    def cost2(v):                              # v = (log T1, log T2); enforce T2 <= 2 T1
        T1, T2 = np.exp(v)
        if T2 > 2.0 * T1 or T1 > 1e4 or T2 < 0.02:
            return 10.0
        zn, zp, _ = terms(T1, T2)
        return 0.5 * (zn + zp)

    r2 = minimize(cost2, [np.log(SET_A["T1"]), np.log(row["both"]["T2_eff"])],
                  method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-7, maxiter=400))
    T1e, T2e = np.exp(r2.x)
    zn, zp, n_m = terms(T1e, T2e)
    row["both_T1_free"] = dict(T1_eff=float(T1e), T2_eff=float(T2e), mismatch=float(r2.fun),
                               Z_n=float(zn), Z_p=float(zp), n_peak_surrogate=float(n_m.max()),
                               n_res_surrogate=float(n_m[-1]), converged=bool(r2.success))
    row["mismatch_with_nominal_T2"] = float(cost1(np.log(SET_A["T2"]), "both"))
    row["n_peak_fractional"] = float(n_f.max())
    row["n_res_fractional"] = float(n_f[-1])
    a2["rows"][str(alpha)] = row
    print("A2", alpha, row, flush=True)
out["A2_markov_surrogate"] = a2

# ---------------------------------------------------------------- A3
a3 = {}
for tm in (0.5, 1.0, 2.0):
    p = md.Params(a_D=2.0, **{**SET_A, "tau_m": tm})
    a3[f"tau_m_{tm:g}"] = {tag: float(nres_rich(m, 0.8, p)[0]) for m, tag in MODELS}
out["A3_tau_m_sensitivity_alpha0.8"] = a3
print("A3", a3, flush=True)

# ---------------------------------------------------------------- A4
# gauge dependence of the reduced model N1: eigenvectors rephased by
# Phi(t) = f * int (2 eps/hbar) dt'  (f = 0: real gauge; f = 1: dynamical-phase gauge)
a4 = {}
for alpha in (0.9, 0.8, 0.7):
    row = {}
    for f in (0.0, 0.5, 1.0):
        _, S = sv.solve_naive_gauge(alpha, prm, X0, XF, N, gauge_fraction=f)
        row[f"gauge_fraction_{f:g}"] = float(0.5 * (1 + S[-1, 0, 2]))
    a4[str(alpha)] = row
    print("A4", alpha, row, flush=True)
out["A4_N1_gauge_dependence"] = a4

# ---------------------------------------------------------------- A5
a5 = {"note": "Richardson-extrapolated from N = 12400 and 6200", "rows": {}}
for alpha in (1.0, 0.995, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
    a5["rows"][str(alpha)] = {tag: float(nres_rich(m, alpha, prm, 12400)[0]) for m, tag in MODELS}
    print("A5", alpha, a5["rows"][str(alpha)], flush=True)
out["A5_residual_vs_alpha"] = a5

# ---------------------------------------------------------------- A6
a6 = {"alpha": 0.8, "rows": {}}
XL, NL = 60.0, 15000
tr = {}
x, S = sv.population("fixed", 1.0, prm, X0, XL, NL)
tr["markov"] = 0.5 * (1 + S[:, 0, 2])
for m, tag in MODELS:
    x, S = sv.population(m, 0.8, prm, X0, XL, NL)
    tr[tag] = 0.5 * (1 + S[:, 0, 2])
for tobs in (8.0, 16.0, 30.0, 60.0):
    i = int(np.argmin(np.abs(x - tobs)))
    a6["rows"][f"{tobs:g}"] = {k: float(v[i]) for k, v in tr.items()}
out["A6_observation_time"] = a6
print("A6", a6, flush=True)

(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data" / "tables.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
print("written data/tables.json")
