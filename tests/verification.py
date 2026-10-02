"""Verification suite for the fractional Bloch solvers (results -> data/verification.json).

Run:  python tests/verification.py        (about 5 minutes)

Tests (numbered as in the verification table of the manuscript)
  1  constant generator: comparison with the exact Mittag-Leffler solution, observed orders
  2  covariance defect: explicit two-time transporter versus fixed-basis integration
  3  driven problem: self-convergence, observed order and absolute accuracy of the three
     fractional models; accuracy of the L1 scheme
  4  Markovian limit alpha -> 1 of the three fractional models
  5  zero-field stationarity of the equilibrium state
  6  independence of the lower terminal t_0 of the Caputo derivative
  7  weak-memory expansion of the fractional operator (first order in 1 - alpha), scalar case
  8  alpha = 1: fixed-basis versus instantaneous-basis (local connection) integration
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as Gamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fracdirac import model as md, solvers as sv  # noqa: E402
from fracdirac.mittag_leffler import ml_matrix_solution  # noqa: E402

SET_A = dict(s_p=3.0, g=1.5, T1=40.0, T2=20.0, tau_m=1.0)
X0, XF = -15.0, 16.0
out = {}


def eoc(e):
    e = np.asarray(e)
    return [float(v) for v in np.log2(e[:-1] / e[1:])]


def nres(model, alpha, prm, N, scheme="bdf2", x0=X0):
    _, St = sv.population(model, alpha, prm, x0, XF, N, scheme)
    return 0.5 * (1 + St[-1, :, 2])


# ---------------------------------------------------------------- test 1
t1 = {}
prm = md.Params(a_D=0.0, kappa=0.8, **SET_A)
M, b = md.gen_fixed(0.0, prm)
s0 = np.array([[0.0, 0.0, -1.0]])
Ns = [250, 500, 1000, 2000, 4000]
for alpha in (0.7, 0.9):
    exact = ml_matrix_solution(alpha, prm.tau_m ** (1 - alpha), M[0], b[0], s0[0], [2.0, 5.0])
    for scheme in ("l1", "bdf2"):
        errs = []
        for N in Ns:
            _, S = sv.solve_fixed(alpha, prm, 0.0, 5.0, N, scheme, s0=s0)
            i2 = int(round(0.4 * N))
            errs.append(float(max(np.abs(S[i2, 0] - exact[0]).max(),
                                  np.abs(S[-1, 0] - exact[1]).max())))
        t1[f"alpha{alpha}_{scheme}"] = dict(N=Ns, max_error=errs, observed_order=eoc(errs))
out["T1_mittag_leffler"] = t1
print("test 1", json.dumps(t1, indent=1), flush=True)

# ---------------------------------------------------------------- test 2
t2 = {}
prm = md.Params(a_D=np.array([0.5, 2.0, 3.0]), **SET_A)
for alpha in (0.7, 0.95):
    N = 1240
    x, S = sv.solve_fixed(alpha, prm, X0, XF, N, "bdf2")
    St = sv.inst_from_fixed(x, S, prm)
    _, Sc = sv.solve_covariant(alpha, prm, X0, XF, N, "bdf2")
    x, S1 = sv.solve_fixed(alpha, prm, X0, XF, N, "l1")
    St1 = sv.inst_from_fixed(x, S1, prm)
    _, Sc1 = sv.solve_covariant_l1_increments(alpha, prm, X0, XF, N)
    t2[f"alpha{alpha}"] = dict(N=N, bdf2_state_form=float(np.abs(Sc - St).max()),
                               l1_increment_form=float(np.abs(Sc1 - St1).max()))
out["T2_covariance_defect"] = t2
print("test 2", t2, flush=True)

# ---------------------------------------------------------------- test 3
t3 = {}
prm = md.Params(a_D=2.0, **SET_A)
Ns = [1550, 3100, 6200, 12400]
for alpha in (0.7, 0.9, 0.95, 0.98):
    for model in ("fixed", "naive", "notransport"):
        vals = [float(nres(model, alpha, prm, N)[0]) for N in Ns]
        d = np.abs(np.diff(vals))
        rich = float((4 * vals[-1] - vals[-2]) / 3)
        t3[f"alpha{alpha}_{model}"] = dict(
            N=Ns, n_res=vals, successive_differences=d.tolist(), observed_order=eoc(d),
            richardson=rich, error_vs_richardson=[abs(v - rich) for v in vals])
    vals = [float(nres("fixed", alpha, prm, N, "l1")[0]) for N in Ns]
    t3[f"alpha{alpha}_fixed_l1"] = dict(
        N=Ns, n_res=vals,
        error_vs_bdf2_richardson=[abs(v - t3[f"alpha{alpha}_fixed"]["richardson"]) for v in vals])
out["T3_driven_convergence"] = t3
print("test 3", json.dumps(t3, indent=1), flush=True)

# ---------------------------------------------------------------- test 4
# residual populations are Richardson-extrapolated from N = 6200 and 12400
t4 = {}
_, Sm = sv.population("fixed", 1.0, prm, X0, XF, 6200)
nm = float(0.5 * (1 + Sm[-1, 0, 2]))
for nu in (1e-2, 1e-3, 1e-4):
    t4[f"one_minus_alpha_{nu:g}"] = {
        m: abs(float((4 * nres(m, 1 - nu, prm, 12400)[0] - nres(m, 1 - nu, prm, 6200)[0]) / 3) - nm)
        for m in ("fixed", "naive", "notransport")}
out["T4_markov_limit_abs_diff_n_res"] = t4
out["T4_n_res_markov"] = nm
print("test 4", t4, flush=True)

# ---------------------------------------------------------------- test 5
prm0 = md.Params(a_D=0.0, kappa=np.array([0.0, 0.7]), **SET_A)
t5 = {}
x, S = sv.solve_fixed(0.8, prm0, X0, XF, 400)
t5["fixed"] = float(np.abs(S - S[0]).max())
for name, f in (("naive", sv.solve_naive), ("notransport", sv.solve_notransport),
                ("covariant", sv.solve_covariant)):
    x, S = f(0.8, prm0, X0, XF, 400)
    t5[name] = float(np.abs(S - S[0]).max())
out["T5_zero_field_drift"] = t5
print("test 5", t5, flush=True)

# ---------------------------------------------------------------- test 6
t6 = {}
for alpha in (0.7, 0.9):
    for model in ("fixed", "naive", "notransport"):
        a = nres(model, alpha, prm, 6200, x0=-15.0)[0]
        bb = nres(model, alpha, prm, 7200, x0=-20.0)[0]      # same step, earlier start
        t6[f"alpha{alpha}_{model}"] = abs(float(a - bb))
out["T6_lower_terminal_independence"] = t6
print("test 6", t6, flush=True)

# ---------------------------------------------------------------- test 7
def Ftest(tau):
    return np.cos(3.0 * tau) * np.exp(-tau / 5.0) + 0.3 * tau


t, tau_m, t7 = 2.0, 0.7, {}
EULER = 0.5772156649015329
first = ((EULER + np.log(t / tau_m)) * Ftest(t)
         + quad(lambda u: (Ftest(u) - Ftest(t)) / (t - u), 0, t, limit=200)[0])
for nu in (1e-1, 1e-2, 1e-3):
    lhs = tau_m ** (-nu) / Gamma(nu) * quad(Ftest, 0, t, weight="alg", wvar=(0.0, nu - 1.0))[0]
    t7[f"nu_{nu:g}"] = dict(exact=float(lhs), first_order=float(Ftest(t) + nu * first),
                            remainder=float(lhs - Ftest(t) - nu * first),
                            remainder_over_nu2=float((lhs - Ftest(t) - nu * first) / nu ** 2))
out["T7_weak_memory_expansion"] = t7
print("test 7", t7, flush=True)

# ---------------------------------------------------------------- test 8
x, Sa = sv.solve_markov(prm, X0, XF, 3100, basis="fixed")
_, Sb = sv.solve_markov(prm, X0, XF, 3100, basis="inst")
out["T8_markov_fixed_vs_instantaneous"] = float(np.abs(Sa - Sb).max())
print("test 8", out["T8_markov_fixed_vs_instantaneous"], flush=True)

(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data" / "verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
print("written data/verification.json")
