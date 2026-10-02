"""Fig. 6 -- adiabatic (large-gap) limit: gap-independent excitation by geometric memory.

(a) residual population versus the gap ratio Delta/(hbar omega) for the Markovian, covariant,
    N1 and N2 models (alpha = 0.9, a_D = 2, set A otherwise), together with the prediction of
    the scalar geometric-memory equation;
(b) adiabatic-limit residual population versus 1 - alpha: scalar equation, its first-order
    expansion, and the full covariant model at Delta/(hbar omega) = 20.
"""
from common import *  # noqa: F401,F403
import pandas as pd
import matplotlib.pyplot as plt

NAME = "fig6_adiabatic"
ALPHA0 = 0.9
GAPS = np.geomspace(0.5, 20.0, 33)
NU = np.array([0.3, 0.25, 0.23, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005])
NG = 24800                   # delta x = 0.00125 (resolves the fastest precession)
G_BIG = 20.0


def compute():
    base = {k: v for k, v in SET_A.items() if k != "g"}
    prm = md.Params(a_D=2.0, g=GAPS, **base)
    res = {"markov": n_res("fixed", 1.0, prm, XA, 620)[0]}
    conv = {}
    for m, tag in (("fixed", "cov"), ("naive", "N1"), ("notransport", "N2")):
        res[tag] = n_res(m, ALPHA0, prm, XA, NG)[0]
        conv[tag] = float(np.abs(n_res(m, ALPHA0, prm, XA, NG // 2)[0] - res[tag]).max())
        print(tag, conv[tag], flush=True)
    # Markovian surrogate: dephasing time fitted to the fractional reference dynamics at the
    # gap ratio of set A (analysis/tables.py, test A2), then applied unchanged to all gaps
    T2eff = json.loads((DATA / "tables.json").read_text())["A2_markov_surrogate"]["rows"][
        str(ALPHA0)]["both"]["T2_eff"]
    prm_s = md.Params(a_D=2.0, g=GAPS, **{**base, "T2": T2eff})
    res["surrogate"] = n_res("fixed", 1.0, prm_s, XA, 620)[0]
    p1 = md.Params(a_D=2.0, **SET_A)
    n_ad = float(sv.solve_adiabatic(ALPHA0, p1, *XA, NA)[1][-1, 0])
    pd.DataFrame({"gap_ratio": GAPS, "n_markov": res["markov"], "n_cov": res["cov"],
                  "n_N1": res["N1"], "n_N2": res["N2"], "n_markov_surrogate": res["surrogate"],
                  "n_adiabatic_scalar": n_ad}).to_csv(DATA / f"{NAME}_gap.csv", index=False, lineterminator="\n")

    pb = md.Params(a_D=2.0, g=G_BIG, **base)
    coef = float(sv.adiabatic_first_order(p1, *XA, NA // 2)[2][-1, 0])
    coef_fine = float(sv.adiabatic_first_order(p1, *XA, NA)[2][-1, 0])
    rows = []
    for nu in NU:
        scal = float(sv.solve_adiabatic(1 - nu, p1, *XA, NA)[1][-1, 0])
        full = float(n_res("fixed", 1 - nu, pb, XA, NG)[0][0])
        rows.append((nu, scal, nu * coef_fine, full))
        print(nu, rows[-1], flush=True)
    pd.DataFrame(rows, columns=["one_minus_alpha", "n_adiabatic_scalar", "n_first_order",
                                "n_cov_full_gap20"]).to_csv(DATA / f"{NAME}_nu.csv", index=False, lineterminator="\n")
    write_meta(NAME, dict(**base, a_D=2.0, kappa=0.0, phi_cep=0.0, x0=XA[0], x_obs=XA[1],
                          alpha_gap_scan=ALPHA0, T2_eff_surrogate=T2eff, N_gap_scan=NG, gap_ratio_full_model=G_BIG,
                          max_abs_change_N_vs_N_over_2=conv,
                          first_order_coefficient=coef_fine,
                          first_order_coefficient_half_grid=coef,
                          n_adiabatic_scalar_alpha0p9=n_ad))


def plot():
    style.setup()
    g = pd.read_csv(DATA / f"{NAME}_gap.csv")
    v = pd.read_csv(DATA / f"{NAME}_nu.csv")
    fig, ax = plt.subplots(1, 2, figsize=(style.COL2, 0.46 * style.COL2), constrained_layout=True)
    a = ax[0]
    a.loglog(g.gap_ratio, g.n_markov, color=style.INK, ls=":", lw=1.0, label=r"Markovian ($\alpha=1$)")
    a.loglog(g.gap_ratio, g.n_markov_surrogate, color=style.INK2, ls=(0, (4, 1.5, 1, 1.5)), lw=1.0,
             label=r"Markovian, $T_2=T_2^{\mathrm{eff}}$")
    a.loglog(g.gap_ratio, g.n_cov, color=style.CAT[0], label="covariant")
    a.loglog(g.gap_ratio, g.n_N1, color=style.CAT[1], ls="--", label="N1")
    a.loglog(g.gap_ratio, g.n_N2, color=style.CAT[2], ls="-.", label="N2")
    a.axhline(g.n_adiabatic_scalar[0], color=style.INK2, lw=0.7, ls=(0, (1, 2)))
    a.text(19.5, g.n_adiabatic_scalar[0] * 1.25, "adiabatic limit (scalar equation)", fontsize=8,
           ha="right",
           color=style.INK2)
    a.set_xlabel(r"Gap ratio $\Delta/(\hbar\omega)$")
    a.set_ylabel(r"Residual population $n^{\mathrm{res}}_{c,k}$")
    a.legend(loc="lower left")
    a = ax[1]
    a.loglog(v.one_minus_alpha, v.n_adiabatic_scalar, color=style.CAT[0],
             label="scalar equation")
    a.loglog(v.one_minus_alpha, v.n_first_order, color=style.INK2, ls=":", lw=1.0,
             label=r"first order in $1-\alpha$")
    a.loglog(v.one_minus_alpha, v.n_cov_full_gap20, "o", color=style.CAT[0], ms=3.5, mfc="white",
             mew=0.9, label=r"covariant model, $\Delta/(\hbar\omega)=20$")
    a.set_xlabel(r"$1-\alpha$")
    a.set_ylabel(r"Adiabatic-limit $n^{\mathrm{res}}_{c,k}$")
    a.legend(loc="upper left")
    for a, lab in zip(ax, "ab"):
        style.panel_label(a, f"({lab})", dx=-0.16)
    style.save(fig, "Fig6")


if __name__ == "__main__":
    if not plot_only():
        compute()
    plot()
