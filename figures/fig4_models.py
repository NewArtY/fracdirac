"""Fig. 4 -- reference (fixed-basis = covariant) model versus the reduced models N1 and N2.

(a) n_c(t) for the Markovian, covariant, N1 (local connection) and N2 (identity transport)
    models at alpha = 0.8, a_D = 2 (set A, k = 0);
(b) signed differences n_c^X - n_c^cov together with the rotation rate |d chi/d(omega t)|;
(c) relative residual discrepancy eta_X versus theta_max for several alpha (model N1),
    Richardson-extrapolated in the time step;
(d) maximum over theta_max of the absolute residual difference versus 1 - alpha (both models),
    demonstrating the linear vanishing in the Markovian limit.
"""
from common import *  # noqa: F401,F403
import pandas as pd
import matplotlib.pyplot as plt

NAME = "fig4_models"
ALPHA0 = 0.8
THETA = np.linspace(0.05, 2.4, 48)
ALPHAS_C = [0.95, 0.9, 0.8, 0.7]
NU = np.array([0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])      # 1 - alpha for panel (d)


def compute():
    prm = md.Params(a_D=2.0, **SET_A)
    x, Sm = sv.population("fixed", 1.0, prm, *XA, NA)
    tr = {"markov": Sm[:, 0, 2]}
    for m in ("fixed", "naive", "notransport"):
        _, St = sv.population(m, ALPHA0, prm, *XA, NA)
        tr[m] = St[:, 0, 2]
    # explicit two-time transporter (independent implementation) on a coarser grid
    _, Sc = sv.population("covariant", ALPHA0, prm, *XA, NA // 4)
    _, Sf = sv.population("fixed", ALPHA0, prm, *XA, NA // 4)
    cov_defect = float(np.abs(Sc - Sf).max())
    dchi = np.array([md.dchi_of_x(xi, prm)[0] for xi in x])
    pd.DataFrame({"omega_t": x, "n_markov": 0.5 * (1 + tr["markov"]),
                  "n_cov": 0.5 * (1 + tr["fixed"]), "n_N1": 0.5 * (1 + tr["naive"]),
                  "n_N2": 0.5 * (1 + tr["notransport"]), "dchi": dchi}
                 ).iloc[::2].to_csv(DATA / f"{NAME}_traces.csv", index=False, lineterminator="\n")

    prm_s = md.Params(a_D=np.sinh(THETA), **SET_A)
    rows = []
    for al in sorted(set(ALPHAS_C) | set(np.round(1 - NU, 6))):
        # Richardson extrapolation of the second-order scheme (N and N/2 steps)
        r = {m: (4.0 * n_res(m, al, prm_s, XA, NA)[0] - n_res(m, al, prm_s, XA, NA // 2)[0]) / 3.0
             for m in ("fixed", "naive", "notransport")}
        for i, th in enumerate(THETA):
            rows.append((al, th, r["fixed"][i], r["naive"][i], r["notransport"][i]))
    scan = pd.DataFrame(rows, columns=["alpha", "theta_max", "n_cov", "n_N1", "n_N2"])
    scan.to_csv(DATA / f"{NAME}_scan.csv", index=False, lineterminator="\n")

    # temporal convergence of the scan at the two extreme orders
    conv = {}
    for al in (0.7, 0.95):
        for m in ("fixed", "naive", "notransport"):
            a = n_res(m, al, prm_s, XA, NA)[0]
            b = n_res(m, al, prm_s, XA, NA // 2)[0]
            conv[f"{m}_alpha{al}"] = float(np.abs(a - b).max())
    g = scan[np.isclose(scan.alpha, ALPHA0)]
    i2 = np.argmin(np.abs(g.theta_max.values - np.arcsinh(2.0)))
    write_meta(NAME, dict(**SET_A, kappa=0.0, phi_cep=0.0, x0=XA[0], x_obs=XA[1], N=NA,
                          alpha_traces=ALPHA0, covariance_defect_explicit_transporter=cov_defect,
                          max_abs_change_N_vs_N_over_2=conv,
                          n_res_at_aD2_alpha08=dict(
                              markov=float(0.5 * (1 + tr["markov"][-1])),
                              cov=float(0.5 * (1 + tr["fixed"][-1])),
                              N1=float(0.5 * (1 + tr["naive"][-1])),
                              N2=float(0.5 * (1 + tr["notransport"][-1])))))


def plot():
    style.setup()
    t = pd.read_csv(DATA / f"{NAME}_traces.csv")
    s = pd.read_csv(DATA / f"{NAME}_scan.csv")
    fig, ax = plt.subplots(2, 2, figsize=(style.COL2, 0.74 * style.COL2), constrained_layout=True)
    w = t[t.omega_t >= -8]
    a = ax[0, 0]
    a.plot(w.omega_t, w.n_markov, color=style.INK, ls=":", lw=1.0, label=r"Markovian ($\alpha=1$)")
    a.plot(w.omega_t, w.n_cov, color=style.CAT[0], label="covariant")
    a.plot(w.omega_t, w.n_N1, color=style.CAT[1], ls="--", label="N1")
    a.plot(w.omega_t, w.n_N2, color=style.CAT[2], ls="-.", label="N2")
    a.set_xlabel(r"$\omega t$"); a.set_ylabel(r"Population $n_{c,k}$")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
             columnspacing=1.0, handlelength=1.6)
    a = ax[0, 1]
    a.fill_between(w.omega_t, 0, 0.1 * np.abs(w.dchi) / np.abs(w.dchi).max(), color=style.GRID,
                   lw=0, label=r"$|\dot\chi_k|$ (arb. units)")
    a.fill_between(w.omega_t, 0, -0.1 * np.abs(w.dchi) / np.abs(w.dchi).max(), color=style.GRID,
                   lw=0)
    a.plot(w.omega_t, w.n_N1 - w.n_cov, color=style.CAT[1], ls="--", label=r"$\delta n^{\mathrm{N1}}_{c,k}$")
    a.plot(w.omega_t, w.n_N2 - w.n_cov, color=style.CAT[2], ls="-.", label=r"$\delta n^{\mathrm{N2}}_{c,k}$")
    a.axhline(0, color=style.INK2, lw=0.5)
    a.set_xlabel(r"$\omega t$"); a.set_ylabel(r"Difference $\delta n^{X}_{c,k}$")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
             columnspacing=1.0, handlelength=1.6)
    a = ax[1, 0]
    cols = {0.95: style.BLUES[0], 0.9: style.BLUES[1], 0.8: style.BLUES[2], 0.7: style.BLUES[3]}
    for al in ALPHAS_C:
        g = s[np.isclose(s.alpha, al)]
        a.plot(g.theta_max, np.abs(g.n_N1 - g.n_cov) / g.n_cov, color=cols[al],
               label=rf"$\alpha={al:g}$")
    a.set_xlabel(r"Maximum rapidity $\theta_{\max}$")
    a.set_ylabel(r"Relative discrepancy $\eta_{\mathrm{N1}}$")
    a.set_ylim(0, 1); a.legend(loc="lower left", ncol=2)
    a = ax[1, 1]
    d1, d2 = [], []
    for nu in NU:
        g = s[np.isclose(s.alpha, 1 - nu)]
        d1.append(np.abs(g.n_N1 - g.n_cov).max()); d2.append(np.abs(g.n_N2 - g.n_cov).max())
    a.loglog(NU, d1, "o--", color=style.CAT[1], ms=3.5, label="N1")
    a.loglog(NU, d2, "s-.", color=style.CAT[2], ms=3.5, label="N2")
    a.loglog(NU, d1[-1] * NU / NU[-1], color=style.INK2, lw=0.7, ls=":", label=r"$\propto 1-\alpha$")
    a.set_xlabel(r"$1-\alpha$")
    a.set_ylabel(r"Largest $|\delta n^{\mathrm{res},X}_{c,k}|$")
    a.legend(loc="upper left")
    for a, lab in zip(ax.flat, "abcd"):
        style.panel_label(a, f"({lab})")
    style.save(fig, "Fig4")


if __name__ == "__main__":
    if not plot_only():
        compute()
    plot()
