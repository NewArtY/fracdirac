"""Fig. 5 -- residual excitation in the (alpha, theta_max) plane (set A, k = 0).

(a) covariant (reference) model; (b), (c) signed differences of the reduced models N1 and N2
with respect to the reference, with contours of the relative discrepancy."""
from common import *  # noqa: F401,F403
import pandas as pd
import matplotlib.pyplot as plt

NAME = "fig5_phase_diagram"
ALPHA = np.round(np.linspace(0.70, 1.00, 25), 4)
THETA = np.linspace(0.05, 2.4, 48)


def compute():
    prm = md.Params(a_D=np.sinh(THETA), **SET_A)
    rows, maxnorm, maxerr = [], 0.0, 0.0
    for al in ALPHA:
        r = {}
        for m in ("fixed", "naive", "notransport"):
            r[m], err, nrm = n_res_rich(m, al, prm, XA, NA)
            maxnorm, maxerr = max(maxnorm, nrm), max(maxerr, err)
        for i, th in enumerate(THETA):
            rows.append((al, th, r["fixed"][i], r["naive"][i], r["notransport"][i]))
        print(al, flush=True)
    df = pd.DataFrame(rows, columns=["alpha", "theta_max", "n_cov", "n_N1", "n_N2"])
    df.to_csv(DATA / f"{NAME}.csv", index=False, lineterminator="\n")
    meta = dict(**SET_A, kappa=0.0, phi_cep=0.0, x0=XA[0], x_obs=XA[1], N=NA,
                max_bloch_norm=maxnorm, richardson=True,
                max_error_estimate_before_extrapolation=maxerr)
    for tag in ("N1", "N2"):
        rel = np.abs(df[f"n_{tag}"] - df.n_cov) / df.n_cov
        ab = np.abs(df[f"n_{tag}"] - df.n_cov)
        i, j = rel.idxmax(), ab.idxmax()
        meta[tag] = dict(max_rel=float(rel[i]), at_alpha=float(df.alpha[i]),
                         at_theta=float(df.theta_max[i]), n_cov_there=float(df.n_cov[i]),
                         max_abs=float(ab[j]), abs_at_alpha=float(df.alpha[j]),
                         abs_at_theta=float(df.theta_max[j]),
                         max_rel_where_n_cov_above_0p05=float(rel[df.n_cov > 0.05].max()),
                         max_abs_at_alpha_1=float(ab[np.isclose(df.alpha, 1.0)].max()))
    write_meta(NAME, meta)


def plot():
    style.setup()
    d = pd.read_csv(DATA / f"{NAME}.csv")
    piv = lambda c: d.pivot(index="theta_max", columns="alpha", values=c).values  # noqa: E731
    ncov, n1, n2 = piv("n_cov"), piv("n_N1"), piv("n_N2")
    fig, ax = plt.subplots(1, 3, figsize=(style.COL2, 0.42 * style.COL2), sharey=True,
                           constrained_layout=True)
    m0 = ax[0].pcolormesh(ALPHA, THETA, ncov, cmap=style.SEQ, vmin=0, shading="gouraud",
                          rasterized=True)
    cb = fig.colorbar(m0, ax=ax[0], location="top", fraction=0.06, pad=0.02)
    cb.set_label(r"$n^{\mathrm{res,cov}}_{c,k}$")
    lim = max(np.abs(n1 - ncov).max(), np.abs(n2 - ncov).max())
    for a, z, tag, lev in ((ax[1], n1, "N1", [0.2, 0.4, 0.6, 0.8]), (ax[2], n2, "N2", [0.1, 0.2, 0.4])):
        m = a.pcolormesh(ALPHA, THETA, z - ncov, cmap=style.DIV, vmin=-lim, vmax=lim,
                         shading="gouraud", rasterized=True)
        rel = np.abs(z - ncov) / ncov
        sel = ALPHA <= 0.951                    # contours are drawn away from the Markovian edge
        cs = a.contour(ALPHA[sel], THETA, rel[:, sel], levels=lev, colors=style.INK,
                       linewidths=0.6)
        a.clabel(cs, fmt="%g", fontsize=7, inline=True, inline_spacing=2)
        cb = fig.colorbar(m, ax=a, location="top", fraction=0.06, pad=0.02)
        cb.set_label(rf"$\delta n^{{\mathrm{{res}},\mathrm{{{tag}}}}}_{{c,k}}$")
    for a, lab in zip(ax, "abc"):
        a.set_xlabel(r"Fractional order $\alpha$")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(True)
        a.text(0.035, 0.045, f"({lab})", transform=a.transAxes, fontsize=9, fontweight="bold",
               va="bottom", ha="left",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax[0].set_ylabel(r"Maximum rapidity $\theta_{\max}$")
    style.save(fig, "Fig5")


if __name__ == "__main__":
    if not plot_only():
        compute()
    plot()
