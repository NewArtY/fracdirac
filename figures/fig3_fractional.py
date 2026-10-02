"""Fig. 3 -- fractional-memory response for several orders alpha (set A, a_D = 2, k = 0).

The Caputo equation is integrated in the fixed basis (BDF2 convolution quadrature) and the
result is projected onto the instantaneous basis. A second run with half the number of
steps provides the temporal-discretization error recorded in the metadata."""
from common import *  # noqa: F401,F403
import pandas as pd
import matplotlib.pyplot as plt

NAME = "fig3_fractional"
ALPHAS = [1.0, 0.95, 0.9, 0.8, 0.7]
XR = (-15.0, 60.0)
N = 15000                   # delta x = 0.005


def run(alpha, N):
    prm = md.Params(a_D=2.0, **SET_A)
    x, St = sv.population("fixed", alpha, prm, XR[0], XR[1], N)
    return x, St[:, 0]


def compute():
    rows, meta = [], dict(a_D=2.0, kappa=0.0, phi_cep=0.0, **SET_A, x0=XR[0], xf=XR[1], N=N,
                          scheme="bdf2", per_alpha={})
    ref = None
    for al in ALPHAS:
        x, St = run(al, N)
        n_c, p = md.observables_inst(St)
        lam = -np.gradient(np.log(np.maximum(n_c, 1e-300)), x)          # effective rate, n_c^eq = 0
        lam[n_c < 1e-8] = np.nan                                         # undefined before the pulse
        if al == 1.0:
            ref = np.unwrap(np.angle(p))
        dphi = np.unwrap(np.angle(p)) - ref
        err = None
        if al < 1.0:
            _, Sc = run(al, N // 2)
            err = float(np.abs(Sc - St[::2]).max())
        i16 = np.argmin(np.abs(x - 16.0))
        meta["per_alpha"][str(al)] = dict(
            n_c_peak=float(n_c.max()), n_c_at_16=float(n_c[i16]), n_c_at_60=float(n_c[-1]),
            abs_p_peak=float(np.abs(p).max()), abs_p_at_16=float(np.abs(p[i16])),
            lambda_eff_at_30=float(lam[np.argmin(np.abs(x - 30.0))]),
            lambda_eff_at_60=float(lam[-2]),
            max_bloch_norm=float(np.linalg.norm(St, axis=1).max()),
            max_diff_N_vs_N_over_2=err)
        for i in range(0, len(x), 5):
            rows.append((al, x[i], n_c[i], np.abs(p[i]), lam[i], dphi[i] / np.pi,
                         np.linalg.norm(St[i])))
    pd.DataFrame(rows, columns=["alpha", "omega_t", "n_c", "abs_p", "lambda_eff_over_omega",
                                "dphase_over_pi", "bloch_norm"]).to_csv(
        DATA / f"{NAME}.csv", index=False, lineterminator="\n")
    write_meta(NAME, meta)


def plot():
    style.setup()
    d = pd.read_csv(DATA / f"{NAME}.csv")
    cols = {1.0: style.INK, 0.95: style.BLUES[0], 0.9: style.BLUES[1], 0.8: style.BLUES[2],
            0.7: style.BLUES[3]}
    lss = {1.0: ":", 0.95: "-", 0.9: "-", 0.8: "-", 0.7: "-"}
    fig, ax = plt.subplots(2, 2, figsize=(style.COL2, 0.70 * style.COL2), constrained_layout=True)
    for al in ALPHAS:
        g = d[np.isclose(d.alpha, al)]
        lab = r"$\alpha=1$" if al == 1.0 else rf"$\alpha={al:g}$"
        kw = dict(color=cols[al], ls=lss[al], label=lab, lw=1.2)
        w = g[(g.omega_t >= -8) & (g.omega_t <= 16)]
        ax[0, 0].plot(w.omega_t, w.n_c, **kw)
        ax[0, 1].semilogy(w.omega_t, w.abs_p, **kw)
        t = g[g.omega_t >= 12]
        ax[1, 0].semilogy(t.omega_t, t.n_c, **kw)
        ax[1, 1].plot(t.omega_t, t.lambda_eff_over_omega, **kw)
    ax[0, 0].set_ylabel(r"Population $n_{c,k}$"); ax[0, 0].set_xlabel(r"$\omega t$")
    h, l = ax[0, 0].get_legend_handles_labels(); ax[1, 0].legend(h, l, loc="lower left", fontsize=8, labelspacing=0.25)
    ax[0, 1].set_ylabel(r"Coherence $|p_k|$"); ax[0, 1].set_xlabel(r"$\omega t$")
    ax[0, 1].set_ylim(1e-4, 1)
    ax[1, 0].set_ylabel(r"Post-pulse population $n_{c,k}$"); ax[1, 0].set_xlabel(r"$\omega t$")
    ax[1, 1].axhline(1 / SET_A["T1"], color=style.NEUTRAL, lw=0.6, ls="--")
    ax[1, 1].text(59.5, 1 / SET_A["T1"] + 0.0015, r"$1/(\omega T_1)$", ha="right", fontsize=8,
                  color=style.INK2)
    ax[1, 1].set_ylabel(r"Effective rate $\lambda_{\mathrm{eff}}/\omega$")
    ax[1, 1].set_xlabel(r"$\omega t$"); ax[1, 1].set_ylim(0, 0.05)
    for a, s in zip(ax.flat, "abcd"):
        style.panel_label(a, f"({s})")
    style.save(fig, "Fig3")


if __name__ == "__main__":
    if not plot_only():
        compute()
    plot()
