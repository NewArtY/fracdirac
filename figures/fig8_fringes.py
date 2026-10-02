"""Fig. 8 -- position and visibility of the CEP interference fringe versus fractional order.

Fixed field parameter a_D (set B); the covariant model and the reduced model N1 are both
integrated for every alpha. The fringe maximum is located on the trigonometric interpolant
of n_res(phi) and followed continuously from alpha = 1 downwards."""
from common import *  # noqa: F401,F403
import pandas as pd
import matplotlib.pyplot as plt

NAME = "fig8_fringes"
A_D = 2.0
ALPHA = np.round(np.linspace(1.0, 0.70, 31), 4)
PHI = np.linspace(0, 2 * np.pi, 64, endpoint=False)
NFINE = 4096


def trig_interp(y, nfine=NFINE):
    Y = np.fft.rfft(y)
    if len(y) % 2 == 0:
        Y[-1] *= 0.5                # split the Nyquist coefficient when zero-padding
    return np.fft.irfft(Y, n=nfine) * (nfine / len(y))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def analyse(profiles):
    """profiles: (n_alpha, n_phi), first row alpha = 1. Returns phi_max, visibility, mean, max, min."""
    fine = np.linspace(0, 2 * np.pi, NFINE, endpoint=False)
    out, prev = [], None
    for y in profiles:
        yf = trig_interp(y)
        loc = np.where((yf >= np.roll(yf, 1)) & (yf > np.roll(yf, -1)))[0]
        if prev is None:
            j = loc[np.argmax(yf[loc])]
        else:
            j = loc[np.argmin(np.abs(wrap(fine[loc] - prev)))]
        prev = fine[j]
        out.append((prev, (yf.max() - yf.min()) / (yf.max() + yf.min()), yf.mean(), yf.max(),
                    yf.min()))
    return np.array(out)


def profiles(model, N, rich=True):
    """n_res(phi) for all alpha; Richardson-extrapolated in the time step if rich."""
    prm = md.Params(a_D=A_D, phi_cep=PHI, kappa=KAPPA_B, **SET_B)
    f = n_res_rich if rich else n_res
    return np.array([f(model if al < 1 else "fixed", al, prm, XB, N)[0] for al in ALPHA])


def compute():
    res = {}
    for m, tag in (("fixed", "cov"), ("naive", "N1")):
        pr = profiles(m, NB)
        res[tag] = (pr, analyse(pr))
        print(tag, "done", flush=True)
    half = analyse(profiles("fixed", NB, rich=False))     # un-extrapolated N-step result
    rows = []
    for i, al in enumerate(ALPHA):
        r = [al]
        for tag in ("cov", "N1"):
            a = res[tag][1][i]
            r += [a[0], wrap(a[0] - res[tag][1][0][0]), a[1], a[2], a[3], a[4]]
        rows.append(r)
    cols = ["alpha"] + [f"{q}_{t}" for t in ("cov", "N1")
                        for q in ("phi_max", "dphi_max", "visibility", "mean", "n_max", "n_min")]
    pd.DataFrame(rows, columns=cols).to_csv(DATA / f"{NAME}.csv", index=False, lineterminator="\n")
    prof = pd.DataFrame(res["cov"][0].T, columns=[f"cov_alpha_{al:g}" for al in ALPHA])
    for i, al in enumerate(ALPHA):
        prof[f"N1_alpha_{al:g}"] = res["N1"][0][i]
    prof.insert(0, "phi_cep", PHI)
    prof.to_csv(DATA / f"{NAME}_profiles.csv", index=False, lineterminator="\n")
    full = res["cov"][1]
    write_meta(NAME, dict(**SET_B, kappa=KAPPA_B, a_D=A_D, x0=XB[0], x_obs=XB[1], N=NB,
                          n_phi=len(PHI),
                          max_change_phi_max_by_extrapolation_rad=float(
                              np.abs(wrap(full[:, 0] - half[:, 0])).max()),
                          max_change_visibility_by_extrapolation=float(
                              np.abs(full[:, 1] - half[:, 1]).max()),
                          phi_max_markov_rad=float(full[0, 0]),
                          visibility_markov=float(full[0, 1])))


def plot():
    style.setup()
    d = pd.read_csv(DATA / f"{NAME}.csv").sort_values("alpha")
    pr = pd.read_csv(DATA / f"{NAME}_profiles.csv")
    fig, ax = plt.subplots(1, 3, figsize=(style.COL2, 0.40 * style.COL2), constrained_layout=True)
    a = ax[0]
    sel = [(1.0, style.INK, ":"), (0.95, style.BLUES[0], "-"), (0.9, style.BLUES[1], "-"),
           (0.8, style.BLUES[2], "-"), (0.7, style.BLUES[3], "-")]
    for al, c, ls in sel:
        y = pr[f"cov_alpha_{al:g}"].values
        a.plot(np.r_[pr.phi_cep, 2 * np.pi] / np.pi, np.r_[y, y[0]], color=c, ls=ls, lw=1.1,
               label=rf"$\alpha={al:g}$")
    a.set_xlabel(r"$\varphi/\pi$"); a.set_ylabel(r"$n^{\mathrm{res,cov}}_{c,k}$")
    a.set_xticks([0, 0.5, 1, 1.5, 2]); a.legend(ncol=2, loc="upper center", fontsize=8, columnspacing=0.8, handlelength=1.4)
    a.set_ylim(top=a.get_ylim()[1] * 1.3)
    a = ax[1]
    a.plot(d.alpha, d.dphi_max_cov / np.pi, color=style.CAT[0], label="covariant")
    a.plot(d.alpha, d.dphi_max_N1 / np.pi, color=style.CAT[1], ls="--", label="N1")
    a.axhline(0, color=style.INK2, lw=0.5)
    a.set_xlabel(r"Fractional order $\alpha$")
    a.set_ylabel(r"Fringe displacement $\delta\varphi_{\max}/\pi$"); a.legend()
    a = ax[2]
    a.plot(d.alpha, d.visibility_cov, color=style.CAT[0], label="covariant")
    a.plot(d.alpha, d.visibility_N1, color=style.CAT[1], ls="--", label="N1")
    a.set_xlabel(r"Fractional order $\alpha$"); a.set_ylabel(r"Fringe visibility $\mathcal{V}$")
    a.set_ylim(bottom=0); a.legend()
    for a, lab in zip(ax, "abc"):
        style.panel_label(a, f"({lab})", dx=-0.26)
    style.save(fig, "Fig8")


if __name__ == "__main__":
    if not plot_only():
        compute()
    plot()
