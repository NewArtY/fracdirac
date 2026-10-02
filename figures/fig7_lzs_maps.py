"""Fig. 7 -- residual population versus carrier-envelope phase and field amplitude (set B).

Covariant (fixed-basis) fractional model for alpha = 1, 0.98, 0.95, 0.9 on identical grids.
The N1 maps are stored as well (used for the discussion in the text)."""
from common import *  # noqa: F401,F403
import pandas as pd
import matplotlib.pyplot as plt

NAME = "fig7_lzs_maps"
ALPHAS = [1.0, 0.98, 0.95, 0.9]
PHI = np.linspace(0, 2 * np.pi, 48, endpoint=False)
AMP = np.linspace(0.5, 3.2, 37)


def compute():
    AA, PP = np.meshgrid(AMP, PHI, indexing="ij")
    prm = md.Params(a_D=AA.ravel(), phi_cep=PP.ravel(), kappa=KAPPA_B, **SET_B)
    rows, meta = [], dict(**SET_B, kappa=KAPPA_B, x0=XB[0], x_obs=XB[1], N=NB, scheme="bdf2",
                          per_alpha={})
    for al in ALPHAS:
        nc, err, nrm = n_res_rich("fixed", al, prm, XB, NB)
        n1 = nc if al == 1.0 else n_res_rich("naive", al, prm, XB, NB)[0]
        info = dict(n_min=float(nc.min()), n_max=float(nc.max()), max_bloch_norm=nrm,
                    max_abs_N1_minus_cov=float(np.abs(n1 - nc).max()),
                    max_rel_N1_minus_cov=float((np.abs(n1 - nc) / nc).max()),
                    richardson=bool(al < 1.0), max_error_estimate_before_extrapolation=err)
        meta["per_alpha"][str(al)] = info
        for i in range(prm.P):
            rows.append((al, AA.ravel()[i], PP.ravel()[i], nc[i], n1[i]))
        print(al, info, flush=True)
    pd.DataFrame(rows, columns=["alpha", "a_D", "phi_cep", "n_res_cov", "n_res_N1"]).to_csv(
        DATA / f"{NAME}.csv", index=False, lineterminator="\n")
    write_meta(NAME, meta)


def plot():
    style.setup()
    d = pd.read_csv(DATA / f"{NAME}.csv")
    maps = {al: d[np.isclose(d.alpha, al)].pivot(index="a_D", columns="phi_cep",
                                                 values="n_res_cov").values for al in ALPHAS}
    vmax = max(m.max() for m in maps.values())
    fig, ax = plt.subplots(2, 2, figsize=(style.COL2, 0.66 * style.COL2), sharex=True,
                           sharey=True, constrained_layout=True)
    for a, al, lab in zip(ax.flat, ALPHAS, "abcd"):
        z = np.c_[maps[al], maps[al][:, :1]]              # close the periodic axis
        m = a.pcolormesh(np.r_[PHI, 2 * np.pi] / np.pi, AMP, z, cmap=style.SEQ, vmin=0,
                         vmax=vmax, shading="gouraud", rasterized=True)
        a.set_title(rf"$\alpha={al:g}$", pad=3)
        a.set_xticks([0, 0.5, 1, 1.5, 2])
        style.panel_label(a, f"({lab})", dx=-0.12)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(True)
    for a in ax[1]:
        a.set_xlabel(r"Carrier-envelope phase $\varphi/\pi$")
    for a in ax[:, 0]:
        a.set_ylabel(r"Field parameter $a_D$")
    cb = fig.colorbar(m, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r"Residual population $n^{\mathrm{res}}_{c,k}$")
    style.save(fig, "Fig7")


if __name__ == "__main__":
    if not plot_only():
        compute()
    plot()
