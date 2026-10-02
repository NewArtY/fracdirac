"""Fig. 2 -- Markovian (alpha = 1) reference dynamics in the instantaneous basis (set A)."""
from common import *  # noqa: F401,F403
import pandas as pd
import matplotlib.pyplot as plt

NAME = "fig2_markov"


def compute():
    prm = md.Params(a_D=2.0, **SET_A)
    N = 3100
    x, S = sv.solve_markov(prm, XA[0], XA[1], N)
    St = sv.inst_from_fixed(x, S, prm)[:, 0]
    # independent integration in the instantaneous basis with the local connection
    _, S2 = sv.solve_markov(prm, XA[0], XA[1], N, basis="inst")
    n_c, p = md.observables_inst(St)
    df = pd.DataFrame({"omega_t": x, "s_x": St[:, 0], "s_y": St[:, 1], "s_z": St[:, 2],
                       "n_c": n_c, "abs_p": np.abs(p), "bloch_norm": np.linalg.norm(St, axis=1)})
    df.to_csv(DATA / f"{NAME}.csv", index=False, lineterminator="\n")
    write_meta(NAME, dict(a_D=2.0, kappa=0.0, phi_cep=0.0, **SET_A, x0=XA[0], x_obs=XA[1],
                          n_c_res=float(n_c[-1]), n_c_peak=float(n_c.max()),
                          abs_p_peak=float(np.abs(p).max()),
                          max_bloch_norm=float(df.bloch_norm.max()),
                          fixed_vs_instantaneous_basis_max_diff=float(np.abs(S - S2).max())))


def plot():
    style.setup()
    d = pd.read_csv(DATA / f"{NAME}.csv")
    fig, ax = plt.subplots(2, 2, figsize=(style.COL2, 0.66 * style.COL2), sharex=True,
                           constrained_layout=True)
    a = ax[0, 0]
    for c, lab, col, ls in (("s_x", r"$\tilde s_{x,k}$", style.CAT[0], "-"),
                            ("s_y", r"$\tilde s_{y,k}$", style.CAT[1], "--"),
                            ("s_z", r"$\tilde s_{z,k}$", style.CAT[2], "-.")):
        a.plot(d.omega_t, d[c], color=col, ls=ls, label=lab)
    a.set_ylabel("Bloch components"); a.set_ylim(-1.05, 1.3)
    a.legend(loc="upper left", ncol=3, columnspacing=1.2, handlelength=2.2)
    a = ax[0, 1]
    a.plot(d.omega_t, d.n_c, color=style.CAT[0]); a.set_ylabel(r"Population $n_{c,k}$")
    a = ax[1, 0]
    a.plot(d.omega_t, d.abs_p, color=style.CAT[0]); a.set_ylabel(r"Coherence $|p_k|$")
    a.set_xlabel(r"$\omega t$")
    a = ax[1, 1]
    a.plot(d.omega_t, d.bloch_norm, color=style.CAT[0])
    a.axhline(1, color=style.NEUTRAL, lw=0.7, ls=":")
    a.set_ylabel(r"Bloch-vector norm $\|\tilde{\mathbf{s}}_k\|$"); a.set_xlabel(r"$\omega t$")
    a.set_xlim(-10, 16)
    for a, s in zip(ax.flat, "abcd"):
        style.panel_label(a, f"({s})")
    style.save(fig, "Fig2")


if __name__ == "__main__":
    if not plot_only():
        compute()
    plot()
