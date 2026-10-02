"""Fig. 1 -- field-driven kinematics in the rapidity representation (set A, k = 0)."""
from common import *  # noqa: F401,F403
import pandas as pd
import matplotlib.pyplot as plt

NAME = "fig1_kinematics"


def compute():
    prm = md.Params(a_D=2.0, **SET_A)
    x = np.linspace(-10.0, 10.0, 2001)
    q = np.array([md.q_of_x(xi, prm)[0] for xi in x])
    dq = np.array([md.dq_of_x(xi, prm)[0] for xi in x])
    theta = np.arcsinh(q)
    gp, gm = np.exp(-theta), np.exp(theta)          # gamma^+ = e^{-theta}, gamma^- = e^{+theta}
    eps, beta = 0.5 * (gm + gp), (gm - gp) / (gm + gp)
    df = pd.DataFrame({
        "omega_t": x, "A_over_A0": q / 2.0, "F_over_Fmax": -dq / np.abs(dq).max(),
        "sinh_theta": q, "theta": theta, "eps_over_Delta": eps, "beta": beta,
        "chi": np.arctan(q), "dchi_domega_t": dq / (1 + q * q),
        "mass_shell_residual": eps ** 2 - q ** 2 - 1.0, "gamma_product_minus_1": gp * gm - 1.0})
    df.to_csv(DATA / f"{NAME}.csv", index=False, lineterminator="\n")
    write_meta(NAME, dict(a_D=2.0, kappa=0.0, phi_cep=0.0, **SET_A,
                          max_abs_mass_shell_residual=float(np.abs(df.mass_shell_residual).max()),
                          max_abs_beta=float(np.abs(beta).max()),
                          theta_max=float(np.abs(theta).max())))


def plot():
    style.setup()
    d = pd.read_csv(DATA / f"{NAME}.csv")
    fig, ax = plt.subplots(2, 2, figsize=(style.COL2, 0.66 * style.COL2), sharex=True,
                           constrained_layout=True)
    a = ax[0, 0]
    a.plot(d.omega_t, d.A_over_A0, color=style.CAT[0], label=r"$A/A_0$")
    a.plot(d.omega_t, d.F_over_Fmax, color=style.CAT[1], ls="--", label=r"$F/F_{\max}$")
    a.set_ylabel("Normalized field"); a.legend(loc="upper right", ncol=1)
    a = ax[0, 1]
    a.plot(d.omega_t, d.theta, color=style.CAT[0], label=r"rapidity $\theta_k$")
    a.plot(d.omega_t, d.chi, color=style.CAT[1], ls="--", label=r"mixing angle $\chi_k$")
    a.axhline(0, color=style.GRID, lw=0.6, zorder=0)
    a.set_ylabel(r"$\theta_k$, $\chi_k$"); a.legend(loc="upper right")
    a = ax[1, 0]
    a.plot(d.omega_t, d.eps_over_Delta, color=style.CAT[0])
    a.axhline(1, color=style.NEUTRAL, lw=0.7, ls=":")
    a.set_ylabel(r"$\varepsilon_k/\Delta=\cosh\theta_k$"); a.set_xlabel(r"$\omega t$")
    a = ax[1, 1]
    a.plot(d.omega_t, d.beta, color=style.CAT[0])
    for y in (-1, 1):
        a.axhline(y, color=style.NEUTRAL, lw=0.7, ls=":")
    a.set_ylabel(r"$\beta_k=\tanh\theta_k$"); a.set_xlabel(r"$\omega t$")
    a.set_ylim(-1.08, 1.08)
    for a, s in zip(ax.flat, "abcd"):
        style.panel_label(a, f"({s})")
    style.save(fig, "Fig1")


if __name__ == "__main__":
    if not plot_only():
        compute()
    plot()
