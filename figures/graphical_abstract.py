"""Graphical abstract: two-time transport of the band basis inside the fractional memory
integral (schematic, left) and its consequence for the gap dependence of the residual
excitation (data of fig6_adiabatic.py, right)."""
from common import *  # noqa: F401,F403
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch


def plot():
    style.setup()
    g = pd.read_csv(DATA / "fig6_adiabatic_gap.csv")
    fig = plt.figure(figsize=(132.8 * style.MM, 53.1 * style.MM))
    # ---------------- left: schematic
    a = fig.add_axes([0.01, 0.04, 0.40, 0.92])
    a.set_xlim(-1.25, 1.25); a.set_ylim(-0.35, 1.45); a.set_aspect("equal"); a.axis("off")
    a.add_patch(Arc((0, 0), 2, 2, theta1=0, theta2=180, color=style.GRID, lw=1.0))
    for ang, lab, col in ((62, r"$\hat{\mathbf{h}}_k(t')$", style.NEUTRAL),
                          (118, r"$\hat{\mathbf{h}}_k(t)$", style.CAT[0])):
        x, y = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        a.add_patch(FancyArrowPatch((0, 0), (x, y), arrowstyle="-|>", mutation_scale=9, lw=1.4,
                                    color=col, shrinkA=0, shrinkB=0))
        a.text(1.17 * x, 1.17 * y, lab, ha="center", va="center", fontsize=8, color=style.INK)
    a.add_patch(Arc((0, 0), 0.9, 0.9, theta1=62, theta2=118, color=style.CAT[1], lw=1.2))
    a.text(0, 0.60, r"$\delta\chi_k$", ha="center", fontsize=8, color=style.INK)
    a.text(0, -0.14, "instantaneous basis rotates during the pulse", ha="center", fontsize=6.5,
           color=style.INK2)
    a.text(0, 1.38, r"memory kernel $\propto (t-t')^{-\alpha}\;R_k(t,t')\,[\,\cdot\,]\,R_k^{\dagger}(t,t')$",
           ha="center", fontsize=7.2, color=style.INK)
    a.text(0, -0.31, r"$R_k(t,t')=U_k^{\dagger}(t)\,U_k(t')$", ha="center", fontsize=7.2,
           color=style.INK)
    # ---------------- right: data
    b = fig.add_axes([0.53, 0.20, 0.45, 0.72])
    b.loglog(g.gap_ratio, g.n_markov, color=style.INK, ls=":", lw=1.0, label=r"Markovian ($\alpha=1$)")
    b.loglog(g.gap_ratio, g.n_cov, color=style.CAT[0], label="fixed-basis (covariant) model")
    b.loglog(g.gap_ratio, g.n_N1, color=style.CAT[1], ls="--", label="reduced model N1")
    b.set_xlabel(r"Gap ratio $\Delta/(\hbar\omega)$", fontsize=7)
    b.set_ylabel("Residual excitation", fontsize=7)
    b.legend(loc="lower left", fontsize=6)
    b.set_title(r"$\alpha=0.9$, fixed $a_D$: gap-independent excitation", fontsize=7, pad=3)
    style.save(fig, "graphical_abstract", tight=False)     # exact size: 1569 x 627 px at 300 dpi
    png = ROOT / "output" / "graphical_abstract.png"                   # the journal asks for RGB, not RGBA
    Image.open(png).convert("RGB").save(png, dpi=(300, 300))


if __name__ == "__main__":
    plot()
