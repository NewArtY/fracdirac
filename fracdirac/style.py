"""Shared Matplotlib style for the manuscript figures (print, single surface)."""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4
COL1, COL15, COL2 = 90 * MM, 140 * MM, 190 * MM          # Elsevier column widths

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
# categorical slots (fixed order; validated all-pairs for the first three)
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
NEUTRAL = "#52514e"
# ordinal single-hue ramp (light -> dark) for the fractional order alpha
BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b", "#071d3b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a5a4", "#e34948", "#8f1f1e"])


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Nimbus Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8.6, "ytick.labelsize": 8.6, "legend.fontsize": 8.6,
        "axes.linewidth": 0.6, "lines.linewidth": 1.3,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "figure.dpi": 150,
        "axes.grid": False, "legend.frameon": False,
    })


def panel_label(ax, s, dx=-0.14, dy=1.03):
    ax.text(dx, dy, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold")


def save(fig, name, outdir=None, tight=True):
    """Write name.pdf (vector) and name.png (300 dpi). tight=False keeps the exact figure size."""
    outdir = Path(outdir) if outdir else Path(__file__).resolve().parents[1] / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    kw = dict(facecolor="white", bbox_inches="tight", pad_inches=0.02) if tight else dict(
        facecolor="white")
    fig.savefig(outdir / f"{name}.pdf", metadata={"Creator": "Matplotlib", "Title": name,
                                                  "CreationDate": None}, **kw)
    fig.savefig(outdir / f"{name}.png", dpi=300, **kw)
    plt.close(fig)
