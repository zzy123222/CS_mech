from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROLE_COLOR = {
    "Monitor":    "#00bcd4",
    "Controller": "#ff9800",
    "Integrator": "#2196f3",
    "Opponent":   "#7e57c2",
}
MODEL_COLOR = {"Base": "#4a5560", "Instruct": "#000000"}
ARM_COLOR = {"General": "#7f8c8d", "Correction": "#c0392b", "Sham": "#2980b9"}
GRADIENT_COLOR = {"0%": "#2980b9", "25%": "#7e57c2", "50%": "#5e35b1",
                  "75%": "#4527a0", "100%": "#c0392b"}
RECONCILE_COLOR = "#8e6c3f"

FONT = {"tick": 8, "label": 9, "legend": 8.5, "panel_title": 10.5,
        "panel_tag": 13, "fig_title": 12}
CI_ALPHA = 0.15
LINE = {"data": 1.6, "ref": 1.0}
DPI = 200

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": FONT["tick"],
    "axes.titlesize": FONT["panel_title"],
    "axes.labelsize": FONT["label"],
    "xtick.labelsize": FONT["tick"],
    "ytick.labelsize": FONT["tick"],
    "legend.fontsize": FONT["legend"],
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.dpi": DPI,
    "figure.dpi": DPI,
}


def apply():
    plt.rcParams.update(_RC)


def panel_tag(ax, letter: str):
    ax.set_title(f"({letter}) {ax.get_title()}", loc="left",
                 fontweight="bold", fontsize=FONT["panel_title"])


def save_both(fig, path, name):
    fig.savefig(f"{path}/{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{path}/{name}.pdf", format="pdf", bbox_inches="tight")
