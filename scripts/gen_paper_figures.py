"""
gen_paper_figures.py — Generate publication-quality figures for the paper.

Generates:
  - fig_tau_tradeoff.png: τ vs accuracy/LLM-rate tradeoff (for §4.6)
  - fig_pareto.png: Refreshed Pareto frontier with all routers (for §4.1)

Usage:
    python3 scripts/gen_paper_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CAL_DIR = ROOT / "results" / "calibrated_routing"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
})

# ACL column width ≈ 3.25 inches, full width ≈ 6.75 inches
COL_W = 3.25
FULL_W = 6.75


def gen_tau_tradeoff():
    """Generate τ tradeoff curve (column-width, for §4.6)."""
    with open(CAL_DIR / "tau_sweep.json") as f:
        sweep = json.load(f)
    with open(CAL_DIR / "comparison.json") as f:
        comp = json.load(f)

    taus = [r["tau"] for r in sweep]
    acc_wllm = [r["accuracy_with_llm_expected"] * 100 for r in sweep]
    llm_rate = [r["llm_call_rate"] * 100 for r in sweep]

    # Grid operating point
    grid_llm = comp["grid_baseline"]["with_llm"]["llm_call_rate"] * 100
    grid_acc = comp["grid_baseline"]["with_llm"]["accuracy_with_llm_expected"] * 100

    fig, ax1 = plt.subplots(figsize=(COL_W, 2.2))

    # Accuracy line (left y-axis)
    color_acc = "#0072B2"
    ax1.plot(taus, acc_wllm, color=color_acc, linewidth=1.5, zorder=3)
    ax1.set_xlabel(r"Threshold $\tau$")
    ax1.set_ylabel("Expected accuracy (%)", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xlim(0.50, 0.95)
    ax1.set_ylim(73, 89)

    # LLM rate line (right y-axis)
    ax2 = ax1.twinx()
    color_llm = "#D55E00"
    ax2.plot(taus, llm_rate, color=color_llm, linewidth=1.5, linestyle="--", zorder=3)
    ax2.set_ylabel("LLM call rate (%)", color=color_llm)
    ax2.tick_params(axis="y", labelcolor=color_llm)
    ax2.set_ylim(-2, 82)

    # Mark τ=0.75 (our operating point)
    tau_075 = [r for r in sweep if r["tau"] == 0.75][0]
    ax1.plot(0.75, tau_075["accuracy_with_llm_expected"] * 100,
             "o", color=color_acc, markersize=5, zorder=5)
    ax2.plot(0.75, tau_075["llm_call_rate"] * 100,
             "D", color=color_llm, markersize=4, zorder=5)
    ax1.annotate(r"$\tau\!=\!0.75$" + "\n83.9%, 25.4% LLM",
                 xy=(0.75, tau_075["accuracy_with_llm_expected"] * 100),
                 xytext=(0.80, 86.5), fontsize=6.5,
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.6),
                 ha="left", color="gray")

    # Mark grid-search point
    ax1.axhline(y=grid_acc, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
    ax1.annotate(f"Grid: {grid_acc:.1f}%, {grid_llm:.0f}% LLM",
                 xy=(0.52, grid_acc), fontsize=6, color="gray", va="bottom")

    ax1.grid(True, alpha=0.15, linewidth=0.4)
    fig.tight_layout(pad=0.3)

    out = FIG_DIR / "fig_tau_tradeoff.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def gen_pareto():
    """Generate Pareto frontier with all routers (full-width, for §4.1)."""
    # Data from paper
    routers = [
        {"name": "R1 keyword",   "acc": 64.8, "llm": 0,   "marker": "^", "color": "#999999"},
        {"name": "R2 embedding", "acc": 74.0, "llm": 0,   "marker": "s", "color": "#999999"},
        {"name": "SetFit",       "acc": 70.2, "llm": 0,   "marker": "v", "color": "#999999"},
        {"name": "R4 no-LLM",   "acc": 74.9, "llm": 0,   "marker": "p", "color": "#999999"},
        {"name": "R3 LLM-only", "acc": 82.9, "llm": 100, "marker": "o", "color": "#D55E00"},
        {"name": "R4+LLM (ours)", "acc": 82.6, "llm": 26, "marker": "o", "color": "#0072B2"},
    ]

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))

    for r in routers:
        ms = 7 if "ours" in r["name"] or "LLM-only" in r["name"] else 5
        zord = 5 if "ours" in r["name"] else 3
        ax.scatter(r["llm"], r["acc"], marker=r["marker"], color=r["color"],
                   s=ms**2, zorder=zord, edgecolors="white", linewidths=0.3)

    # Arrow connecting R4+LLM and R3
    ax.annotate("", xy=(26, 82.6), xytext=(100, 82.9),
                arrowprops=dict(arrowstyle="<->", color="#0072B2", lw=0.8,
                                connectionstyle="arc3,rad=0.1"))
    ax.annotate(r"$\Delta$=0.3pp (n.s.)" + "\n74% fewer LLM calls",
                xy=(58, 83.8), fontsize=6.5, ha="center", color="#0072B2")

    # Labels
    offsets = {
        "R1 keyword": (8, -2), "R2 embedding": (8, -2),
        "SetFit": (8, -2), "R4 no-LLM": (8, 3),
        "R3 LLM-only": (-8, -10), "R4+LLM (ours)": (-8, -10),
    }
    for r in routers:
        dx, dy = offsets.get(r["name"], (5, 5))
        ax.annotate(r["name"], xy=(r["llm"], r["acc"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6, color=r["color"])

    ax.set_xlabel(r"LLM call rate (%) $\rightarrow$ higher cost")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-5, 108)
    ax.set_ylim(62, 86)
    ax.grid(True, alpha=0.15, linewidth=0.4)
    fig.tight_layout(pad=0.3)

    out = FIG_DIR / "fig_pareto.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("Generating paper figures...")
    gen_pareto()
    gen_tau_tradeoff()
    print("Done.")
