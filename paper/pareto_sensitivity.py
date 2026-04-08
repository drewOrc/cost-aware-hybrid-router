"""
W4 — Pareto sensitivity analysis: sweep α in objective acc − α·llm_rate.

For each α, find the optimal (kt, et) from the validation grid,
then plot the resulting Pareto frontier showing the accuracy–cost trade-off
as the cost penalty α varies.

Also generates W8 analysis: R2-stop error rate per agent from seed-42 cascade data.

Reads:
  results/tuned_thresholds.json  (grid_with_llm: 42 configurations)
  results/metrics_merged.json    (per-seed cascade data)

Outputs:
  paper/figures/F5_pareto_sensitivity.png
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Wong colorblind-safe palette ──────────────────────
WONG = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "gray":    "#999999",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "normal",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def fig5_pareto_sensitivity():
    """Sweep α and show optimal operating points on a single Pareto curve."""

    with open(RESULTS / "tuned_thresholds.json") as f:
        thresholds = json.load(f)

    grid = thresholds["grid_with_llm"]

    # α values to sweep (objective: max acc − α·llm_rate)
    alphas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    # For each α, find optimal (kt, et)
    selected = []
    for alpha in alphas:
        best = max(grid, key=lambda g: g["expected_accuracy"] - alpha * g["llm_call_rate"])
        selected.append({
            "alpha": alpha,
            "kt": best["keyword_threshold"],
            "et": best["embed_threshold"],
            "acc": best["expected_accuracy"] * 100,
            "llm_rate": best["llm_call_rate"] * 100,
        })

    # Paper's actual config: min LLM_call_rate s.t. acc ≥ 0.88
    paper_cfg = [g for g in grid
                 if g["keyword_threshold"] == 0.5 and g["embed_threshold"] == 0.1][0]
    paper_point = {
        "acc": paper_cfg["expected_accuracy"] * 100,
        "llm_rate": paper_cfg["llm_call_rate"] * 100,
    }

    # Print table
    print("α-sensitivity analysis (validation set, expected accuracy):")
    print(f"  {'α':<6} {'kt':<5} {'et':<6} {'Acc%':<8} {'LLM%':<8} {'Saved%'}")
    print("  " + "─" * 45)
    for s in selected:
        saved = 100 - s["llm_rate"]
        print(f"  {s['alpha']:<6.2f} {s['kt']:<5.1f} {s['et']:<6.2f} {s['acc']:<8.1f} {s['llm_rate']:<8.1f} {saved:.1f}%")

    # ─── Plot ───
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Background: all grid points as light dots
    all_rates = [g["llm_call_rate"] * 100 for g in grid]
    all_accs = [g["expected_accuracy"] * 100 for g in grid]
    ax.scatter(all_rates, all_accs, c=WONG["gray"], alpha=0.25, s=20, zorder=1,
               label="All grid configurations")

    # Pareto frontier: connect selected points
    rates = [s["llm_rate"] for s in selected]
    accs = [s["acc"] for s in selected]
    ax.plot(rates, accs, color=WONG["blue"], linewidth=2, zorder=3, alpha=0.5)

    # Color gradient for α
    cmap = plt.cm.RdYlGn_r  # red=aggressive cost, green=accuracy-first
    norm = plt.Normalize(vmin=0, vmax=0.5)

    # Only label unique operating points (many α map to same config)
    seen = set()
    unique_points = []
    for s in selected:
        key = (s["kt"], s["et"])
        if key not in seen:
            seen.add(key)
            unique_points.append(s)

    for s in selected:
        color = cmap(norm(s["alpha"]))
        ax.scatter(s["llm_rate"], s["acc"], c=[color], s=100, zorder=4,
                   edgecolors="white", linewidth=1.2)

    # Label only unique operating points with their α range
    alpha_ranges = {}
    for s in selected:
        key = (s["kt"], s["et"])
        if key not in alpha_ranges:
            alpha_ranges[key] = [s["alpha"], s["alpha"]]
        else:
            alpha_ranges[key][1] = s["alpha"]

    label_offsets = {
        (1.5, 0.15): (8, 0.3),     # acc-first regime
        (1.5, 0.10): (8, 0.3),     # moderate
        (1.5, 0.05): (-15, 0.5),   # aggressive
        (0.5, 0.05): (-12, -0.5),  # most aggressive
    }
    for pt in unique_points:
        key = (pt["kt"], pt["et"])
        amin, amax = alpha_ranges[key]
        if amin == amax:
            label = f"α={amin:.2f}"
        else:
            label = f"α∈[{amin:.2f},{amax:.2f}]"
        label += f"\nkt={pt['kt']}, et={pt['et']}"
        ox, oy = label_offsets.get(key, (5, 0.3))
        ax.annotate(label,
                     xy=(pt["llm_rate"], pt["acc"]),
                     xytext=(pt["llm_rate"] + ox, pt["acc"] + oy),
                     fontsize=7, color="#333333",
                     arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.6))

    # Mark the paper's actual config: min LLM rate s.t. acc >= 88%
    ax.scatter([paper_point["llm_rate"]], [paper_point["acc"]],
               c=[WONG["red"]], s=180, zorder=5, marker="*",
               edgecolors="white", linewidth=1)
    ax.annotate("Paper config\n(kt=0.5, et=0.10)\nmin LLM rate s.t. acc≥88%",
                xy=(paper_point["llm_rate"], paper_point["acc"]),
                xytext=(paper_point["llm_rate"] + 15, paper_point["acc"] + 0.5),
                fontsize=8, color=WONG["red"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WONG["red"], lw=1.2))

    # R3 LLM-only reference line (100% LLM calls)
    # R3 expected acc on val ≈ P_LLM_CORRECT from tune.py = 85.6%
    ax.axvline(x=100, color=WONG["red"], linestyle="--", linewidth=1, alpha=0.4, zorder=1)
    ax.text(98, min(accs) - 0.3, "R3: 100% LLM", fontsize=7.5, color=WONG["red"],
            alpha=0.6, ha="right", va="top")

    ax.set_xlabel("LLM Call Rate (%)")
    ax.set_ylabel("Expected Accuracy (%)")
    ax.set_title("Threshold Sensitivity to Cost Penalty α")
    ax.set_xlim(-2, 100)
    ax.set_ylim(85.5, 90.5)

    # Adjust y-range to fit data
    min_acc = min(accs) - 1
    max_acc = max(accs) + 1
    min_rate = -2
    max_rate = max(all_rates) + 5
    ax.set_xlim(min_rate, max_rate)
    ax.set_ylim(min_acc, max_acc)

    # Colorbar for α
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Cost penalty α", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    path = OUT / "F5_pareto_sensitivity.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"\n✅ Saved {path}")


if __name__ == "__main__":
    fig5_pareto_sensitivity()
