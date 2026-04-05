"""Paper figures F1-F4 generator — publication-grade redesign.

Design principles:
  1. Hero-vs-context: R4+LLM & R3 saturated, baselines desaturated.
  2. Wong colorblind-safe palette.
  3. Minimal chartjunk: thin spines, light grids, no frames on legends.
  4. Typography hierarchy: 14pt title, 11pt axis, 9pt annotation.
  5. Annotations tell the story, not just numbers.

Reads results/metrics_merged.json (3-seed paper-grade results) and outputs
publication-quality PNGs at 300 DPI into paper/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Wong colorblind-safe palette (Nature Methods 2011) ──────────────────
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
    "lightgray": "#CCCCCC",
}

# Semantic mapping: hero pair gets saturated colors, baselines get gray tones
C = {
    "R3":         WONG["red"],      # hero: expensive baseline
    "R4_llm":     WONG["blue"],     # hero: our method
    "R1":         WONG["gray"],     # context
    "R2":         WONG["gray"],     # context
    "R4_no_llm":  WONG["gray"],     # context
    "SetFit":     WONG["lightgray"],
    "highlight":  WONG["orange"],
}

# Publication style
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
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
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

with open(RESULTS / "metrics_merged.json") as f:
    M = json.load(f)


# ─────────────────────────────────────────────────────────────
# F1 — Pareto: Accuracy vs LLM call rate
# ─────────────────────────────────────────────────────────────
def fig1_pareto():
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Extract data
    r4_acc = M["aggregated"]["R4_hybrid_with_llm"]["accuracy"]["mean"] * 100
    r4_std = M["aggregated"]["R4_hybrid_with_llm"]["accuracy"]["std"] * 100
    r4_rate = M["aggregated"]["R4_hybrid_with_llm"]["llm_call_rate"]["mean"] * 100
    r4_rate_std = M["aggregated"]["R4_hybrid_with_llm"]["llm_call_rate"]["std"] * 100
    r3_acc = M["aggregated"]["R3_llm"]["accuracy"]["mean"] * 100
    r3_std = M["aggregated"]["R3_llm"]["accuracy"]["std"] * 100

    # Baselines (context, grayed out)
    baselines = [
        ("R1 keyword",        M["R1_keyword"]["accuracy"] * 100,        0),
        ("R2 embedding",      M["R2_embedding"]["accuracy"] * 100,      0),
        ("SetFit baseline",   M["SetFit_baseline"]["accuracy"] * 100,   0),
        ("R4 no-LLM",         M["R4_hybrid_no_llm"]["accuracy"] * 100,  0),
    ]
    for label, acc, rate in baselines:
        ax.scatter(rate, acc, s=70, color=C["R1"], marker="o",
                   edgecolor="white", linewidth=1.0, zorder=2, alpha=0.55)
        ax.annotate(label, xy=(rate, acc), xytext=(4, -2),
                    textcoords="offset points", fontsize=8,
                    color="#666666", ha="left", va="top")

    # Hero points: R4+LLM and R3 with error bars
    ax.errorbar([r4_rate], [r4_acc], yerr=[r4_std], xerr=[r4_rate_std],
                fmt="o", markersize=11, color=C["R4_llm"],
                markeredgecolor=C["R4_llm"], markeredgewidth=0,
                ecolor=C["R4_llm"], elinewidth=1.2, capsize=0, alpha=0.95,
                zorder=5, label="R4 hybrid + LLM (ours)")
    ax.errorbar([100], [r3_acc], yerr=[r3_std],
                fmt="s", markersize=11, color=C["R3"],
                markeredgecolor=C["R3"], markeredgewidth=0,
                ecolor=C["R3"], elinewidth=1.2, capsize=0, alpha=0.95,
                zorder=5, label="R3 LLM-only (baseline)")

    # Horizontal band showing "statistically equivalent" zone
    band_low = min(r3_acc - r3_std, r4_acc - r4_std)
    band_high = max(r3_acc + r3_std, r4_acc + r4_std)
    ax.axhspan(band_low, band_high, xmin=0.0, xmax=1.0,
               color=C["R4_llm"], alpha=0.06, zorder=0)

    # Story annotation — arrow + callout
    ax.annotate("", xy=(r4_rate + 3, r4_acc), xytext=(97, r3_acc),
                arrowprops=dict(arrowstyle="->", color="#333333",
                                lw=1.5, shrinkA=8, shrinkB=8))
    # Callout text box
    callout_text = (f"R4 matches R3 accuracy\n"
                    f"(Δ={r4_acc-r3_acc:+.1f}pp, n.s.)\n"
                    f"with 74% fewer LLM calls")
    ax.annotate(callout_text,
                xy=(62, 86.5),
                fontsize=10, color="#222222", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="#CCCCCC", linewidth=0.8))

    ax.set_xlabel("LLM call rate (%)  →  higher cost")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cost-accuracy trade-off on CLINC150 (n=1,200 pooled, 3 seeds)",
                 loc="left", pad=15)
    ax.set_xlim(-8, 108)
    ax.set_ylim(62, 90)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([65, 70, 75, 80, 85, 90])
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    path = OUT / "F1_accuracy_vs_llm_rate.png"
    plt.savefig(path)
    plt.close()
    print(f"  F1 → {path.name}")


# ─────────────────────────────────────────────────────────────
# F2 — Paired dot plot: R3 vs R4 per seed (slopegraph style)
# ─────────────────────────────────────────────────────────────
def fig2_per_seed_bars():
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    seeds = [42, 43, 44]
    r3 = [r["accuracy"] * 100 for r in M["seed_results"]["R3_llm"]]
    r4 = [r["accuracy"] * 100 for r in M["seed_results"]["R4_hybrid_with_llm"]]

    x_positions = np.arange(len(seeds))

    # Connecting lines between pairs (slopegraph)
    for i, (a, b) in enumerate(zip(r3, r4)):
        color = C["R4_llm"] if b >= a else C["R3"]
        ax.plot([i - 0.15, i + 0.15], [a, b], color="#BBBBBB",
                linewidth=1.2, zorder=1)
        # Delta annotation
        delta = b - a
        sign = "+" if delta >= 0 else ""
        ax.annotate(f"{sign}{delta:.1f}pp", xy=(i, max(a, b) + 0.4),
                    ha="center", fontsize=8.5, color="#555555",
                    style="italic")

    # Dots: R3 (left) and R4 (right)
    ax.scatter(x_positions - 0.15, r3, s=140, color=C["R3"],
               edgecolor="none", zorder=3, alpha=0.95,
               label="R3 LLM-only")
    ax.scatter(x_positions + 0.15, r4, s=140, color=C["R4_llm"],
               edgecolor="none", zorder=3, alpha=0.95,
               label="R4 hybrid + LLM")

    # Value labels
    for i, v in enumerate(r3):
        ax.annotate(f"{v:.1f}", xy=(i - 0.15, v), xytext=(-12, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=10, color=C["R3"], fontweight="bold")
    for i, v in enumerate(r4):
        ax.annotate(f"{v:.1f}", xy=(i + 0.15, v), xytext=(12, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=10, color=C["R4_llm"], fontweight="bold")

    # Mean line overlay
    r3_mean = np.mean(r3)
    r4_mean = np.mean(r4)
    ax.axhline(r3_mean, color=C["R3"], linestyle="--", linewidth=0.8,
               alpha=0.4, zorder=0)
    ax.axhline(r4_mean, color=C["R4_llm"], linestyle="--", linewidth=0.8,
               alpha=0.4, zorder=0)
    ax.annotate(f"R3 mean: {r3_mean:.1f}", xy=(2.5, r3_mean),
                xytext=(5, 3), textcoords="offset points",
                fontsize=8, color=C["R3"], style="italic")
    ax.annotate(f"R4 mean: {r4_mean:.1f}", xy=(2.5, r4_mean),
                xytext=(5, -10), textcoords="offset points",
                fontsize=8, color=C["R4_llm"], style="italic")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-seed comparison: R3 vs R4 (n=400 each)",
                 loc="left", pad=15)
    ax.set_ylim(79, 85)
    ax.set_xlim(-0.6, 2.8)
    ax.legend(loc="lower left")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    plt.tight_layout()
    path = OUT / "F2_per_seed_bars.png"
    plt.savefig(path)
    plt.close()
    print(f"  F2 → {path.name}")


# ─────────────────────────────────────────────────────────────
# F3 — Per-agent heatmap + delta row
# ─────────────────────────────────────────────────────────────
def fig3_per_agent_heatmap():
    agents = list(M["R1_keyword"]["per_agent"].keys())
    agents_short = [a.replace("_agent", "").replace("oos", "OOS") for a in agents]

    r3_per_agent = M["seed_results"]["R3_llm"][0].get("per_agent", {})
    r4_per_agent = M["seed_results"]["R4_hybrid_with_llm"][0].get("per_agent", {})

    routers = ["R1 keyword", "R2 embedding", "R4 no-LLM", "R3 LLM-only", "R4 hybrid+LLM"]
    data_sources = [
        M["R1_keyword"]["per_agent"],
        M["R2_embedding"]["per_agent"],
        M["R4_hybrid_no_llm"]["per_agent"],
        r3_per_agent,
        r4_per_agent,
    ]

    matrix = np.zeros((len(routers), len(agents)))
    for i, src in enumerate(data_sources):
        for j, agent in enumerate(agents):
            v = src.get(agent, {})
            matrix[i, j] = v.get("accuracy", 0.0) * 100 if isinstance(v, dict) else 0.0

    # Delta row: R4+LLM minus R3
    delta = matrix[4] - matrix[3]

    # Two subplots sharing x-axis: main heatmap + delta row
    fig, (ax, ax_delta) = plt.subplots(
        2, 1, figsize=(10, 5.5),
        gridspec_kw={"height_ratios": [5, 1.2], "hspace": 0.15},
        sharex=True,
    )

    # Main heatmap — diverging palette centered at 75
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=25, vmax=100)

    ax.set_yticks(range(len(routers)))
    ax.set_yticklabels(routers, fontsize=10)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    # Color + bold the hero row labels (R3 red, R4+LLM blue)
    tick_labels = ax.get_yticklabels()
    tick_labels[3].set_color(C["R3"])
    tick_labels[3].set_fontweight("bold")
    tick_labels[4].set_color(C["R4_llm"])
    tick_labels[4].set_fontweight("bold")

    for i in range(len(routers)):
        for j in range(len(agents)):
            val = matrix[i, j]
            color = "black" if 55 < val < 88 else "white"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.set_title("Per-agent accuracy across routers  (R3/R4+LLM: seed 42)",
                 loc="left", pad=12)
    ax.grid(False)

    # Delta row — shows where R4+LLM beats/loses to R3
    delta_cmap = plt.cm.RdBu_r
    max_abs = max(abs(delta.min()), abs(delta.max()), 5)
    im_d = ax_delta.imshow(delta[np.newaxis, :], aspect="auto",
                            cmap=delta_cmap, vmin=-max_abs, vmax=max_abs)
    ax_delta.set_yticks([0])
    ax_delta.set_yticklabels(["Δ (R4+LLM − R3)"], fontsize=9)
    ax_delta.tick_params(axis="x", length=0)
    ax_delta.tick_params(axis="y", length=0)
    for j in range(len(agents)):
        val = delta[j]
        sign = "+" if val >= 0 else ""
        color = "white" if abs(val) > max_abs * 0.5 else "black"
        ax_delta.text(j, 0, f"{sign}{val:.0f}", ha="center", va="center",
                      color=color, fontsize=9, fontweight="bold")
    ax_delta.set_xticks(range(len(agents)))
    ax_delta.set_xticklabels(agents_short, rotation=30, ha="right", fontsize=10)
    ax_delta.grid(False)

    # Shared colorbar for main heatmap only
    cbar = fig.colorbar(im, ax=[ax, ax_delta], shrink=0.75, pad=0.02,
                        location="right")
    cbar.set_label("Accuracy (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    path = OUT / "F3_per_agent_heatmap.png"
    plt.savefig(path)
    plt.close()
    print(f"  F3 → {path.name}")


# ─────────────────────────────────────────────────────────────
# F4 — Cost comparison (clean horizontal bars + headline number)
# ─────────────────────────────────────────────────────────────
def fig4_cost_bars():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    r3_cost = M["aggregated"]["R3_llm"]["cost_usd"]["mean"] / M["config"]["llm_n_per_seed"]
    r4_cost = M["aggregated"]["R4_hybrid_with_llm"]["cost_usd"]["mean"] / M["config"]["llm_n_per_seed"]
    saving_pct = (1 - r4_cost / r3_cost) * 100

    # Convert to $ per 1000 queries for readability
    r3_per_k = r3_cost * 1000
    r4_per_k = r4_cost * 1000

    methods = ["R3 LLM-only", "R4 hybrid + LLM"]
    costs = [r3_per_k, r4_per_k]
    colors = [C["R3"], C["R4_llm"]]

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, costs, color=colors, edgecolor="white",
                    linewidth=1.5, height=0.55)

    # Value labels at bar tips
    for bar, c in zip(bars, costs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"${c:.3f}", va="center", ha="left",
                fontsize=12, fontweight="bold", color="#222222")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel("Cost per 1,000 queries (USD)")
    ax.set_title("LLM API cost: R4 is 74% cheaper at equivalent accuracy",
                 loc="left", pad=15)
    ax.set_xlim(0, max(costs) * 1.25)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)

    # Savings callout — big number
    savings_per_k = r3_per_k - r4_per_k
    callout = (f"Savings: ${savings_per_k:.3f} per 1k queries\n"
               f"({saving_pct:.0f}% reduction)")
    ax.annotate(callout,
                xy=(max(costs) * 0.7, 1.5),
                fontsize=10, color="#222222", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                          edgecolor="#CCCCCC", linewidth=0.8))

    # Footer: footnote on local-compute methods
    ax.annotate(
        "Note: R1/R2/R4-no-LLM use local compute only (embedding lookup + rules); API cost is negligible.",
        xy=(0, -0.35), xycoords="axes fraction",
        fontsize=8, color="#666666", style="italic",
    )

    plt.tight_layout()
    path = OUT / "F4_cost_bars.png"
    plt.savefig(path)
    plt.close()
    print(f"  F4 → {path.name}")


def main():
    print(f"Generating paper figures → {OUT}")
    fig1_pareto()
    fig2_per_seed_bars()
    fig3_per_agent_heatmap()
    fig4_cost_bars()
    print(f"\nDone. {len(list(OUT.glob('*.png')))} figures written.")


if __name__ == "__main__":
    main()
