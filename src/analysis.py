"""
analysis.py — 從 metrics.json 產生圖表

輸出：
    results/figures/accuracy_vs_cost.png — Pareto scatter plot
    results/figures/per_agent_accuracy.png — 各 agent 準確率比較
    results/figures/hybrid_flow.png — Hybrid Router 流量分配
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / "metrics.json") as f:
    metrics = json.load(f)


# ─────────────────────────────────────────────
# Fig 1: Accuracy vs Cost (Pareto Frontier)
# ─────────────────────────────────────────────

def plot_accuracy_vs_cost():
    fig, ax = plt.subplots(figsize=(8, 6))

    # 為了公平比較，把成本標準化到「每 1000 query 的美元成本」
    routers = {
        "R1\nKeyword": {"acc": metrics["R1_keyword"]["accuracy"], "cost_per_1k": 0},
        "R2\nEmbedding": {"acc": metrics["R2_embedding"]["accuracy"], "cost_per_1k": 0},
        "R4\nHybrid\n(no LLM)": {"acc": metrics["R4_hybrid_no_llm"]["accuracy"], "cost_per_1k": 0},
    }

    if "R3_llm" in metrics:
        r3 = metrics["R3_llm"]
        cost_per_q = r3["cost_usd"] / r3["sample_size"]
        routers["R3\nLLM"] = {"acc": r3["accuracy"], "cost_per_1k": cost_per_q * 1000}

    if "R4_hybrid_with_llm" in metrics:
        r4l = metrics["R4_hybrid_with_llm"]
        cost_per_q = r4l["cost_usd"] / r4l["sample_size"]
        routers["R4+LLM\nHybrid\n(with LLM)"] = {"acc": r4l["accuracy"], "cost_per_1k": cost_per_q * 1000}

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    markers = ["s", "^", "D", "o", "*"]

    for i, (name, data) in enumerate(routers.items()):
        ax.scatter(
            data["cost_per_1k"], data["acc"] * 100,
            s=200, c=colors[i], marker=markers[i],
            zorder=5, edgecolors="white", linewidths=1.5
        )
        # Label offset
        offset_x = 0.005 if data["cost_per_1k"] == 0 else -0.02
        ax.annotate(
            name,
            (data["cost_per_1k"], data["acc"] * 100),
            textcoords="offset points",
            xytext=(12, -5),
            fontsize=9,
            ha="left"
        )

    # Pareto frontier line (connect non-dominated points)
    points = sorted(routers.items(), key=lambda x: x[1]["cost_per_1k"])
    pareto_x, pareto_y = [], []
    best_acc = 0
    for name, data in points:
        if data["acc"] >= best_acc:
            pareto_x.append(data["cost_per_1k"])
            pareto_y.append(data["acc"] * 100)
            best_acc = data["acc"]
    ax.plot(pareto_x, pareto_y, "k--", alpha=0.3, linewidth=1, label="Pareto frontier")

    ax.set_xlabel("Cost per 1,000 queries (USD)", fontsize=12)
    ax.set_ylabel("Routing Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs. Cost — Router Comparison on CLINC150", fontsize=13, fontweight="bold")
    ax.set_ylim(60, 95)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "accuracy_vs_cost.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────
# Fig 2: Per-Agent Accuracy Comparison
# ─────────────────────────────────────────────

def plot_per_agent():
    fig, ax = plt.subplots(figsize=(10, 6))

    # 只比較跑完整 test set 的 router
    router_names = ["R1_keyword", "R2_embedding", "R4_hybrid_no_llm"]
    display_names = ["R1 Keyword", "R2 Embedding", "R4 Hybrid"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # 取所有 agent
    agents = sorted(metrics["R1_keyword"]["per_agent"].keys())

    x = np.arange(len(agents))
    width = 0.25

    for i, (rname, dname) in enumerate(zip(router_names, display_names)):
        accs = [
            metrics[rname]["per_agent"].get(a, {}).get("accuracy", 0) * 100
            for a in agents
        ]
        ax.bar(x + i * width, accs, width, label=dname, color=colors[i], alpha=0.85)

    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Agent Routing Accuracy — R1 vs R2 vs R4", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([a.replace("_agent", "").replace("_", "\n") for a in agents], fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIGURES_DIR / "per_agent_accuracy.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────
# Fig 3: Hybrid Router Flow Distribution
# ─────────────────────────────────────────────

def plot_hybrid_flow():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: R4 no LLM (full test set)
    hs = metrics["R4_hybrid_no_llm"]["hybrid_stats"]
    total = hs["total_queries"]
    sizes_left = [hs["keyword_accepted"], hs["embedding_accepted"]]
    labels_left = [
        f"Keyword\n{hs['keyword_accepted']} ({hs['keyword_accepted']/total*100:.1f}%)",
        f"Embedding\n{hs['embedding_accepted']} ({hs['embedding_accepted']/total*100:.1f}%)",
    ]
    colors_left = ["#2196F3", "#4CAF50"]

    axes[0].pie(sizes_left, labels=labels_left, colors=colors_left, autopct="", startangle=90,
                textprops={"fontsize": 10})
    axes[0].set_title(f"R4 Hybrid (no LLM)\nAccuracy: {metrics['R4_hybrid_no_llm']['accuracy']*100:.1f}%\nn={total}",
                      fontsize=11, fontweight="bold")

    # Right: R4+LLM (sample)
    if "R4_hybrid_with_llm" in metrics:
        hs2 = metrics["R4_hybrid_with_llm"]["hybrid_stats"]
        total2 = hs2["total_queries"]
        sizes_right = [hs2["keyword_accepted"], hs2["embedding_accepted"], hs2["llm_fallback"]]
        labels_right = [
            f"Keyword\n{hs2['keyword_accepted']} ({hs2['keyword_accepted']/total2*100:.1f}%)",
            f"Embedding\n{hs2['embedding_accepted']} ({hs2['embedding_accepted']/total2*100:.1f}%)",
            f"LLM\n{hs2['llm_fallback']} ({hs2['llm_fallback']/total2*100:.1f}%)",
        ]
        colors_right = ["#2196F3", "#4CAF50", "#F44336"]

        axes[1].pie(sizes_right, labels=labels_right, colors=colors_right, autopct="", startangle=90,
                    textprops={"fontsize": 10})
        axes[1].set_title(f"R4 Hybrid (with LLM)\nAccuracy: {metrics['R4_hybrid_with_llm']['accuracy']*100:.1f}%\nn={total2}",
                          fontsize=11, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "No LLM data", ha="center", va="center", fontsize=12)
        axes[1].set_title("R4+LLM")

    plt.suptitle("Hybrid Router — Query Flow Distribution", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "hybrid_flow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Generating Figures ===\n")
    plot_accuracy_vs_cost()
    plot_per_agent()
    plot_hybrid_flow()
    print("\nDone.")
