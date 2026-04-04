"""Flatten metrics_merged.json into metrics.json format for analysis.py."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
with open(R / "metrics_merged.json") as f:
    m = json.load(f)

# Use mean across seeds as the representative R3/R4+LLM point
r3_mean_acc = m["aggregated"]["R3_llm"]["accuracy"]["mean"]
r3_mean_cost = m["aggregated"]["R3_llm"]["cost_usd"]["mean"]
r3_n = m["seed_results"]["R3_llm"][0]["total"]

r4_mean_acc = m["aggregated"]["R4_hybrid_with_llm"]["accuracy"]["mean"]
r4_mean_cost = m["aggregated"]["R4_hybrid_with_llm"]["cost_usd"]["mean"]
r4_n = m["seed_results"]["R4_hybrid_with_llm"][0]["total"]
r4_stages = m["seed_results"]["R4_hybrid_with_llm"][0].get("stages", {})
r4_llm_calls = m["seed_results"]["R4_hybrid_with_llm"][0]["llm_fallback_count"]

# Aggregate per_agent across seeds for R3 and R4+LLM (mean of accuracies weighted by total)
def merge_per_agent(records):
    from collections import defaultdict
    agg = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in records:
        for a, v in r["per_agent"].items():
            agg[a]["total"] += v["total"]
            agg[a]["correct"] += v["correct"]
    return {a: {"total": v["total"], "correct": v["correct"],
                "accuracy": round(v["correct"]/v["total"], 4) if v["total"] else 0}
            for a, v in sorted(agg.items())}

flat = {
    "config": m["config"],
    "R1_keyword": m["R1_keyword"],
    "R2_embedding": m["R2_embedding"],
    "R4_hybrid_no_llm": m["R4_hybrid_no_llm"],
    "R3_llm": {
        "router": "R3_llm",
        "total": r3_n,
        "accuracy": r3_mean_acc,
        "sample_size": r3_n,
        "cost_usd": r3_mean_cost,
        "per_agent": merge_per_agent(m["seed_results"]["R3_llm"]),
    },
    "R4_hybrid_with_llm": {
        "router": "R4_hybrid_with_llm",
        "total": r4_n,
        "accuracy": r4_mean_acc,
        "sample_size": r4_n,
        "cost_usd": r4_mean_cost,
        "per_agent": merge_per_agent(m["seed_results"]["R4_hybrid_with_llm"]),
        "stages": r4_stages,
        "hybrid_stats": {
            "keyword_accepted": r4_stages.get("keyword", 0),
            "embedding_accepted": r4_stages.get("embedding", 0),
            "llm_fallback": r4_llm_calls,
            "total_queries": r4_n,
        },
    },
}
if m.get("SetFit_baseline"):
    flat["SetFit_baseline"] = m["SetFit_baseline"]

with open(R / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(flat, f, indent=2, ensure_ascii=False, default=str)

print(f"Wrote {R / 'metrics.json'}")
print(f"  R3:     {r3_mean_acc*100:.1f}% cost/1k=${r3_mean_cost/r3_n*1000:.3f}")
print(f"  R4+LLM: {r4_mean_acc*100:.1f}% cost/1k=${r4_mean_cost/r4_n*1000:.3f}")
