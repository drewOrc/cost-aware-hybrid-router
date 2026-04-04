"""Merge per-seed metrics files into unified metrics.json with mean±std + McNemar."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Load baseline (R1/R2/R4-no-LLM/SetFit from full test set)
with open(RESULTS / "metrics_no_llm_baseline.json") as f:
    baseline = json.load(f)

# Load per-seed LLM results
seed_files = sorted(RESULTS.glob("metrics_seed*.json"))
seeds_data = []
for f in seed_files:
    with open(f) as fp:
        seeds_data.append(json.load(fp))

def agg(values):
    import statistics
    if len(values) < 2:
        return {"mean": round(values[0], 4), "std": 0.0, "min": values[0], "max": values[0]}
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }

# Collect per-seed records
r3_records = []
r4_records = []
mcnemar_records = []

for sd in seeds_data:
    r3_records.extend(sd["seed_results"]["R3_llm"])
    r4_records.extend(sd["seed_results"]["R4_hybrid_with_llm"])
    mcnemar_records.extend(sd["seed_results"]["mcnemar_r4llm_vs_r3"])

r3_accs = [r["accuracy"] for r in r3_records]
r4_accs = [r["accuracy"] for r in r4_records]
r3_costs = [r["cost_usd"] for r in r3_records]
r4_costs = [r["cost_usd"] for r in r4_records]
r4_llm_rates = [r["llm_call_rate"] for r in r4_records]

# Wilson CI (use first seed)
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return {"lower": 0, "upper": 0}
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * ((p*(1-p)/n + z**2/(4*n**2)) ** 0.5) / denom
    return {"lower": round(center - margin, 4), "upper": round(center + margin, 4), "point": round(p, 4)}

# Pooled Wilson CI (concatenate correct counts)
r3_pool_k = sum(r["correct"] for r in r3_records)
r3_pool_n = sum(r["total"] for r in r3_records)
r4_pool_k = sum(r["correct"] for r in r4_records)
r4_pool_n = sum(r["total"] for r in r4_records)

merged = {
    "config": {
        "llm_n_per_seed": seeds_data[0]["config"]["llm_n"],
        "n_per_agent": seeds_data[0]["config"]["n_per_agent"],
        "seeds": [sd["config"]["seeds"][0] for sd in seeds_data],
        "n_seeds": len(seeds_data),
        "workers": seeds_data[0]["config"]["workers"],
        "thresholds_with_llm": seeds_data[0]["config"]["thresholds_with_llm"],
        "thresholds_source": baseline["config"]["threshold_source"],
        "thresholds_no_llm": baseline["config"]["thresholds_no_llm"],
    },
    # Full test-set baselines (from metrics_no_llm_baseline.json)
    "R1_keyword": baseline["R1_keyword"],
    "R2_embedding": baseline["R2_embedding"],
    "R4_hybrid_no_llm": baseline["R4_hybrid_no_llm"],
    "SetFit_baseline": baseline.get("SetFit_baseline"),
    # Multi-seed LLM
    "seed_results": {
        "R3_llm": r3_records,
        "R4_hybrid_with_llm": r4_records,
        "mcnemar_r4llm_vs_r3": mcnemar_records,
    },
    "aggregated": {
        "R3_llm": {
            "accuracy": agg(r3_accs),
            "cost_usd": agg(r3_costs),
        },
        "R4_hybrid_with_llm": {
            "accuracy": agg(r4_accs),
            "cost_usd": agg(r4_costs),
            "llm_call_rate": agg(r4_llm_rates),
        },
        "wilson_ci_R3_pooled": wilson_ci(r3_pool_k, r3_pool_n),
        "wilson_ci_R4LLM_pooled": wilson_ci(r4_pool_k, r4_pool_n),
        "mcnemar_significant_count": sum(1 for m in mcnemar_records if m["significant_at_0.05"]),
        "mcnemar_p_values": [m["p_value"] for m in mcnemar_records],
        "mcnemar_deltas": [m["delta"] for m in mcnemar_records],
        "n_seeds": len(seeds_data),
        "total_pooled_n": r3_pool_n,
    },
    "total_cost_usd": round(sum(r3_costs) + sum(r4_costs), 4),
}

out = RESULTS / "metrics_merged.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False, default=str)

# Print summary
print("=" * 70)
print("  Merged Results (3 seeds × n=400 = 1,200 pooled queries)")
print("=" * 70)
print(f"\n  Full test set (n=5,500, deterministic):")
print(f"    R1_keyword          {baseline['R1_keyword']['accuracy']*100:>5.1f}%")
print(f"    R2_embedding        {baseline['R2_embedding']['accuracy']*100:>5.1f}%")
print(f"    R4_hybrid_no_llm    {baseline['R4_hybrid_no_llm']['accuracy']*100:>5.1f}%")
if baseline.get("SetFit_baseline"):
    print(f"    SetFit_baseline     {baseline['SetFit_baseline']['accuracy']*100:>5.1f}%")

print(f"\n  LLM-involved routers (mean ± std across 3 seeds):")
print(f"    R3_llm              {merged['aggregated']['R3_llm']['accuracy']['mean']*100:>5.1f}% "
      f"± {merged['aggregated']['R3_llm']['accuracy']['std']*100:.1f}pp")
print(f"    R4_hybrid_with_llm  {merged['aggregated']['R4_hybrid_with_llm']['accuracy']['mean']*100:>5.1f}% "
      f"± {merged['aggregated']['R4_hybrid_with_llm']['accuracy']['std']*100:.1f}pp")

print(f"\n  LLM call rate (R4+LLM): "
      f"{merged['aggregated']['R4_hybrid_with_llm']['llm_call_rate']['mean']*100:.1f}% "
      f"± {merged['aggregated']['R4_hybrid_with_llm']['llm_call_rate']['std']*100:.1f}pp")

print(f"\n  Wilson 95% CI (pooled, n={r3_pool_n}):")
print(f"    R3:       [{merged['aggregated']['wilson_ci_R3_pooled']['lower']*100:.1f}%, "
      f"{merged['aggregated']['wilson_ci_R3_pooled']['upper']*100:.1f}%]")
print(f"    R4+LLM:   [{merged['aggregated']['wilson_ci_R4LLM_pooled']['lower']*100:.1f}%, "
      f"{merged['aggregated']['wilson_ci_R4LLM_pooled']['upper']*100:.1f}%]")

print(f"\n  McNemar (R4+LLM vs R3):")
for m in mcnemar_records:
    print(f"    seed {m['seed']}: p={m['p_value']:.4f}  delta={m['delta']*100:+.2f}pp  "
          f"sig={m['significant_at_0.05']}")
print(f"    Significant in {merged['aggregated']['mcnemar_significant_count']}/3 seeds")

print(f"\n  Cost:")
print(f"    R3 total:     ${sum(r3_costs):.4f}")
print(f"    R4+LLM total: ${sum(r4_costs):.4f}")
print(f"    Grand total:  ${merged['total_cost_usd']:.4f}")
print(f"\n  Saved to: {out}")
