"""
W8 + W6 analysis script — run locally to produce:
  1. R2-stop error rate per agent (for updated Table 3)
  2. Per-stage latency benchmark (for paper §5)

Usage:
  cd cost-aware-hybrid-router/
  PYTHONPATH=. python3 paper/analyze_cascade_stages.py

No API key needed — only uses R1/R2 (deterministic, local compute).
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# ── Load data ──────────────────────────────────────────────
def load_seed_sample(seed: int, n_per_agent: int = 50):
    """Reproduce the same stratified sample as evaluate_llm_parallel.py."""
    with open(DATA_DIR / "clinc150" / "test.json") as f:
        test_data = json.load(f)
    with open(DATA_DIR / "clinc150" / "intent_names.json") as f:
        intent_names = json.load(f)
    with open(DATA_DIR / "intent_to_agent.json") as f:
        raw = json.load(f)
        mapping = {k: v for k, v in raw.items() if k != "_meta"}

    # Group by agent
    by_agent = defaultdict(list)
    for item in test_data:
        agent = mapping[intent_names[item["intent"]]]
        by_agent[agent].append(item)

    # Stratified sample
    rng = np.random.RandomState(seed)
    sample = []
    for agent in sorted(by_agent.keys()):
        pool = by_agent[agent]
        idx = rng.choice(len(pool), size=n_per_agent, replace=False)
        for i in idx:
            sample.append({
                "text": pool[i]["text"],
                "true_agent": agent,
            })
    return sample


def main():
    from src.routers import keyword_router, embedding_router

    # Thresholds from tuned_thresholds.json
    KT = 0.5
    ET = 0.10

    print("=" * 60)
    print("W8: Per-agent R2-stop error rate (seed 42, n=400)")
    print("=" * 60)

    sample = load_seed_sample(42, 50)
    print(f"Loaded {len(sample)} queries\n")

    # Track per-agent, per-stage
    stages_data = defaultdict(lambda: {
        "r1_stop": 0, "r1_correct": 0,
        "r2_stop": 0, "r2_correct": 0,
        "escalated": 0, "total": 0,
    })

    for item in sample:
        agent = item["true_agent"]
        text = item["text"]
        stages_data[agent]["total"] += 1

        r1 = keyword_router.route(text)
        if r1["confidence"] >= KT:
            stages_data[agent]["r1_stop"] += 1
            if r1["agent"] == agent:
                stages_data[agent]["r1_correct"] += 1
        else:
            r2 = embedding_router.route(text)
            if r2["confidence"] >= ET:
                stages_data[agent]["r2_stop"] += 1
                if r2["agent"] == agent:
                    stages_data[agent]["r2_correct"] += 1
            else:
                stages_data[agent]["escalated"] += 1

    # Print table
    print(f"{'Agent':<22} {'Total':<6} {'R1-stop':<8} {'R1-err%':<8} {'R2-stop':<8} {'R2-err%':<8} {'Escalated':<10} {'Esc%'}")
    print("─" * 88)
    for agent in sorted(stages_data.keys()):
        d = stages_data[agent]
        r1_err = (1 - d["r1_correct"]/d["r1_stop"])*100 if d["r1_stop"] > 0 else 0
        r2_err = (1 - d["r2_correct"]/d["r2_stop"])*100 if d["r2_stop"] > 0 else 0
        esc_pct = d["escalated"]/d["total"]*100
        r2_str = f"{r2_err:.1f}%" if d["r2_stop"] > 0 else "n/a"
        print(f"{agent:<22} {d['total']:<6} {d['r1_stop']:<8} {r1_err:.1f}%{'':<4} {d['r2_stop']:<8} {r2_str:<8} {d['escalated']:<10} {esc_pct:.1f}%")

    # Summary
    totals = {"r1": 0, "r1c": 0, "r2": 0, "r2c": 0, "esc": 0}
    for d in stages_data.values():
        totals["r1"] += d["r1_stop"]
        totals["r1c"] += d["r1_correct"]
        totals["r2"] += d["r2_stop"]
        totals["r2c"] += d["r2_correct"]
        totals["esc"] += d["escalated"]
    print(f"\nSummary: R1 stops {totals['r1']} ({totals['r1c']} correct, {(1-totals['r1c']/totals['r1'])*100:.1f}% error)")
    print(f"         R2 stops {totals['r2']} ({totals['r2c']} correct, {(1-totals['r2c']/totals['r2'])*100:.1f}% error)")
    print(f"         Escalated {totals['esc']}")

    # ── W6: Latency benchmark ──────────────────────────────
    print("\n" + "=" * 60)
    print("W6: Per-stage latency benchmark")
    print("=" * 60)

    texts = [item["text"] for item in sample]

    # R1 keyword latency
    times_r1 = []
    for t in texts:
        start = time.perf_counter()
        keyword_router.route(t)
        times_r1.append(time.perf_counter() - start)

    # R2 embedding latency
    times_r2 = []
    for t in texts:
        start = time.perf_counter()
        embedding_router.route(t)
        times_r2.append(time.perf_counter() - start)

    r1_ms = np.array(times_r1) * 1000
    r2_ms = np.array(times_r2) * 1000

    print(f"\nR1 Keyword Router (n={len(texts)}):")
    print(f"  Mean:   {r1_ms.mean():.3f} ms/query")
    print(f"  Median: {np.median(r1_ms):.3f} ms/query")
    print(f"  P95:    {np.percentile(r1_ms, 95):.3f} ms/query")
    print(f"  Total:  {r1_ms.sum():.1f} ms for {len(texts)} queries")

    print(f"\nR2 Embedding Router (n={len(texts)}):")
    print(f"  Mean:   {r2_ms.mean():.3f} ms/query")
    print(f"  Median: {np.median(r2_ms):.3f} ms/query")
    print(f"  P95:    {np.percentile(r2_ms, 95):.3f} ms/query")
    print(f"  Total:  {r2_ms.sum():.1f} ms for {len(texts)} queries")

    print(f"\nR3 LLM (estimated from API, not benchmarked here):")
    print(f"  Typical: 300-800 ms/query (includes network RTT)")

    print(f"\nFor paper: 'R1 processes queries in {{r1_median:.1f}}ms median, "
          f"R2 in {{r2_median:.1f}}ms; both are 100-1000× faster than "
          f"R3's ~500ms API round-trip.'".format(
              r1_median=np.median(r1_ms),
              r2_median=np.median(r2_ms)))

    # Save results as JSON for reference
    output = {
        "w8_cascade_stages_seed42": {
            agent: {
                "total": d["total"],
                "r1_stop": d["r1_stop"],
                "r1_correct": d["r1_correct"],
                "r1_error_rate": round(1 - d["r1_correct"]/d["r1_stop"], 4) if d["r1_stop"] > 0 else None,
                "r2_stop": d["r2_stop"],
                "r2_correct": d["r2_correct"],
                "r2_error_rate": round(1 - d["r2_correct"]/d["r2_stop"], 4) if d["r2_stop"] > 0 else None,
                "escalated": d["escalated"],
                "escalation_rate": round(d["escalated"]/d["total"], 4),
            }
            for agent, d in sorted(stages_data.items())
        },
        "w6_latency": {
            "r1_keyword_ms": {
                "mean": round(r1_ms.mean(), 3),
                "median": round(float(np.median(r1_ms)), 3),
                "p95": round(float(np.percentile(r1_ms, 95)), 3),
            },
            "r2_embedding_ms": {
                "mean": round(r2_ms.mean(), 3),
                "median": round(float(np.median(r2_ms)), 3),
                "p95": round(float(np.percentile(r2_ms, 95)), 3),
            },
        },
    }
    out_path = RESULTS_DIR / "cascade_stage_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved to {out_path}")


if __name__ == "__main__":
    main()
