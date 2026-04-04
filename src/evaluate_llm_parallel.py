"""
evaluate_llm_parallel.py — Parallel LLM evaluation for R3 and R4+LLM

Hits Anthropic API concurrently via ThreadPoolExecutor to fit within time budgets.
Runs R3 (all queries) and R4+LLM (cascade then LLM fallback) across multiple seeds.

Usage:
    ANTHROPIC_API_KEY=sk-... PYTHONPATH=. python3 src/evaluate_llm_parallel.py \
        --llm-n 1000 --seeds 42 43 44 --workers 10
"""

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

with open(DATA_DIR / "clinc150" / "test.json") as f:
    TEST_DATA = json.load(f)
with open(DATA_DIR / "clinc150" / "intent_names.json") as f:
    INTENT_NAMES = json.load(f)
with open(DATA_DIR / "intent_to_agent.json") as f:
    raw = json.load(f)
    MAPPING = {k: v for k, v in raw.items() if k != "_meta"}


def get_true_agent(item):
    return MAPPING[INTENT_NAMES[item["intent"]]]


def stratified_sample(data, n_per_agent, seed):
    rng = random.Random(seed)
    by_agent = defaultdict(list)
    for item in data:
        by_agent[get_true_agent(item)].append(item)
    sampled = []
    for agent, items in sorted(by_agent.items()):
        if len(items) <= n_per_agent:
            sampled.extend(items)
        else:
            sampled.extend(rng.sample(items, n_per_agent))
    return sampled


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-n", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--use-tuned-thresholds", action="store_true")
    p.add_argument("--output", type=str, default="metrics_llm.json")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    from src.routers import keyword_router, embedding_router, llm_router
    from src.stats import mcnemar_test, wilson_ci, aggregate_seeds

    # Load tuned thresholds
    kt, et = 0.5, 0.10  # with_llm defaults
    if args.use_tuned_thresholds:
        with open(RESULTS_DIR / "tuned_thresholds.json") as f:
            tuned = json.load(f)
        kt = tuned["with_llm"]["keyword_threshold"]
        et = tuned["with_llm"]["embed_threshold"]

    n_per_agent = max(1, args.llm_n // 8)

    print(f"Config: llm_n={args.llm_n}, seeds={args.seeds}, workers={args.workers}")
    print(f"Thresholds (R4+LLM): kt={kt}, et={et}")
    print(f"n_per_agent={n_per_agent}")
    print()

    # ── Build unified LLM task list ──
    # For each seed: R3 needs LLM for ALL queries; R4+LLM only for uncertain.
    # Pre-compute R1/R2 cascade to identify R4+LLM queries + fast predictions.
    tasks = []  # list of (task_id, query) — unique LLM calls
    task_cache = {}  # query → task_id (dedupe identical queries)
    seed_data = {}  # seed → dict

    for seed in args.seeds:
        sample = stratified_sample(TEST_DATA, n_per_agent=n_per_agent, seed=seed)
        true_agents = [get_true_agent(x) for x in sample]
        queries = [x["text"] for x in sample]

        # Pre-compute R1+R2 cascade (fast, sequential)
        cascade_preds = []  # {"predicted": str, "needs_llm": bool, "stage": str}
        t0 = time.time()
        for q in queries:
            kw = keyword_router.route(q)
            if kw["confidence"] >= kt:
                cascade_preds.append({"predicted": kw["agent"], "needs_llm": False, "stage": "keyword"})
                continue
            emb = embedding_router.route(q)
            if emb["confidence"] >= et:
                cascade_preds.append({"predicted": emb["agent"], "needs_llm": False, "stage": "embedding"})
            else:
                cascade_preds.append({"predicted": None, "needs_llm": True, "stage": "llm"})
        cascade_time = time.time() - t0

        r4_llm_count = sum(1 for c in cascade_preds if c["needs_llm"])
        print(f"[seed={seed}] cascade done in {cascade_time:.1f}s, R4+LLM needs {r4_llm_count}/{len(queries)} LLM calls")

        # Queue up LLM tasks: R3 needs all queries; R4+LLM only uncertain ones
        r3_task_ids = []
        r4_task_ids = []  # None if not needed
        for i, q in enumerate(queries):
            # R3 task
            if q not in task_cache:
                tid = len(tasks)
                task_cache[q] = tid
                tasks.append((tid, q))
            r3_task_ids.append(task_cache[q])
            # R4+LLM task (reuse same tid since same query)
            if cascade_preds[i]["needs_llm"]:
                r4_task_ids.append(task_cache[q])
            else:
                r4_task_ids.append(None)

        seed_data[seed] = {
            "queries": queries,
            "true_agents": true_agents,
            "cascade_preds": cascade_preds,
            "r3_task_ids": r3_task_ids,
            "r4_task_ids": r4_task_ids,
        }

    print(f"\nTotal unique LLM calls to make: {len(tasks)}")
    est_min = len(tasks) * 0.5 / args.workers / 60
    print(f"Estimated wall-time with {args.workers} workers: ~{est_min:.1f} min\n")

    # ── Execute LLM calls in parallel ──
    results = {}  # task_id → {"agent": str, "input_tokens": int, "output_tokens": int}
    llm_router.reset_stats()
    t0 = time.time()
    done = 0

    def call_one(tid, query):
        # Exponential backoff on 429 or transient errors
        delays = [1, 2, 4, 8, 16, 32]
        last_err = None
        for d in delays:
            try:
                r = llm_router.route(query)
                return tid, r, None
            except Exception as e:
                last_err = str(e)
                if "429" in last_err or "rate_limit" in last_err or "overloaded" in last_err.lower():
                    time.sleep(d)
                    continue
                time.sleep(d)
        return tid, None, last_err

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(call_one, tid, q) for tid, q in tasks]
        for fut in as_completed(futures):
            tid, r, err = fut.result()
            done += 1
            if err:
                print(f"  task {tid} final failure: {err[:120]}")
                r = {"agent": "oos", "input_tokens": 0, "output_tokens": 0, "latency_ms": 0}
            results[tid] = r
            if done % 100 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(tasks)} done, {rate:.1f} req/s, ETA {eta:.0f}s")

    total_time = time.time() - t0
    total_in = sum(r["input_tokens"] for r in results.values())
    total_out = sum(r["output_tokens"] for r in results.values())
    total_cost = (total_in * 0.80 + total_out * 4.0) / 1_000_000
    print(f"\nAll LLM calls done in {total_time:.0f}s, {len(tasks)} calls, ${total_cost:.4f}\n")

    # ── Assemble per-seed results ──
    seed_results = {"R3_llm": [], "R4_hybrid_with_llm": [], "mcnemar_r4llm_vs_r3": []}

    for seed in args.seeds:
        sd = seed_data[seed]
        queries = sd["queries"]
        true_agents = sd["true_agents"]
        cascade_preds = sd["cascade_preds"]

        # R3 predictions
        r3_preds = [results[tid]["agent"] for tid in sd["r3_task_ids"]]
        r3_flags = [1 if p == t else 0 for p, t in zip(r3_preds, true_agents)]
        r3_correct = sum(r3_flags)
        r3_acc = r3_correct / len(r3_preds)

        # R3 per-agent + confusion
        per_agent_r3 = defaultdict(lambda: {"total": 0, "correct": 0})
        confusion_r3 = Counter()
        for p, t in zip(r3_preds, true_agents):
            per_agent_r3[t]["total"] += 1
            if p == t:
                per_agent_r3[t]["correct"] += 1
            else:
                confusion_r3[(t, p)] += 1

        # R4+LLM predictions
        r4_preds = []
        for i in range(len(queries)):
            if sd["r4_task_ids"][i] is not None:
                r4_preds.append(results[sd["r4_task_ids"][i]]["agent"])
            else:
                r4_preds.append(cascade_preds[i]["predicted"])
        r4_flags = [1 if p == t else 0 for p, t in zip(r4_preds, true_agents)]
        r4_correct = sum(r4_flags)
        r4_acc = r4_correct / len(r4_preds)

        per_agent_r4 = defaultdict(lambda: {"total": 0, "correct": 0})
        confusion_r4 = Counter()
        for p, t in zip(r4_preds, true_agents):
            per_agent_r4[t]["total"] += 1
            if p == t:
                per_agent_r4[t]["correct"] += 1
            else:
                confusion_r4[(t, p)] += 1

        # Stage counts for R4+LLM
        stages = Counter(c["stage"] for c in cascade_preds)

        # Cost breakdown per seed (approximate — full R3 call set, partial R4)
        r3_cost_seed = sum(
            (results[tid]["input_tokens"] * 0.80 + results[tid]["output_tokens"] * 4.0) / 1_000_000
            for tid in sd["r3_task_ids"]
        )
        r4_llm_ids_seed = [tid for tid in sd["r4_task_ids"] if tid is not None]
        r4_cost_seed = sum(
            (results[tid]["input_tokens"] * 0.80 + results[tid]["output_tokens"] * 4.0) / 1_000_000
            for tid in r4_llm_ids_seed
        )

        r3_record = {
            "router": f"R3_llm_seed{seed}",
            "seed": seed,
            "total": len(queries),
            "correct": r3_correct,
            "accuracy": round(r3_acc, 4),
            "correct_flags": r3_flags,
            "per_agent": {a: {"total": v["total"], "correct": v["correct"],
                              "accuracy": round(v["correct"]/v["total"], 4) if v["total"] else 0}
                          for a, v in sorted(per_agent_r3.items())},
            "top_confusion": [{"true": t, "predicted": p, "count": c}
                              for (t, p), c in confusion_r3.most_common(15)],
            "cost_usd": round(r3_cost_seed, 4),
            "sample_size": len(queries),
        }
        r4_record = {
            "router": f"R4_hybrid_with_llm_seed{seed}",
            "seed": seed,
            "total": len(queries),
            "correct": r4_correct,
            "accuracy": round(r4_acc, 4),
            "correct_flags": r4_flags,
            "per_agent": {a: {"total": v["total"], "correct": v["correct"],
                              "accuracy": round(v["correct"]/v["total"], 4) if v["total"] else 0}
                          for a, v in sorted(per_agent_r4.items())},
            "top_confusion": [{"true": t, "predicted": p, "count": c}
                              for (t, p), c in confusion_r4.most_common(15)],
            "cost_usd": round(r4_cost_seed, 4),
            "stages": dict(stages),
            "llm_fallback_count": len(r4_llm_ids_seed),
            "llm_call_rate": round(len(r4_llm_ids_seed) / len(queries), 4),
            "sample_size": len(queries),
        }

        seed_results["R3_llm"].append(r3_record)
        seed_results["R4_hybrid_with_llm"].append(r4_record)

        mc = mcnemar_test(r3_flags, r4_flags)
        mc["seed"] = seed
        seed_results["mcnemar_r4llm_vs_r3"].append(mc)

        print(f"[seed={seed}]")
        print(f"  R3:     acc={r3_acc*100:.1f}% ({r3_correct}/{len(queries)})  cost=${r3_cost_seed:.4f}")
        print(f"  R4+LLM: acc={r4_acc*100:.1f}% ({r4_correct}/{len(queries)})  cost=${r4_cost_seed:.4f}  "
              f"LLM={len(r4_llm_ids_seed)}/{len(queries)} ({len(r4_llm_ids_seed)/len(queries)*100:.1f}%)")
        print(f"  McNemar: p={mc['p_value']:.4f}  sig={mc['significant_at_0.05']}  "
              f"delta={mc['delta']*100:+.1f}pp")

    # ── Aggregate ──
    r3_accs = [r["accuracy"] for r in seed_results["R3_llm"]]
    r4_accs = [r["accuracy"] for r in seed_results["R4_hybrid_with_llm"]]
    r3_costs = [r["cost_usd"] for r in seed_results["R3_llm"]]
    r4_costs = [r["cost_usd"] for r in seed_results["R4_hybrid_with_llm"]]

    agg = {
        "R3_llm": {
            "accuracy": aggregate_seeds(r3_accs),
            "cost_usd": aggregate_seeds(r3_costs),
        },
        "R4_hybrid_with_llm": {
            "accuracy": aggregate_seeds(r4_accs),
            "cost_usd": aggregate_seeds(r4_costs),
        },
        "wilson_ci_R3": wilson_ci(seed_results["R3_llm"][0]["correct"],
                                   seed_results["R3_llm"][0]["total"]),
        "wilson_ci_R4LLM": wilson_ci(seed_results["R4_hybrid_with_llm"][0]["correct"],
                                      seed_results["R4_hybrid_with_llm"][0]["total"]),
        "mcnemar_significant_count": sum(1 for m in seed_results["mcnemar_r4llm_vs_r3"]
                                         if m["significant_at_0.05"]),
        "n_seeds": len(args.seeds),
    }

    print("\n" + "=" * 60)
    print("  Aggregate (across seeds)")
    print("=" * 60)
    print(f"  R3_llm:             {agg['R3_llm']['accuracy']['mean']*100:.1f}% ± "
          f"{agg['R3_llm']['accuracy']['std']*100:.1f}pp")
    print(f"  R4_hybrid_with_llm: {agg['R4_hybrid_with_llm']['accuracy']['mean']*100:.1f}% ± "
          f"{agg['R4_hybrid_with_llm']['accuracy']['std']*100:.1f}pp")
    print(f"  McNemar significant: {agg['mcnemar_significant_count']}/{agg['n_seeds']} seeds")
    print(f"  Total cost:          ${sum(r3_costs) + sum(r4_costs):.4f}")

    # ── Save ──
    output = {
        "config": {
            "llm_n": args.llm_n,
            "n_per_agent": n_per_agent,
            "seeds": args.seeds,
            "workers": args.workers,
            "thresholds_with_llm": {"kt": kt, "et": et},
        },
        "seed_results": seed_results,
        "aggregated": agg,
        "total_cost_usd": round(sum(r3_costs) + sum(r4_costs), 4),
        "wall_time_sec": round(total_time, 1),
    }
    # Strip correct_flags from saved copy (keep file small)
    import copy
    to_save = copy.deepcopy(output)
    for rlist in to_save["seed_results"].values():
        for r in rlist:
            if isinstance(r, dict):
                r.pop("correct_flags", None)

    out_path = RESULTS_DIR / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
