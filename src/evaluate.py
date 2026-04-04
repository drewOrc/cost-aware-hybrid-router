"""
evaluate.py — 統一評估所有 Router 變體（multi-seed + McNemar + SetFit）

跑法：
    # 快速（小樣本，單 seed，不跑 SetFit）
    PYTHONPATH=. ANTHROPIC_API_KEY=sk-... python3 src/evaluate.py \
        --llm-n 160 --seeds 42

    # 完整 (paper-grade)：n=1000 LLM eval × 3 seeds + SetFit baseline
    PYTHONPATH=. ANTHROPIC_API_KEY=sk-... python3 src/evaluate.py \
        --llm-n 1000 --seeds 42 43 44 --setfit --use-tuned-thresholds

輸出：
    results/metrics.json        — 每 seed 的完整數據
    results/summary.json        — 跨 seed 彙整 (mean ± std) + McNemar
    stdout                      — 摘要表格

流程：
- R1, R2, R4-no-LLM 跑完整 test set（5,500 queries）— 無隨機性，只需跑一次
- R3 LLM、R4+LLM、SetFit 在 LLM sample（stratified，每 agent n筆）上跑
  每個 seed 抽不同樣本 → 得到多組 (R3, R4+LLM) 結果 → mean ± std + McNemar
- Thresholds: --use-tuned-thresholds 會讀取 results/tuned_thresholds.json
"""

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / "clinc150" / "test.json") as f:
    TEST_DATA = json.load(f)
with open(DATA_DIR / "clinc150" / "intent_names.json") as f:
    INTENT_NAMES = json.load(f)
with open(DATA_DIR / "intent_to_agent.json") as f:
    raw = json.load(f)
    MAPPING = {k: v for k, v in raw.items() if k != "_meta"}


def get_true_agent(item: dict) -> str:
    return MAPPING[INTENT_NAMES[item["intent"]]]


# ─────────────────────────────────────────────
# Stratified sampling
# ─────────────────────────────────────────────

def stratified_sample(data: list, n_per_agent: int, seed: int) -> list:
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


# ─────────────────────────────────────────────
# Router evaluation (returns correct_flags + metrics)
# ─────────────────────────────────────────────

def evaluate_router(router_fn, data: list, name: str, verbose: bool = True) -> dict:
    total = 0
    correct = 0
    correct_flags = []
    per_agent = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = Counter()
    latencies = []
    total_in = 0
    total_out = 0
    stages = Counter()

    for i, item in enumerate(data):
        true_agent = get_true_agent(item)
        result = router_fn(item["text"])
        pred = result["agent"]
        is_correct = (pred == true_agent)

        total += 1
        correct_flags.append(1 if is_correct else 0)
        if is_correct:
            correct += 1
            per_agent[true_agent]["correct"] += 1
        per_agent[true_agent]["total"] += 1
        if not is_correct:
            confusion[(true_agent, pred)] += 1

        if "latency_ms" in result:
            latencies.append(result["latency_ms"])
        if "input_tokens" in result:
            total_in += result["input_tokens"]
            total_out += result["output_tokens"]
        if "stage" in result:
            stages[result["stage"]] += 1

        if verbose and ((i + 1) % 500 == 0 or i == len(data) - 1):
            print(f"    [{name}] {i+1}/{len(data)} — running acc: {correct/total*100:.1f}%")

    per_agent_out = {
        a: {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0,
        }
        for a, s in sorted(per_agent.items())
    }

    return {
        "router": name,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "correct_flags": correct_flags,
        "per_agent": per_agent_out,
        "top_confusion": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in confusion.most_common(15)
        ],
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "stages": dict(stages) if stages else {},
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-n", type=int, default=160,
                   help="Total LLM eval sample size (stratified across 8 classes).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Random seeds for stratified sampling.")
    p.add_argument("--setfit", action="store_true", help="Include SetFit baseline.")
    p.add_argument("--use-tuned-thresholds", action="store_true",
                   help="Load thresholds from results/tuned_thresholds.json")
    p.add_argument("--skip-full", action="store_true",
                   help="Skip R1/R2/R4-no-llm full test set (for quick iteration).")
    p.add_argument("--output", type=str, default="metrics.json",
                   help="Output filename under results/ (default: metrics.json)")
    return p.parse_args()


def main():
    args = parse_args()
    from src.routers import keyword_router, embedding_router, hybrid_router

    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    n_per_agent = max(1, args.llm_n // 8)  # 7 agents + oos = 8 classes

    # ── Load tuned thresholds if requested ──
    kt_no_llm, et_no_llm = 1.5, 0.05
    kt_with_llm, et_with_llm = 1.5, 0.15
    threshold_source = "manual defaults"
    if args.use_tuned_thresholds:
        tuned_path = RESULTS_DIR / "tuned_thresholds.json"
        if tuned_path.exists():
            with open(tuned_path) as f:
                tuned = json.load(f)
            kt_no_llm = tuned["no_llm"]["keyword_threshold"]
            et_no_llm = tuned["no_llm"]["embed_threshold"]
            kt_with_llm = tuned["with_llm"]["keyword_threshold"]
            et_with_llm = tuned["with_llm"]["embed_threshold"]
            threshold_source = f"tuned on val set ({tuned_path.name})"
        else:
            print(f"⚠️  {tuned_path} not found — run `python src/tune.py` first. Falling back to defaults.")

    print("=" * 70)
    print("  Hybrid Router Experiment — Full Evaluation")
    print("=" * 70)
    print(f"  Test set:       {len(TEST_DATA)} queries")
    print(f"  LLM sample:     {n_per_agent}/agent × 8 ≈ {n_per_agent*8} queries")
    print(f"  Seeds:          {args.seeds}")
    print(f"  SetFit:         {'yes' if args.setfit else 'no'}")
    print(f"  Thresholds:     {threshold_source}")
    print(f"                  no-LLM: kt={kt_no_llm}, et={et_no_llm}")
    print(f"                  with-LLM: kt={kt_with_llm}, et={et_with_llm}")
    print(f"  API key:        {'✅' if has_api_key else '❌ (skipping LLM)'}")
    print()

    all_results: dict = {"config": {
        "llm_n": args.llm_n,
        "n_per_agent": n_per_agent,
        "seeds": args.seeds,
        "setfit": args.setfit,
        "threshold_source": threshold_source,
        "thresholds_no_llm": {"kt": kt_no_llm, "et": et_no_llm},
        "thresholds_with_llm": {"kt": kt_with_llm, "et": et_with_llm},
    }}

    # ── R1, R2, R4-no-LLM on full test (deterministic, seed-independent) ──
    if not args.skip_full:
        print("[R1] Keyword Router...")
        t0 = time.time()
        r1 = evaluate_router(keyword_router.route, TEST_DATA, "R1_keyword")
        r1["wall_time_sec"] = round(time.time() - t0, 1)
        r1["cost_usd"] = 0.0
        all_results["R1_keyword"] = r1
        print(f"  → {r1['accuracy']*100:.1f}% ({r1['wall_time_sec']}s)\n")

        print("[R2] Embedding Router (TF-IDF)...")
        t0 = time.time()
        r2 = evaluate_router(embedding_router.route, TEST_DATA, "R2_embedding")
        r2["wall_time_sec"] = round(time.time() - t0, 1)
        r2["cost_usd"] = 0.0
        all_results["R2_embedding"] = r2
        print(f"  → {r2['accuracy']*100:.1f}% ({r2['wall_time_sec']}s)\n")

        print(f"[R4] Hybrid Router (no LLM, kt={kt_no_llm}, et={et_no_llm})...")
        hybrid_router.reset_stats()
        t0 = time.time()

        def hybrid_no_llm(q):
            return hybrid_router.route(q, keyword_threshold=kt_no_llm,
                                       embed_threshold=et_no_llm, use_llm_fallback=False)

        r4 = evaluate_router(hybrid_no_llm, TEST_DATA, "R4_hybrid_no_llm")
        r4["wall_time_sec"] = round(time.time() - t0, 1)
        r4["cost_usd"] = 0.0
        r4["hybrid_stats"] = hybrid_router.get_stats()
        all_results["R4_hybrid_no_llm"] = r4
        print(f"  → {r4['accuracy']*100:.1f}% ({r4['wall_time_sec']}s)\n")

    # ── SetFit baseline on full test ──
    if args.setfit:
        try:
            from src.routers import setfit_router
            print("[SetFit] Loading model + evaluating on full test set...")
            t0 = time.time()
            rs = evaluate_router(setfit_router.route, TEST_DATA, "SetFit_baseline")
            rs["wall_time_sec"] = round(time.time() - t0, 1)
            rs["cost_usd"] = 0.0
            all_results["SetFit_baseline"] = rs
            print(f"  → {rs['accuracy']*100:.1f}% ({rs['wall_time_sec']}s)\n")
        except Exception as e:
            print(f"⚠️  SetFit failed: {e}")

    # ── R3, R4+LLM, McNemar — per seed ──
    if has_api_key:
        from src.routers import llm_router
        seed_results = {"R3_llm": [], "R4_hybrid_with_llm": [], "mcnemar_r4llm_vs_r3": []}

        for seed in args.seeds:
            sample = stratified_sample(TEST_DATA, n_per_agent=n_per_agent, seed=seed)
            print(f"\n─── Seed {seed} (n={len(sample)}) ───")

            print(f"[R3] LLM Router (Haiku)...")
            llm_router.reset_stats()
            t0 = time.time()
            r3 = evaluate_router(llm_router.route, sample, f"R3_llm_seed{seed}")
            r3["wall_time_sec"] = round(time.time() - t0, 1)
            stats = llm_router.get_stats()
            r3["cost_usd"] = round((stats["total_input_tokens"] * 0.80
                                    + stats["total_output_tokens"] * 4.0) / 1_000_000, 4)
            r3["seed"] = seed
            r3["sample_size"] = len(sample)
            seed_results["R3_llm"].append(r3)
            print(f"  → {r3['accuracy']*100:.1f}%  cost=${r3['cost_usd']:.4f}")

            print(f"[R4+LLM] Hybrid with LLM fallback (kt={kt_with_llm}, et={et_with_llm})...")
            hybrid_router.reset_stats()
            llm_router.reset_stats()
            t0 = time.time()

            def hybrid_with_llm(q):
                return hybrid_router.route(q, keyword_threshold=kt_with_llm,
                                           embed_threshold=et_with_llm, use_llm_fallback=True)

            r4llm = evaluate_router(hybrid_with_llm, sample, f"R4_hybrid_with_llm_seed{seed}")
            r4llm["wall_time_sec"] = round(time.time() - t0, 1)
            stats2 = llm_router.get_stats()
            r4llm["cost_usd"] = round((stats2["total_input_tokens"] * 0.80
                                       + stats2["total_output_tokens"] * 4.0) / 1_000_000, 4)
            r4llm["hybrid_stats"] = hybrid_router.get_stats()
            r4llm["seed"] = seed
            r4llm["sample_size"] = len(sample)
            seed_results["R4_hybrid_with_llm"].append(r4llm)
            print(f"  → {r4llm['accuracy']*100:.1f}%  cost=${r4llm['cost_usd']:.4f}  "
                  f"LLM calls={r4llm['hybrid_stats']['llm_fallback']}/{len(sample)}")

            # McNemar: R4+LLM vs R3 on same sample
            from src.stats import mcnemar_test
            mc = mcnemar_test(r3["correct_flags"], r4llm["correct_flags"])
            mc["seed"] = seed
            seed_results["mcnemar_r4llm_vs_r3"].append(mc)
            print(f"  McNemar (R4+LLM vs R3): p={mc['p_value']:.4f}  "
                  f"significant={mc['significant_at_0.05']}  "
                  f"delta={mc['delta']*100:+.1f}pp")

        all_results["seed_results"] = seed_results

        # ── Aggregate ──
        from src.stats import aggregate_seeds, wilson_ci
        r3_accs = [r["accuracy"] for r in seed_results["R3_llm"]]
        r4llm_accs = [r["accuracy"] for r in seed_results["R4_hybrid_with_llm"]]
        r3_costs = [r["cost_usd"] for r in seed_results["R3_llm"]]
        r4llm_costs = [r["cost_usd"] for r in seed_results["R4_hybrid_with_llm"]]

        all_results["aggregated"] = {
            "R3_llm": {
                "accuracy": aggregate_seeds(r3_accs),
                "cost_usd": aggregate_seeds(r3_costs),
            },
            "R4_hybrid_with_llm": {
                "accuracy": aggregate_seeds(r4llm_accs),
                "cost_usd": aggregate_seeds(r4llm_costs),
            },
            "wilson_ci_R3": wilson_ci(seed_results["R3_llm"][0]["correct"],
                                      seed_results["R3_llm"][0]["total"]),
            "wilson_ci_R4LLM": wilson_ci(seed_results["R4_hybrid_with_llm"][0]["correct"],
                                          seed_results["R4_hybrid_with_llm"][0]["total"]),
            "mcnemar_significant_count": sum(
                1 for m in seed_results["mcnemar_r4llm_vs_r3"] if m["significant_at_0.05"]
            ),
            "n_seeds": len(args.seeds),
        }

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    if "R1_keyword" in all_results:
        for key in ["R1_keyword", "R2_embedding", "R4_hybrid_no_llm", "SetFit_baseline"]:
            if key in all_results:
                r = all_results[key]
                print(f"  {key:<25s} acc={r['accuracy']*100:>5.1f}%  (n={r['total']})")
    if "aggregated" in all_results:
        ag = all_results["aggregated"]
        print(f"  R3_llm (multi-seed)        acc={ag['R3_llm']['accuracy']['mean']*100:.1f}% "
              f"± {ag['R3_llm']['accuracy']['std']*100:.1f}pp  "
              f"(n_seeds={ag['n_seeds']})")
        print(f"  R4_hybrid_with_llm (ms)    acc={ag['R4_hybrid_with_llm']['accuracy']['mean']*100:.1f}% "
              f"± {ag['R4_hybrid_with_llm']['accuracy']['std']*100:.1f}pp")
        print(f"  McNemar significant in {ag['mcnemar_significant_count']}/{ag['n_seeds']} seeds")

    # ── Save ──
    # Strip correct_flags to keep file small (save separately)
    import copy
    to_save = copy.deepcopy(all_results)
    for key in ["R1_keyword", "R2_embedding", "R4_hybrid_no_llm", "SetFit_baseline"]:
        if key in to_save:
            to_save[key].pop("correct_flags", None)
    if "seed_results" in to_save:
        for rlist in to_save["seed_results"].values():
            for r in rlist:
                if isinstance(r, dict):
                    r.pop("correct_flags", None)

    out_path = RESULTS_DIR / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
