"""
evaluate.py — 統一評估所有 Router 變體

跑法：
    cd experiment/
    PYTHONPATH=. ANTHROPIC_API_KEY=sk-... python3 src/evaluate.py

輸出：
    results/metrics.json — 完整數據
    stdout — 摘要表格

設計：
- R1 (Keyword), R2 (Embedding), R4 (Hybrid no-LLM) 跑完整 test set（5,500 queries）
- R3 (LLM) 跑 stratified sample（每 agent 50 筆 = ~400 筆）以控制 API 成本
- R4+LLM (Hybrid with LLM fallback) 只在 R4 低信心時才呼叫 LLM
- 所有結果記錄到 results/metrics.json
"""

import json
import time
import os
import sys
import random
from pathlib import Path
from collections import Counter, defaultdict

# ─────────────────────────────────────────────
# 載入資料
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
# Stratified sample for LLM Router（控制成本）
# ─────────────────────────────────────────────

def stratified_sample(data: list, n_per_agent: int = 50, seed: int = 42) -> list:
    """每個 agent 取 n 筆，固定 seed 確保可重現"""
    rng = random.Random(seed)
    by_agent = defaultdict(list)
    for item in data:
        agent = get_true_agent(item)
        by_agent[agent].append(item)

    sampled = []
    for agent, items in sorted(by_agent.items()):
        if len(items) <= n_per_agent:
            sampled.extend(items)
        else:
            sampled.extend(rng.sample(items, n_per_agent))

    return sampled


# ─────────────────────────────────────────────
# 評估單一 router
# ─────────────────────────────────────────────

def evaluate_router(router_fn, data: list, router_name: str) -> dict:
    """
    跑一個 router 在給定資料上，回傳 metrics。

    Returns:
        {
            "router": str,
            "total": int,
            "correct": int,
            "accuracy": float,
            "per_agent": {agent: {"total": int, "correct": int, "accuracy": float}},
            "confusion": {(true, pred): count},
            "avg_latency_ms": float,     # 只有 LLM router 有意義
            "total_input_tokens": int,
            "total_output_tokens": int,
            "llm_call_rate": float,       # hybrid only
        }
    """
    total = 0
    correct = 0
    per_agent = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = Counter()
    latencies = []
    total_input_tokens = 0
    total_output_tokens = 0
    llm_calls = 0
    stages = Counter()

    for i, item in enumerate(data):
        true_agent = get_true_agent(item)
        result = router_fn(item["text"])
        pred_agent = result["agent"]

        total += 1
        if pred_agent == true_agent:
            correct += 1
            per_agent[true_agent]["correct"] += 1
        per_agent[true_agent]["total"] += 1

        if pred_agent != true_agent:
            confusion[(true_agent, pred_agent)] += 1

        # LLM-specific metrics
        if "latency_ms" in result:
            latencies.append(result["latency_ms"])
        if "input_tokens" in result:
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]

        # Hybrid-specific metrics
        if "stage" in result:
            stages[result["stage"]] += 1
            if result["stage"] == "llm":
                llm_calls += 1

        # Progress
        if (i + 1) % 500 == 0 or i == len(data) - 1:
            print(f"    [{router_name}] {i+1}/{len(data)} — running acc: {correct/total*100:.1f}%")

    # Compute per-agent accuracy
    per_agent_out = {}
    for agent in sorted(per_agent.keys()):
        s = per_agent[agent]
        per_agent_out[agent] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0,
        }

    # Top confusions
    top_confusion = [
        {"true": t, "predicted": p, "count": c}
        for (t, p), c in confusion.most_common(15)
    ]

    return {
        "router": router_name,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "per_agent": per_agent_out,
        "top_confusion": top_confusion,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "llm_call_rate": round(llm_calls / total, 4) if total > 0 else 0,
        "stages": dict(stages) if stages else {},
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    from src.routers import keyword_router, embedding_router, hybrid_router

    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))

    print("=" * 60)
    print("  Hybrid Router Experiment — Full Evaluation")
    print("=" * 60)
    print(f"  Test set: {len(TEST_DATA)} queries")
    print(f"  API key:  {'✅' if has_api_key else '❌ (skipping LLM router)'}")
    print()

    all_results = {}

    # ── R1: Keyword ──
    print("[R1] Keyword Router...")
    t0 = time.time()
    r1 = evaluate_router(keyword_router.route, TEST_DATA, "R1_keyword")
    r1["wall_time_sec"] = round(time.time() - t0, 1)
    r1["cost_usd"] = 0.0
    all_results["R1_keyword"] = r1
    print(f"  → {r1['accuracy']*100:.1f}% ({r1['wall_time_sec']}s)\n")

    # ── R2: Embedding ──
    print("[R2] Embedding Router (TF-IDF)...")
    t0 = time.time()
    r2 = evaluate_router(embedding_router.route, TEST_DATA, "R2_embedding")
    r2["wall_time_sec"] = round(time.time() - t0, 1)
    r2["cost_usd"] = 0.0
    all_results["R2_embedding"] = r2
    print(f"  → {r2['accuracy']*100:.1f}% ({r2['wall_time_sec']}s)\n")

    # ── R4: Hybrid (no LLM) ──
    print("[R4] Hybrid Router (keyword → embedding, no LLM fallback)...")
    hybrid_router.reset_stats()
    t0 = time.time()

    def hybrid_no_llm(query):
        return hybrid_router.route(query, keyword_threshold=1.5, embed_threshold=0.05, use_llm_fallback=False)

    r4 = evaluate_router(hybrid_no_llm, TEST_DATA, "R4_hybrid_no_llm")
    r4["wall_time_sec"] = round(time.time() - t0, 1)
    r4["cost_usd"] = 0.0
    r4["hybrid_stats"] = hybrid_router.get_stats()
    all_results["R4_hybrid_no_llm"] = r4
    print(f"  → {r4['accuracy']*100:.1f}% ({r4['wall_time_sec']}s)")
    hs = r4["hybrid_stats"]
    print(f"    Keyword accepted: {hs['keyword_accepted']}/{hs['total_queries']} ({hs['keyword_accepted']/hs['total_queries']*100:.1f}%)")
    print(f"    Embedding fallback: {hs['embedding_accepted']}/{hs['total_queries']} ({hs['embedding_accepted']/hs['total_queries']*100:.1f}%)\n")

    # ── R3: LLM (stratified sample) ──
    if has_api_key:
        from src.routers import llm_router

        llm_sample = stratified_sample(TEST_DATA, n_per_agent=20, seed=42)
        print(f"[R3] LLM Router (Haiku, stratified sample: {len(llm_sample)} queries)...")
        llm_router.reset_stats()
        t0 = time.time()
        r3 = evaluate_router(llm_router.route, llm_sample, "R3_llm")
        r3["wall_time_sec"] = round(time.time() - t0, 1)

        stats = llm_router.get_stats()
        # Haiku pricing: $0.80/M input, $4/M output
        cost = (stats["total_input_tokens"] * 0.80 + stats["total_output_tokens"] * 4.0) / 1_000_000
        r3["cost_usd"] = round(cost, 4)
        r3["llm_stats"] = stats
        r3["sample_size"] = len(llm_sample)
        r3["note"] = "Evaluated on stratified sample (50/agent), not full test set"
        all_results["R3_llm"] = r3
        print(f"  → {r3['accuracy']*100:.1f}% ({r3['wall_time_sec']}s)")
        print(f"    Cost: ${r3['cost_usd']:.4f} ({stats['total_input_tokens']} in + {stats['total_output_tokens']} out tokens)")
        print(f"    Avg latency: {r3['avg_latency_ms']}ms/query\n")

        # ── R4+LLM: Hybrid with LLM fallback (on same sample) ──
        print(f"[R4+LLM] Hybrid Router with LLM fallback (same sample: {len(llm_sample)} queries)...")
        hybrid_router.reset_stats()
        llm_router.reset_stats()

        def hybrid_with_llm(query):
            return hybrid_router.route(query, keyword_threshold=1.5, embed_threshold=0.15, use_llm_fallback=True)

        r4llm = evaluate_router(hybrid_with_llm, llm_sample, "R4_hybrid_with_llm")
        r4llm["wall_time_sec"] = round(time.time() - t0, 1)

        llm_stats = llm_router.get_stats()
        cost_llm = (llm_stats["total_input_tokens"] * 0.80 + llm_stats["total_output_tokens"] * 4.0) / 1_000_000
        r4llm["cost_usd"] = round(cost_llm, 4)
        r4llm["hybrid_stats"] = hybrid_router.get_stats()
        r4llm["llm_stats"] = llm_stats
        r4llm["sample_size"] = len(llm_sample)
        all_results["R4_hybrid_with_llm"] = r4llm
        hs2 = r4llm["hybrid_stats"]
        print(f"  → {r4llm['accuracy']*100:.1f}%")
        print(f"    Keyword accepted: {hs2['keyword_accepted']}/{hs2['total_queries']} ({hs2['keyword_accepted']/hs2['total_queries']*100:.1f}%)")
        print(f"    Embedding accepted: {hs2['embedding_accepted']}/{hs2['total_queries']} ({hs2['embedding_accepted']/hs2['total_queries']*100:.1f}%)")
        print(f"    LLM fallback: {hs2['llm_fallback']}/{hs2['total_queries']} ({hs2['llm_fallback']/hs2['total_queries']*100:.1f}%)")
        print(f"    LLM cost: ${r4llm['cost_usd']:.4f}\n")

    # ── Summary Table ──
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  {'Router':<25s} {'Accuracy':>10s} {'Cost':>10s} {'LLM calls':>10s} {'Note':>15s}")
    print("  " + "-" * 72)
    for name, r in all_results.items():
        note = f"n={r.get('sample_size', r['total'])}"
        llm_info = f"{r.get('llm_call_rate', 0)*100:.0f}%" if r.get('llm_call_rate', 0) > 0 else "0%"
        print(f"  {name:<25s} {r['accuracy']*100:>9.1f}% ${r['cost_usd']:>8.4f} {llm_info:>10s} {note:>15s}")

    # ── Save ──
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
