"""
Hybrid Router (R4) — Keyword first → confidence threshold → fallback

核心主張：
keyword router 處理大部分查詢已足夠，透過分析失敗模式設計 hybrid 策略，
可用最少 LLM 呼叫達到接近 full-LLM 準確率。

設計：
1. 先跑 Keyword Router (R1)
2. 如果 R1 confidence ≥ high_threshold → 直接採用（零額外成本）
3. 如果 R1 confidence < high_threshold → fallback 到 Embedding Router (R2)
4. 如果 R2 confidence ≥ embed_threshold → 採用 R2 結果
5. 否則 → fallback 到 LLM Router (R3)（最貴，但最準）

Threshold tuning：
- high_threshold 和 embed_threshold 在 validation set 上 grid search
- 目標：最小化 LLM call rate，同時維持 accuracy ≥ R3 × 0.95

變體：
- hybrid_no_llm：只做 R1 → R2 fallback（完全不用 LLM，成本 = 0）
- hybrid_full：R1 → R2 → R3 完整 pipeline
"""

import json
from pathlib import Path
from typing import Optional

from src.routers import keyword_router, embedding_router


# ─────────────────────────────────────────────
# 預設 threshold（可由 evaluate.py 的 grid search 覆蓋）
# ─────────────────────────────────────────────

DEFAULT_KEYWORD_THRESHOLD = 1.5   # R1 confidence ≥ 此值 → 直接採用
DEFAULT_EMBED_THRESHOLD = 0.15    # R2 confidence ≥ 此值 → 採用 R2


# ─────────────────────────────────────────────
# 統計
# ─────────────────────────────────────────────

_stats = {
    "total_queries": 0,
    "keyword_accepted": 0,    # R1 直接通過
    "embedding_accepted": 0,  # R1 不確定 → R2 接手
    "llm_fallback": 0,        # R1 + R2 都不確定 → R3
}


def get_stats() -> dict:
    return _stats.copy()


def reset_stats():
    global _stats
    _stats = {
        "total_queries": 0,
        "keyword_accepted": 0,
        "embedding_accepted": 0,
        "llm_fallback": 0,
    }


# ─────────────────────────────────────────────
# Router 實作
# ─────────────────────────────────────────────

def route(
    query: str,
    keyword_threshold: float = DEFAULT_KEYWORD_THRESHOLD,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
    use_llm_fallback: bool = False,
) -> dict:
    """
    Hybrid routing: R1 → R2 → (optional) R3

    Args:
        query: 用戶查詢
        keyword_threshold: R1 confidence 門檻
        embed_threshold: R2 confidence 門檻
        use_llm_fallback: 是否在 R1+R2 都不確定時 fallback 到 LLM

    Returns:
        {
            "agent": str,
            "confidence": float,
            "scores": dict,
            "method": "hybrid",
            "stage": "keyword" | "embedding" | "llm",
            "keyword_result": dict,
            "embedding_result": dict | None,
            "llm_result": dict | None,
        }
    """
    _stats["total_queries"] += 1

    # Stage 1: Keyword Router
    kw_result = keyword_router.route(query)

    if kw_result["confidence"] >= keyword_threshold:
        _stats["keyword_accepted"] += 1
        return {
            "agent": kw_result["agent"],
            "confidence": kw_result["confidence"],
            "scores": kw_result["scores"],
            "method": "hybrid",
            "stage": "keyword",
            "keyword_result": kw_result,
            "embedding_result": None,
            "llm_result": None,
        }

    # Stage 2: Embedding Router
    emb_result = embedding_router.route(query)

    if emb_result["confidence"] >= embed_threshold:
        _stats["embedding_accepted"] += 1
        return {
            "agent": emb_result["agent"],
            "confidence": emb_result["confidence"],
            "scores": emb_result["scores"],
            "method": "hybrid",
            "stage": "embedding",
            "keyword_result": kw_result,
            "embedding_result": emb_result,
            "llm_result": None,
        }

    # Stage 3: LLM fallback (optional)
    if use_llm_fallback:
        from src.routers import llm_router
        llm_result = llm_router.route(query)
        _stats["llm_fallback"] += 1
        return {
            "agent": llm_result["agent"],
            "confidence": llm_result["confidence"],
            "scores": {},
            "method": "hybrid",
            "stage": "llm",
            "keyword_result": kw_result,
            "embedding_result": emb_result,
            "llm_result": llm_result,
        }

    # 沒有 LLM fallback → 用 R2 的結果（即使 confidence 低）
    _stats["embedding_accepted"] += 1
    return {
        "agent": emb_result["agent"],
        "confidence": emb_result["confidence"],
        "scores": emb_result["scores"],
        "method": "hybrid",
        "stage": "embedding_low_conf",
        "keyword_result": kw_result,
        "embedding_result": emb_result,
        "llm_result": None,
    }


# ─────────────────────────────────────────────
# Threshold tuning（在 validation set 上 grid search）
# ─────────────────────────────────────────────

def tune_thresholds(
    val_data: list[dict],
    intent_names: list[str],
    intent_to_agent: dict[str, str],
    keyword_range: list[float] = None,
    embed_range: list[float] = None,
) -> dict:
    """
    在 validation set 上做 grid search 找最佳 threshold 組合。

    Args:
        val_data: validation set（list of {"text": str, "intent": int}）
        intent_names: intent ID → name
        intent_to_agent: intent name → agent name
        keyword_range: R1 threshold 候選值
        embed_range: R2 threshold 候選值

    Returns:
        {
            "best_keyword_threshold": float,
            "best_embed_threshold": float,
            "best_accuracy": float,
            "keyword_accept_rate": float,
            "grid_results": list[dict],
        }
    """
    if keyword_range is None:
        keyword_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    if embed_range is None:
        embed_range = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # 預計算所有 query 的 R1 和 R2 結果（避免重複計算）
    print("  Pre-computing R1 and R2 results for validation set...")
    precomputed = []
    for item in val_data:
        intent_name = intent_names[item["intent"]]
        true_agent = intent_to_agent[intent_name]
        kw = keyword_router.route(item["text"])
        emb = embedding_router.route(item["text"])
        precomputed.append({
            "true_agent": true_agent,
            "kw_agent": kw["agent"],
            "kw_conf": kw["confidence"],
            "emb_agent": emb["agent"],
            "emb_conf": emb["confidence"],
        })

    # Grid search
    grid_results = []
    best = {"acc": 0, "kt": 0, "et": 0, "kw_rate": 0}

    for kt in keyword_range:
        for et in embed_range:
            correct = 0
            kw_accepted = 0
            for p in precomputed:
                if p["kw_conf"] >= kt:
                    predicted = p["kw_agent"]
                    kw_accepted += 1
                elif p["emb_conf"] >= et:
                    predicted = p["emb_agent"]
                else:
                    predicted = p["emb_agent"]  # 無 LLM fallback

                if predicted == p["true_agent"]:
                    correct += 1

            acc = correct / len(precomputed)
            kw_rate = kw_accepted / len(precomputed)

            grid_results.append({
                "keyword_threshold": kt,
                "embed_threshold": et,
                "accuracy": round(acc, 4),
                "keyword_accept_rate": round(kw_rate, 4),
            })

            if acc > best["acc"]:
                best = {"acc": acc, "kt": kt, "et": et, "kw_rate": kw_rate}

    return {
        "best_keyword_threshold": best["kt"],
        "best_embed_threshold": best["et"],
        "best_accuracy": round(best["acc"], 4),
        "keyword_accept_rate": round(best["kw_rate"], 4),
        "grid_results": grid_results,
    }


# ─────────────────────────────────────────────
# 直接執行：tune + evaluate
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

    with open(_DATA_DIR / "clinc150" / "validation.json") as f:
        val_data = json.load(f)
    with open(_DATA_DIR / "clinc150" / "intent_names.json") as f:
        intent_names = json.load(f)
    with open(_DATA_DIR / "intent_to_agent.json") as f:
        raw = json.load(f)
        mapping = {k: v for k, v in raw.items() if k != "_meta"}

    print("=== Hybrid Router (R4) — Threshold Tuning ===\n")
    result = tune_thresholds(val_data, intent_names, mapping)

    print(f"\nBest thresholds:")
    print(f"  Keyword threshold: {result['best_keyword_threshold']}")
    print(f"  Embed threshold:   {result['best_embed_threshold']}")
    print(f"  Accuracy:          {result['best_accuracy']*100:.1f}%")
    print(f"  Keyword accept %:  {result['keyword_accept_rate']*100:.1f}%")

    # 顯示 grid 中的 top 10
    print(f"\nTop 10 configurations:")
    sorted_grid = sorted(result["grid_results"], key=lambda x: -x["accuracy"])
    for i, g in enumerate(sorted_grid[:10]):
        print(f"  {i+1}. kt={g['keyword_threshold']:.1f} et={g['embed_threshold']:.2f} "
              f"→ acc={g['accuracy']*100:.1f}% kw_rate={g['keyword_accept_rate']*100:.1f}%")
