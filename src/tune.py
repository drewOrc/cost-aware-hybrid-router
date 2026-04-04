"""
tune.py — 在 validation set 上 grid search 選 threshold

跑法：
    cd cost-aware-hybrid-router/
    PYTHONPATH=. python3 src/tune.py

輸出：
    results/tuned_thresholds.json
      {
        "no_llm":   {"keyword_threshold": ..., "embed_threshold": ...},
        "with_llm": {"keyword_threshold": ..., "embed_threshold": ...},
        "grid_no_llm":   [ ... ],
        "grid_with_llm": [ ... ]
      }

說明：
- "no_llm" 優化目標：validation accuracy（hybrid cascade, 無 LLM fallback）
- "with_llm" 優化目標：在 expected accuracy ≥ 0.88 的 configs 中，挑
  LLM-call-rate 最低者；若無達標則回退到最大 acc
- 選到的 thresholds 由 evaluate.py 讀取並用於 test set 評估，
  確保 "threshold 不是從 test set 挑的"（避免 test-set peeking）
"""

import json
from pathlib import Path

from src.routers import keyword_router, embedding_router

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

KEYWORD_RANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
EMBED_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LLM_ACC_TARGET = 0.88  # 用於 with_llm 組：滿足此 expected acc 下最低 LLM 呼叫率


def precompute_on_val() -> list[dict]:
    with open(DATA_DIR / "clinc150" / "validation.json") as f:
        val_data = json.load(f)
    with open(DATA_DIR / "clinc150" / "intent_names.json") as f:
        intent_names = json.load(f)
    with open(DATA_DIR / "intent_to_agent.json") as f:
        raw = json.load(f)
        mapping = {k: v for k, v in raw.items() if k != "_meta"}

    print(f"Pre-computing R1 + R2 on val set (n={len(val_data)})...")
    rows = []
    for i, item in enumerate(val_data):
        true_agent = mapping[intent_names[item["intent"]]]
        kw = keyword_router.route(item["text"])
        emb = embedding_router.route(item["text"])
        rows.append({
            "true": true_agent,
            "kw_agent": kw["agent"], "kw_conf": kw["confidence"],
            "emb_agent": emb["agent"], "emb_conf": emb["confidence"],
        })
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(val_data)}")
    return rows


def grid_no_llm(rows: list[dict]) -> list[dict]:
    """Hybrid without LLM fallback: R1 → R2 (use R2 even at low conf)."""
    results = []
    for kt in KEYWORD_RANGE:
        for et in EMBED_RANGE:
            correct = 0
            kw_acc = 0
            for r in rows:
                if r["kw_conf"] >= kt:
                    pred = r["kw_agent"]
                    kw_acc += 1
                else:
                    pred = r["emb_agent"]
                if pred == r["true"]:
                    correct += 1
            n = len(rows)
            results.append({
                "keyword_threshold": kt,
                "embed_threshold": et,
                "accuracy": round(correct / n, 4),
                "keyword_accept_rate": round(kw_acc / n, 4),
            })
    return results


def grid_with_llm(rows: list[dict]) -> list[dict]:
    """
    Hybrid with LLM fallback simulated as 'oracle' (assume LLM gets it right).
    This is standard practice: LLM fallback's expected accuracy is estimated
    separately (from R3 sample). Here we compute:
      - expected_acc = P(R1 accept & R1 correct) + P(R2 accept & R2 correct)
                      + P(LLM fallback) * p_llm_correct  (p_llm_correct = 0.856 from R3)
      - llm_call_rate = fraction routed to LLM
    """
    P_LLM_CORRECT = 0.856  # from R3 on sample; update if you re-run R3
    results = []
    for kt in KEYWORD_RANGE:
        for et in EMBED_RANGE:
            correct_known = 0   # R1 or R2 stage, their own correctness
            llm_calls = 0
            for r in rows:
                if r["kw_conf"] >= kt:
                    if r["kw_agent"] == r["true"]:
                        correct_known += 1
                elif r["emb_conf"] >= et:
                    if r["emb_agent"] == r["true"]:
                        correct_known += 1
                else:
                    llm_calls += 1
            n = len(rows)
            expected_correct = correct_known + llm_calls * P_LLM_CORRECT
            results.append({
                "keyword_threshold": kt,
                "embed_threshold": et,
                "expected_accuracy": round(expected_correct / n, 4),
                "llm_call_rate": round(llm_calls / n, 4),
            })
    return results


def pick_best_no_llm(grid: list[dict]) -> dict:
    best = max(grid, key=lambda g: g["accuracy"])
    return {
        "keyword_threshold": best["keyword_threshold"],
        "embed_threshold": best["embed_threshold"],
        "val_accuracy": best["accuracy"],
        "val_keyword_accept_rate": best["keyword_accept_rate"],
    }


def pick_best_with_llm(grid: list[dict]) -> dict:
    qualified = [g for g in grid if g["expected_accuracy"] >= LLM_ACC_TARGET]
    if qualified:
        best = min(qualified, key=lambda g: g["llm_call_rate"])
        criterion = f"min LLM-call-rate s.t. expected_acc ≥ {LLM_ACC_TARGET}"
    else:
        best = max(grid, key=lambda g: g["expected_accuracy"])
        criterion = f"max expected_acc (no config reached {LLM_ACC_TARGET})"
    return {
        "keyword_threshold": best["keyword_threshold"],
        "embed_threshold": best["embed_threshold"],
        "val_expected_accuracy": best["expected_accuracy"],
        "val_llm_call_rate": best["llm_call_rate"],
        "selection_criterion": criterion,
    }


def main():
    rows = precompute_on_val()

    print("\nGrid search: Hybrid (no LLM fallback)")
    g1 = grid_no_llm(rows)
    best1 = pick_best_no_llm(g1)
    print(f"  → kt={best1['keyword_threshold']}, et={best1['embed_threshold']}, "
          f"acc={best1['val_accuracy']*100:.1f}%")

    print("\nGrid search: Hybrid + LLM fallback")
    g2 = grid_with_llm(rows)
    best2 = pick_best_with_llm(g2)
    print(f"  → kt={best2['keyword_threshold']}, et={best2['embed_threshold']}, "
          f"expected_acc={best2['val_expected_accuracy']*100:.1f}%, "
          f"llm_rate={best2['val_llm_call_rate']*100:.1f}%")

    out = {
        "no_llm": best1,
        "with_llm": best2,
        "grid_no_llm": g1,
        "grid_with_llm": g2,
        "note": (
            "Thresholds selected on validation.json (3,100 queries), "
            "NOT on test set. p_llm_correct=0.856 estimated from R3 sample."
        ),
    }
    path = RESULTS_DIR / "tuned_thresholds.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved to {path}")


if __name__ == "__main__":
    main()
