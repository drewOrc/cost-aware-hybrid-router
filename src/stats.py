"""
stats.py — 統計檢定與信賴區間工具

用於 Hybrid Router 實驗的統計嚴謹化：
- McNemar's test：比較兩個分類器在相同樣本上的差異是否顯著
- Wilson score interval：二項比例的信賴區間（比 normal approximation 精確）
- Bootstrap CI：透過重抽樣計算任意 metric 的信賴區間

所有函數皆支援 seed 以確保可重現。
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from scipy import stats


# ─────────────────────────────────────────────
# McNemar's Test
# ─────────────────────────────────────────────

def mcnemar_test(
    preds_a: Sequence[bool],
    preds_b: Sequence[bool],
    exact: bool = True,
) -> dict:
    """
    McNemar's test：兩個分類器在相同樣本上的配對比較。

    Args:
        preds_a: classifier A 的正確與否 (list of bool/0/1, True=正確)
        preds_b: classifier B 的正確與否 (list of bool/0/1)
        exact:   是否用 exact binomial（樣本小時建議 True）

    Returns:
        {
            "b": int,              # A 對 B 錯
            "c": int,              # A 錯 B 對
            "n_disagree": int,     # b + c
            "statistic": float,    # chi-squared statistic (continuity-corrected)
            "p_value": float,      # two-sided p-value
            "test_type": "exact" | "chi2",
            "significant_at_0.05": bool,
            "accuracy_a": float,
            "accuracy_b": float,
            "delta": float,        # acc_b - acc_a
        }
    """
    if len(preds_a) != len(preds_b):
        raise ValueError(f"preds length mismatch: {len(preds_a)} vs {len(preds_b)}")

    n = len(preds_a)
    preds_a = [bool(x) for x in preds_a]
    preds_b = [bool(x) for x in preds_b]

    # Contingency table
    b = sum(1 for a, bb in zip(preds_a, preds_b) if a and not bb)
    c = sum(1 for a, bb in zip(preds_a, preds_b) if not a and bb)
    n_disagree = b + c

    if n_disagree == 0:
        return {
            "b": 0, "c": 0, "n_disagree": 0,
            "statistic": 0.0, "p_value": 1.0,
            "test_type": "none",
            "significant_at_0.05": False,
            "accuracy_a": sum(preds_a) / n,
            "accuracy_b": sum(preds_b) / n,
            "delta": 0.0,
        }

    if exact or n_disagree < 25:
        # Exact binomial test: P(X <= min(b,c) | n_disagree, p=0.5) * 2
        k = min(b, c)
        p_value = 2 * stats.binom.cdf(k, n_disagree, 0.5)
        p_value = min(p_value, 1.0)
        test_type = "exact"
        statistic = float((b - c) ** 2 / n_disagree) if n_disagree > 0 else 0.0
    else:
        # Chi-squared with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / n_disagree
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        test_type = "chi2"

    return {
        "b": b,
        "c": c,
        "n_disagree": n_disagree,
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 6),
        "test_type": test_type,
        "significant_at_0.05": bool(p_value < 0.05),
        "accuracy_a": round(sum(preds_a) / n, 4),
        "accuracy_b": round(sum(preds_b) / n, 4),
        "delta": round(sum(preds_b) / n - sum(preds_a) / n, 4),
    }


# ─────────────────────────────────────────────
# Wilson score interval
# ─────────────────────────────────────────────

def wilson_ci(correct: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wilson score interval for binomial proportion.

    比 normal approximation 更精確，特別是在 p 接近 0 或 1、或樣本小的時候。

    Args:
        correct: 正確的個數
        total:   總樣本數
        alpha:   significance level（default 0.05 → 95% CI）

    Returns:
        (lower, upper) 比例形式
    """
    if total == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - alpha / 2)
    p = correct / total
    n = total

    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


# ─────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────

def bootstrap_accuracy_ci(
    correct_flags: Sequence[bool],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    透過 bootstrap resampling 估計 accuracy 的信賴區間。

    Args:
        correct_flags: list of bool/0/1
        n_resamples:   重抽樣次數
        alpha:         significance level
        seed:          random seed

    Returns:
        {
            "mean": float,
            "lower": float,   # alpha/2 percentile
            "upper": float,   # 1-alpha/2 percentile
            "std": float,
        }
    """
    rng = random.Random(seed)
    flags = [int(bool(x)) for x in correct_flags]
    n = len(flags)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}

    accs = []
    for _ in range(n_resamples):
        resample = [flags[rng.randrange(n)] for _ in range(n)]
        accs.append(sum(resample) / n)

    accs.sort()
    lo_idx = int(n_resamples * alpha / 2)
    hi_idx = int(n_resamples * (1 - alpha / 2))

    mean = sum(accs) / n_resamples
    variance = sum((a - mean) ** 2 for a in accs) / n_resamples
    std = math.sqrt(variance)

    return {
        "mean": round(mean, 4),
        "lower": round(accs[lo_idx], 4),
        "upper": round(accs[hi_idx], 4),
        "std": round(std, 4),
    }


# ─────────────────────────────────────────────
# Multi-seed aggregation
# ─────────────────────────────────────────────

def aggregate_seeds(values: Sequence[float]) -> dict:
    """
    跨多個 seed 彙整結果，回傳 mean ± std。

    Args:
        values: 各 seed 的 metric 值（例如 accuracy）

    Returns:
        {"mean": float, "std": float, "min": float, "max": float, "n_seeds": int}
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_seeds": 0}

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    std = math.sqrt(variance)

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n_seeds": n,
    }


# ─────────────────────────────────────────────
# Quick sanity tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== stats.py sanity tests ===\n")

    # McNemar: 兩個分類器有顯著差異
    a = [1, 1, 0, 0, 1, 1, 0, 1, 1, 0] * 20  # acc 0.6
    b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 20  # acc 1.0
    r = mcnemar_test(a, b)
    print(f"McNemar (大差異): p={r['p_value']:.4f}, significant={r['significant_at_0.05']}")
    print(f"  acc_a={r['accuracy_a']:.2%}, acc_b={r['accuracy_b']:.2%}, delta={r['delta']:.2%}")
    print(f"  b={r['b']}, c={r['c']}, test={r['test_type']}\n")

    # McNemar: 無顯著差異
    a2 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 20
    b2 = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0] * 20
    r2 = mcnemar_test(a2, b2)
    print(f"McNemar (小差異): p={r2['p_value']:.4f}, significant={r2['significant_at_0.05']}")
    print(f"  acc_a={r2['accuracy_a']:.2%}, acc_b={r2['accuracy_b']:.2%}, delta={r2['delta']:.2%}\n")

    # Wilson CI
    lo, hi = wilson_ci(137, 160)  # R3 actual result
    print(f"Wilson CI R3 (137/160): {lo:.4f} - {hi:.4f}  (acc={137/160:.4f})")

    lo2, hi2 = wilson_ci(141, 160)  # R4+LLM actual result
    print(f"Wilson CI R4+LLM (141/160): {lo2:.4f} - {hi2:.4f}  (acc={141/160:.4f})\n")

    # Bootstrap
    flags = [1] * 141 + [0] * 19  # R4+LLM: 141/160 correct
    bc = bootstrap_accuracy_ci(flags, n_resamples=1000, seed=42)
    print(f"Bootstrap R4+LLM: mean={bc['mean']:.4f}, CI=({bc['lower']:.4f}, {bc['upper']:.4f}), std={bc['std']:.4f}\n")

    # Seed aggregation
    seed_accs = [0.881, 0.875, 0.894]
    agg = aggregate_seeds(seed_accs)
    print(f"Seed aggregation: {agg}")
