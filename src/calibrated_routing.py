"""
calibrated_routing.py — Platt Scaling Calibrated Routing Ablation

Suggested by Prof. Sahba: replace dataset-specific hard thresholds with
calibrated P(correct) via logistic regression (Platt scaling).

R1 keyword_score → P(correct|R1) = σ(a₁·score + b₁)
R2 cosine_sim    → P(correct|R2) = σ(a₂·score + b₂)
Unified threshold τ: P(correct) ≥ τ → accept router's prediction.

Usage:
    PYTHONPATH=. python3 src/calibrated_routing.py

Output → results/calibrated_routing/
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.routers import keyword_router, embedding_router
from src.stats import mcnemar_test

# ─────────────────────────────────────────────
# Paths & data
# ─────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "calibrated_routing"
OUT_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / "clinc150" / "validation.json") as f:
    VAL_DATA = json.load(f)
with open(DATA_DIR / "clinc150" / "test.json") as f:
    TEST_DATA = json.load(f)
with open(DATA_DIR / "clinc150" / "intent_names.json") as f:
    INTENT_NAMES = json.load(f)
with open(DATA_DIR / "intent_to_agent.json") as f:
    raw = json.load(f)
    MAPPING = {k: v for k, v in raw.items() if k != "_meta"}

P_LLM_CORRECT = 0.856

GRID_NO_LLM = {"kt": 1.5, "et": 0.05}
GRID_WITH_LLM = {"kt": 0.5, "et": 0.1}

SEEDS = [42, 43, 44]
N_PER_AGENT = 50


def get_true_agent(item: dict) -> str:
    return MAPPING[INTENT_NAMES[item["intent"]]]


# ─────────────────────────────────────────────
# Pre-compute R1/R2 scores
# ─────────────────────────────────────────────

def precompute_scores(data: list[dict], label: str) -> list[dict]:
    print(f"  Pre-computing R1/R2 on {label} ({len(data)} queries)...")
    scores = []
    for i, item in enumerate(data):
        true_agent = get_true_agent(item)
        r1 = keyword_router.route(item["text"])
        r2 = embedding_router.route(item["text"])
        scores.append({
            "true_agent": true_agent,
            "r1_score": r1["confidence"],
            "r1_agent": r1["agent"],
            "r1_correct": int(r1["agent"] == true_agent),
            "r2_score": r2["confidence"],
            "r2_agent": r2["agent"],
            "r2_correct": int(r2["agent"] == true_agent),
        })
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(data)}")
    return scores


# ─────────────────────────────────────────────
# Cascade evaluation helpers
# ─────────────────────────────────────────────

def evaluate_calibrated_cascade(scores, p_r1, p_r2, tau):
    kw_count = emb_count = llm_count = 0
    kw_correct = emb_correct = 0
    flags_no_llm = []

    for i, s in enumerate(scores):
        if p_r1[i] >= tau:
            kw_count += 1
            kw_correct += s["r1_correct"]
            flags_no_llm.append(s["r1_correct"])
        elif p_r2[i] >= tau:
            emb_count += 1
            emb_correct += s["r2_correct"]
            flags_no_llm.append(s["r2_correct"])
        else:
            llm_count += 1
            flags_no_llm.append(s["r2_correct"])

    n = len(scores)
    correct_no_llm = sum(flags_no_llm)
    acc_no_llm = correct_no_llm / n
    llm_rate = llm_count / n
    acc_with_llm = (kw_correct + emb_correct + llm_count * P_LLM_CORRECT) / n

    return {
        "accuracy_no_llm": round(acc_no_llm, 4),
        "accuracy_with_llm_expected": round(acc_with_llm, 4),
        "llm_call_rate": round(llm_rate, 4),
        "stages": {"keyword": kw_count, "embedding": emb_count, "llm_fallback": llm_count},
        "flags_no_llm": flags_no_llm,
    }


def evaluate_grid_cascade(scores, kt, et):
    kw_count = emb_count = llm_count = 0
    kw_correct = emb_correct = 0
    flags = []

    for s in scores:
        if s["r1_score"] >= kt:
            kw_count += 1
            kw_correct += s["r1_correct"]
            flags.append(s["r1_correct"])
        elif s["r2_score"] >= et:
            emb_count += 1
            emb_correct += s["r2_correct"]
            flags.append(s["r2_correct"])
        else:
            llm_count += 1
            flags.append(s["r2_correct"])

    n = len(scores)
    acc_no_llm = sum(flags) / n
    acc_with_llm = (kw_correct + emb_correct + llm_count * P_LLM_CORRECT) / n

    return {
        "accuracy_no_llm": round(acc_no_llm, 4),
        "accuracy_with_llm_expected": round(acc_with_llm, 4),
        "llm_call_rate": round(llm_count / n, 4),
        "stages": {"keyword": kw_count, "embedding": emb_count, "llm_fallback": llm_count},
        "flags": flags,
    }


# ─────────────────────────────────────────────
# Stratified sampling (same as evaluate.py)
# ─────────────────────────────────────────────

import random
from collections import defaultdict

def stratified_sample_indices(scores, n_per_agent, seed):
    rng = random.Random(seed)
    by_agent = defaultdict(list)
    for i, s in enumerate(scores):
        by_agent[s["true_agent"]].append(i)
    indices = []
    for agent in sorted(by_agent):
        items = by_agent[agent]
        if len(items) <= n_per_agent:
            indices.extend(items)
        else:
            indices.extend(rng.sample(items, n_per_agent))
    return indices


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()

    print("=" * 70)
    print("  Platt Scaling Calibrated Routing — Ablation Experiment")
    print("=" * 70)

    # ── Step 1: Pre-compute scores ──
    print("\n[Step 1] Pre-compute R1/R2 scores")
    val_scores = precompute_scores(VAL_DATA, "val")
    test_scores = precompute_scores(TEST_DATA, "test")

    # ── Step 2: Fit Platt scaling ──
    print("\n[Step 2] Fit Platt scaling on val set")

    X_r1 = np.array([s["r1_score"] for s in val_scores]).reshape(-1, 1)
    y_r1 = np.array([s["r1_correct"] for s in val_scores])
    X_r2 = np.array([s["r2_score"] for s in val_scores]).reshape(-1, 1)
    y_r2 = np.array([s["r2_correct"] for s in val_scores])

    platt_r1 = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt_r1.fit(X_r1, y_r1)
    platt_r2 = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt_r2.fit(X_r2, y_r2)

    a1, b1 = float(platt_r1.coef_[0][0]), float(platt_r1.intercept_[0])
    a2, b2 = float(platt_r2.coef_[0][0]), float(platt_r2.intercept_[0])

    print(f"  R1: P(correct) = σ({a1:.4f} × kw_score + ({b1:.4f}))")
    print(f"  R2: P(correct) = σ({a2:.4f} × cos_sim + ({b2:.4f}))")

    # Calibrated probabilities on val set (for diagnostics)
    p_r1_val = platt_r1.predict_proba(X_r1)[:, 1]
    p_r2_val = platt_r2.predict_proba(X_r2)[:, 1]
    print(f"  Val R1 P(correct) range: [{p_r1_val.min():.4f}, {p_r1_val.max():.4f}]")
    print(f"  Val R2 P(correct) range: [{p_r2_val.min():.4f}, {p_r2_val.max():.4f}]")

    platt_params = {
        "r1": {"a": a1, "b": b1},
        "r2": {"a": a2, "b": b2},
        "val_r1_prob_range": [round(float(p_r1_val.min()), 4), round(float(p_r1_val.max()), 4)],
        "val_r2_prob_range": [round(float(p_r2_val.min()), 4), round(float(p_r2_val.max()), 4)],
        "val_set_size": len(VAL_DATA),
        "p_llm_correct": P_LLM_CORRECT,
    }

    # ── Step 3: Calibrate test set scores ──
    print("\n[Step 3] Calibrate test set scores")
    X_r1_test = np.array([s["r1_score"] for s in test_scores]).reshape(-1, 1)
    X_r2_test = np.array([s["r2_score"] for s in test_scores]).reshape(-1, 1)
    p_r1_test = platt_r1.predict_proba(X_r1_test)[:, 1]
    p_r2_test = platt_r2.predict_proba(X_r2_test)[:, 1]
    print(f"  Test R1 P(correct) range: [{p_r1_test.min():.4f}, {p_r1_test.max():.4f}]")
    print(f"  Test R2 P(correct) range: [{p_r2_test.min():.4f}, {p_r2_test.max():.4f}]")

    # ── Step 4: τ sweep on full test set ──
    print("\n[Step 4] τ sweep (0.50 → 0.95) on full test set")
    tau_values = list(np.arange(0.50, 0.96, 0.01))
    tau_results = []

    for tau in tau_values:
        r = evaluate_calibrated_cascade(test_scores, p_r1_test, p_r2_test, tau)
        r["tau"] = round(float(tau), 2)
        tau_results.append(r)

        if abs(float(tau) * 100 % 5) < 0.5:
            print(f"  τ={tau:.2f}: acc_no_llm={r['accuracy_no_llm']:.4f}  "
                  f"acc_with_llm≈{r['accuracy_with_llm_expected']:.4f}  "
                  f"LLM_rate={r['llm_call_rate']:.4f}  "
                  f"(kw={r['stages']['keyword']}, emb={r['stages']['embedding']}, "
                  f"llm={r['stages']['llm_fallback']})")

    # ── Step 5: Grid-search baseline on test set ──
    print("\n[Step 5] Grid-search baseline on test set")

    grid_no_llm_result = evaluate_grid_cascade(test_scores, **GRID_NO_LLM)
    grid_with_llm_result = evaluate_grid_cascade(test_scores, **GRID_WITH_LLM)

    print(f"  Grid no-LLM  (kt={GRID_NO_LLM['kt']}, et={GRID_NO_LLM['et']}): "
          f"acc={grid_no_llm_result['accuracy_no_llm']:.4f}  "
          f"LLM_rate={grid_no_llm_result['llm_call_rate']:.4f}")
    print(f"  Grid with-LLM (kt={GRID_WITH_LLM['kt']}, et={GRID_WITH_LLM['et']}): "
          f"acc_no_llm={grid_with_llm_result['accuracy_no_llm']:.4f}  "
          f"acc_with_llm≈{grid_with_llm_result['accuracy_with_llm_expected']:.4f}  "
          f"LLM_rate={grid_with_llm_result['llm_call_rate']:.4f}")

    # Compute full grid on test set for tradeoff curve
    kt_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    et_range = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    grid_points = []
    for kt in kt_range:
        for et in et_range:
            g = evaluate_grid_cascade(test_scores, kt, et)
            grid_points.append({
                "kt": kt, "et": et,
                "accuracy_no_llm": g["accuracy_no_llm"],
                "accuracy_with_llm_expected": g["accuracy_with_llm_expected"],
                "llm_call_rate": g["llm_call_rate"],
            })

    # ── Step 6: 3-seed evaluation on stratified samples ──
    print("\n[Step 6] 3-seed evaluation on stratified samples")
    seed_results = []

    for seed in SEEDS:
        indices = stratified_sample_indices(test_scores, N_PER_AGENT, seed)
        sample_scores = [test_scores[i] for i in indices]
        sample_p_r1 = p_r1_test[indices]
        sample_p_r2 = p_r2_test[indices]

        # Find τ that matches grid with-LLM operating point's LLM rate
        best_tau_for_seed = None
        best_diff = float("inf")
        for tr in tau_results:
            diff = abs(tr["llm_call_rate"] - grid_with_llm_result["llm_call_rate"])
            if diff < best_diff:
                best_diff = diff
                best_tau_for_seed = tr["tau"]

        cal = evaluate_calibrated_cascade(
            sample_scores, sample_p_r1, sample_p_r2, best_tau_for_seed)
        grid = evaluate_grid_cascade(sample_scores, **GRID_WITH_LLM)

        mc = mcnemar_test(grid["flags"], cal["flags_no_llm"])

        seed_results.append({
            "seed": seed,
            "n": len(sample_scores),
            "tau": best_tau_for_seed,
            "calibrated_acc_no_llm": cal["accuracy_no_llm"],
            "calibrated_llm_rate": cal["llm_call_rate"],
            "grid_acc_no_llm": grid["accuracy_no_llm"],
            "grid_llm_rate": grid["llm_call_rate"],
            "mcnemar_p": mc["p_value"],
            "mcnemar_delta": mc["delta"],
            "mcnemar_significant": mc["significant_at_0.05"],
        })
        print(f"  Seed {seed} (n={len(sample_scores)}): "
              f"cal={cal['accuracy_no_llm']:.4f} vs grid={grid['accuracy_no_llm']:.4f}  "
              f"McNemar p={mc['p_value']:.4f}  Δ={mc['delta']*100:+.1f}pp")

    # ── Step 7: McNemar on full test set ──
    print("\n[Step 7] McNemar on full test set (no-LLM)")

    # Match τ to grid no-LLM operating point
    target_rate = grid_no_llm_result["llm_call_rate"]
    match_no_llm = min(tau_results, key=lambda r: abs(r["llm_call_rate"] - target_rate))
    mc_no_llm = mcnemar_test(
        grid_no_llm_result["flags"], match_no_llm["flags_no_llm"])
    print(f"  Calibrated τ={match_no_llm['tau']} vs Grid (kt={GRID_NO_LLM['kt']}, et={GRID_NO_LLM['et']})")
    print(f"  Cal acc={match_no_llm['accuracy_no_llm']:.4f} vs Grid acc={grid_no_llm_result['accuracy_no_llm']:.4f}")
    print(f"  McNemar p={mc_no_llm['p_value']:.4f}  Δ={mc_no_llm['delta']*100:+.1f}pp  "
          f"sig={mc_no_llm['significant_at_0.05']}")

    # Match τ to grid with-LLM operating point
    target_rate_llm = grid_with_llm_result["llm_call_rate"]
    match_with_llm = min(tau_results, key=lambda r: abs(r["llm_call_rate"] - target_rate_llm))
    mc_with_llm = mcnemar_test(
        grid_with_llm_result["flags"], match_with_llm["flags_no_llm"])
    print(f"\n  Calibrated τ={match_with_llm['tau']} vs Grid (kt={GRID_WITH_LLM['kt']}, et={GRID_WITH_LLM['et']})")
    print(f"  Cal acc={match_with_llm['accuracy_no_llm']:.4f} vs Grid acc={grid_with_llm_result['accuracy_no_llm']:.4f}")
    print(f"  Cal LLM_rate={match_with_llm['llm_call_rate']:.4f} vs Grid LLM_rate={grid_with_llm_result['llm_call_rate']:.4f}")
    print(f"  McNemar p={mc_with_llm['p_value']:.4f}  Δ={mc_with_llm['delta']*100:+.1f}pp  "
          f"sig={mc_with_llm['significant_at_0.05']}")

    # ── Step 8: Tradeoff curve ──
    print("\n[Step 8] Generating tradeoff curve")

    cal_rates = [r["llm_call_rate"] for r in tau_results]
    cal_acc_wllm = [r["accuracy_with_llm_expected"] for r in tau_results]
    cal_acc_nollm = [r["accuracy_no_llm"] for r in tau_results]

    grid_rates = [g["llm_call_rate"] for g in grid_points]
    grid_acc_wllm = [g["accuracy_with_llm_expected"] for g in grid_points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: with-LLM expected accuracy vs LLM call rate
    ax1.plot(cal_rates, cal_acc_wllm, "b-", linewidth=2, label="Calibrated (τ sweep)")
    for r in tau_results:
        if r["tau"] in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
            ax1.annotate(f'τ={r["tau"]}',
                         (r["llm_call_rate"], r["accuracy_with_llm_expected"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7,
                         color="blue")
            ax1.plot(r["llm_call_rate"], r["accuracy_with_llm_expected"],
                     "bo", markersize=5)

    ax1.scatter(grid_rates, grid_acc_wllm, c="red", s=15, alpha=0.4, zorder=3,
                label="Grid search (kt,et)")
    ax1.scatter([grid_with_llm_result["llm_call_rate"]],
                [grid_with_llm_result["accuracy_with_llm_expected"]],
                c="red", s=120, marker="*", zorder=5,
                label=f'Grid best (kt={GRID_WITH_LLM["kt"]}, et={GRID_WITH_LLM["et"]})')

    ax1.set_xlabel("LLM Call Rate (lower = cheaper)")
    ax1.set_ylabel("Expected Accuracy (with LLM fallback)")
    ax1.set_title(f"Calibrated vs Grid: Accuracy–Cost Tradeoff\n(p_LLM={P_LLM_CORRECT})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: no-LLM accuracy vs LLM call rate
    ax2.plot(cal_rates, cal_acc_nollm, "b-", linewidth=2, label="Calibrated (τ sweep)")
    for r in tau_results:
        if r["tau"] in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
            ax2.annotate(f'τ={r["tau"]}',
                         (r["llm_call_rate"], r["accuracy_no_llm"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7,
                         color="blue")
            ax2.plot(r["llm_call_rate"], r["accuracy_no_llm"], "bo", markersize=5)

    ax2.scatter([grid_no_llm_result["llm_call_rate"]],
                [grid_no_llm_result["accuracy_no_llm"]],
                c="red", s=120, marker="*", zorder=5,
                label=f'Grid no-LLM (kt={GRID_NO_LLM["kt"]}, et={GRID_NO_LLM["et"]})')
    ax2.scatter([grid_with_llm_result["llm_call_rate"]],
                [grid_with_llm_result["accuracy_no_llm"]],
                c="darkred", s=120, marker="D", zorder=5,
                label=f'Grid with-LLM thresholds (kt={GRID_WITH_LLM["kt"]}, et={GRID_WITH_LLM["et"]})')

    ax2.set_xlabel("LLM Call Rate (lower = cheaper)")
    ax2.set_ylabel("Accuracy (no LLM fallback)")
    ax2.set_title("No-LLM Variant: Accuracy vs Routing Aggressiveness")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / "tau_tradeoff_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── Step 9: Save results ──
    print("\n[Step 9] Saving results")

    with open(OUT_DIR / "platt_params.json", "w") as f:
        json.dump(platt_params, f, indent=2)

    tau_sweep_clean = [
        {k: v for k, v in r.items() if k != "flags_no_llm"}
        for r in tau_results
    ]
    with open(OUT_DIR / "tau_sweep.json", "w") as f:
        json.dump(tau_sweep_clean, f, indent=2)

    comparison = {
        "grid_baseline": {
            "no_llm": {
                "thresholds": GRID_NO_LLM,
                "accuracy_no_llm": grid_no_llm_result["accuracy_no_llm"],
                "accuracy_with_llm_expected": grid_no_llm_result["accuracy_with_llm_expected"],
                "llm_call_rate": grid_no_llm_result["llm_call_rate"],
                "stages": grid_no_llm_result["stages"],
            },
            "with_llm": {
                "thresholds": GRID_WITH_LLM,
                "accuracy_no_llm": grid_with_llm_result["accuracy_no_llm"],
                "accuracy_with_llm_expected": grid_with_llm_result["accuracy_with_llm_expected"],
                "llm_call_rate": grid_with_llm_result["llm_call_rate"],
                "stages": grid_with_llm_result["stages"],
            },
        },
        "calibrated_match": {
            "no_llm": {
                "tau": match_no_llm["tau"],
                "accuracy_no_llm": match_no_llm["accuracy_no_llm"],
                "accuracy_with_llm_expected": match_no_llm["accuracy_with_llm_expected"],
                "llm_call_rate": match_no_llm["llm_call_rate"],
                "stages": match_no_llm["stages"],
            },
            "with_llm": {
                "tau": match_with_llm["tau"],
                "accuracy_no_llm": match_with_llm["accuracy_no_llm"],
                "accuracy_with_llm_expected": match_with_llm["accuracy_with_llm_expected"],
                "llm_call_rate": match_with_llm["llm_call_rate"],
                "stages": match_with_llm["stages"],
            },
        },
        "mcnemar": {
            "no_llm_full_test": {
                "grid": GRID_NO_LLM,
                "calibrated_tau": match_no_llm["tau"],
                "p_value": mc_no_llm["p_value"],
                "delta": mc_no_llm["delta"],
                "significant": mc_no_llm["significant_at_0.05"],
                "b": mc_no_llm["b"],
                "c": mc_no_llm["c"],
                "n_disagree": mc_no_llm["n_disagree"],
            },
            "with_llm_full_test": {
                "grid": GRID_WITH_LLM,
                "calibrated_tau": match_with_llm["tau"],
                "p_value": mc_with_llm["p_value"],
                "delta": mc_with_llm["delta"],
                "significant": mc_with_llm["significant_at_0.05"],
                "b": mc_with_llm["b"],
                "c": mc_with_llm["c"],
                "n_disagree": mc_with_llm["n_disagree"],
            },
        },
        "seed_results": seed_results,
        "platt_params": platt_params,
        "p_llm_correct": P_LLM_CORRECT,
        "test_set_size": len(TEST_DATA),
    }

    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Grid points for reference
    with open(OUT_DIR / "grid_test_sweep.json", "w") as f:
        json.dump(grid_points, f, indent=2)

    wall = round(time.time() - t_start, 1)

    print("\n" + "=" * 70)
    print("  Results Summary")
    print("=" * 70)
    print(f"\n  Platt parameters:")
    print(f"    R1: σ({a1:.4f} × kw_score + ({b1:.4f}))")
    print(f"    R2: σ({a2:.4f} × cos_sim + ({b2:.4f}))")
    print(f"\n  Full test set (n={len(TEST_DATA)}):")
    print(f"    Grid no-LLM  (kt={GRID_NO_LLM['kt']}, et={GRID_NO_LLM['et']}):  "
          f"acc={grid_no_llm_result['accuracy_no_llm']:.4f}  "
          f"LLM_rate={grid_no_llm_result['llm_call_rate']:.4f}")
    print(f"    Cal  no-LLM  (τ={match_no_llm['tau']}):  "
          f"acc={match_no_llm['accuracy_no_llm']:.4f}  "
          f"LLM_rate={match_no_llm['llm_call_rate']:.4f}  "
          f"McNemar p={mc_no_llm['p_value']:.4f}")
    print(f"    Grid with-LLM (kt={GRID_WITH_LLM['kt']}, et={GRID_WITH_LLM['et']}): "
          f"acc≈{grid_with_llm_result['accuracy_with_llm_expected']:.4f}  "
          f"LLM_rate={grid_with_llm_result['llm_call_rate']:.4f}")
    print(f"    Cal  with-LLM (τ={match_with_llm['tau']}): "
          f"acc≈{match_with_llm['accuracy_with_llm_expected']:.4f}  "
          f"LLM_rate={match_with_llm['llm_call_rate']:.4f}  "
          f"McNemar p={mc_with_llm['p_value']:.4f}")
    print(f"\n  3-seed stratified samples (n≈{N_PER_AGENT*8}):")
    for sr in seed_results:
        print(f"    Seed {sr['seed']}: cal={sr['calibrated_acc_no_llm']:.4f} vs "
              f"grid={sr['grid_acc_no_llm']:.4f}  "
              f"McNemar p={sr['mcnemar_p']:.4f}  Δ={sr['mcnemar_delta']*100:+.1f}pp")
    print(f"\n  Wall time: {wall}s  |  Cost: $0")
    print(f"  Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
