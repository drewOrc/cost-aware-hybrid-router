# Reviewer 2 — Round 1 Review

**Paper:** "Cost-Aware Hybrid Routing for Intent Classification: 74% LLM Cost Reduction at Equal Accuracy"
**Date:** 2026-04-08
**Score: 4/5 (Weak Accept)** | **Confidence: 5/5 (Expert)**

---

## Summary

This workshop paper proposes a confidence-gated cascade routing architecture (R4) that combines keyword, embedding, and LLM stages to reduce LLM API calls on the CLINC150 intent classification benchmark. The headline result is 82.6% accuracy (±1.2pp) using only 26% of LLM calls versus an LLM-only baseline's 82.9% (±0.6pp), achieving a 74% cost reduction while remaining statistically indistinguishable via McNemar tests (all p > 0.37, n=1,200 pooled across 3 seeds). The paper includes proper significance testing, per-seed trajectories, qualitative error analysis, and code release.

---

## Strengths

- **[S1] Statistically rigorous evaluation.** McNemar paired exact-binomial tests, Wilson 95% CIs, 3-seed evaluation with stratified sampling (50 × 8 = 400 per seed). Above typical workshop standards.
- **[S2] Honest cost accounting.** Total API cost transparently reported ($0.44). Per-seed LLM call rates disaggregated. Acknowledges Haiku 4.5 is already cheap.
- **[S3] Informative qualitative analysis.** Table 2 (cascade-stage examples) and Table 3 (per-agent escalation rates) show real failure modes, not just successes.
- **[S4] Clear problem motivation.** "Which queries need the LLM" is more useful than "LLM vs no LLM."
- **[S5] Reproducibility.** Code under MIT, seeds documented, thresholds explicit.

## Weaknesses

- **[W1] Single benchmark.** CLINC150 only, 8 agents. → Add "on CLINC150" qualifier to abstract. ✅ FIXED
- **[W2] Validation set construction unclear.** → Clarify size, stratification, disjointness from test. ✅ FIXED
- **[W3] OOS trade-off hidden in abstract.** R4 loses ~16pp on OOS vs R3. → Acknowledge in abstract. ✅ FIXED
- **[W4] No α sensitivity analysis.** Fixed 0.1 coefficient without justification. → Add Pareto curve for multiple α values. (FUTURE)
- **[W5] No alternative cheap-stage comparison.** → Consider DistilBERT or TinyBERT baseline. (FUTURE)
- **[W6] Latency mentioned but not measured.** → Add one line with measured latencies. (FUTURE)
- **[W7] CI overlap not explicitly stated.** → Add sentence in §4.2. ✅ FIXED
- **[W8] R2 failure modes not detailed.** → Add R2-stop error rate column to Table 3. (FUTURE)

## Questions for Authors

- [Q1] Was the validation set for threshold tuning held out before selecting test samples? → YES, CLINC150 val split is disjoint.
- [Q2] How was the 0.1 coefficient chosen? → Heuristic; sensitivity not explored.
- [Q3] Intermediate granularities (20, 50 agents)? → Not tested.
- [Q4] Prompt sensitivity? → Not tested.
- [Q5] Cascade vs ensemble design choice? → Not empirically justified.

## Domain-Specific Checklist: 9/9 ✅

---

## Revision Status

| Fix | Status |
|-----|--------|
| Abstract: "on CLINC150" qualifier | ✅ Done |
| Abstract: OOS trade-off mention | ✅ Done |
| §3.3: Validation set details | ✅ Done |
| §4.2: CI overlap sentence | ✅ Done |
| Reproducibility: dual-run note | ✅ Done |
| Pareto sensitivity analysis | 📋 Future (Paper 1 v2) |
| Alternative cheap-stage baselines | 📋 Future |
| Latency measurements | 📋 Future |
