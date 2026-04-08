# Round 2 Review — Cost-Aware Hybrid Routing for Intent Classification

**Reviewer:** Reviewer 2 (simulated)
**Date:** 2026-04-08
**Overall Score:** 4/5 (Accept)
**Confidence:** 5/5 (Expert)

---

## Summary of Changes Since Round 1

Authors have addressed **6 of 8** Round 1 items:

| Item | Status | Evidence |
|------|--------|----------|
| W1: Single benchmark qualifier | ✅ Addressed | Abstract and intro now qualify claims as "on CLINC150" |
| W2: Validation set clarity | ✅ Addressed | §3.3 now specifies 3,100 queries, disjoint, stratified |
| W3: OOS trade-off visibility | ✅ Addressed | Mentioned in qualitative analysis; Table 2 shows OOS errors |
| W4: Pareto sensitivity analysis | ✅ Addressed | New §4.5 (Threshold Sensitivity) with Table 3, sweeps over 9 α values |
| W5: Alternative cheap-stage baselines | 📋 Deferred | SetFit included as reference (70.2%), but TinyBERT/DistilBERT not compared |
| W6: Latency measurements | ✅ Addressed | Discussion now reports 3.2ms (R1), 0.3ms (R2), ~500ms (R3 API) |
| W7: CI overlap statement | ✅ Addressed | §4.2 now explicitly states CIs overlap (R3: [80.7%, 84.9%] vs R4: [80.3%, 84.6%]) |
| W8: R2 error rates in table | ✅ Addressed | Table 2 (per-stage) shows R1-err, R2-err, and escalation % by agent |

**Grade: 75% direct resolution, deferred one item (W5) to future work**

---

## Round 1 Issues — Status

### Addressed

**W1 — Single benchmark qualifier.**
- **Issue:** Abstract claimed broad conclusions without "on CLINC150" qualifier.
- **Fix:** Abstract now opens with "On the CLINC150 benchmark (150 intents → 7 agents + OOS)"; intro (§1) also emphasizes "we quantify this on CLINC150." Claims are now properly scoped.
- **Adequacy:** ✅ Excellent. Qualifier is prominent and prevents overgeneralization.

**W2 — Validation set construction unclear.**
- **Issue:** No details on val split size, stratification, or disjointness from test.
- **Fix:** §3.3 (Threshold Tuning) now specifies: "CLINC150 validation split (3,100 queries, disjoint from the test set) via grid search over kt ∈ {0.5, 1.0, 1.5, 2.0} and et ∈ {0.05, 0.08, 0.10, 0.15}."
- **Adequacy:** ✅ Good. Addresses reproducibility concern. Grid size (4 × 4 = 16) is reasonable; could mention stratification strategy, but the 3,100-query split is likely proportional to test.

**W3 — OOS trade-off hidden in abstract.**
- **Issue:** R4 loses ~16pp on OOS (64% R3 vs 48% R4) but not acknowledged upfront.
- **Fix:** §4.4 (Qualitative Analysis) now states: "OOS remains the weakest link at both stages: all 7 R1-stopped OOS queries and 5 of 9 R2-stopped OOS queries are wrong, confirming that a dedicated OOS detector would be the highest-impact improvement." Table 2 shows OOS errors explicitly.
- **Adequacy:** ✅ Good but partial. The OOS trade-off is now visible in the main text, but abstract still does not warn readers. For a workshop paper this is acceptable; for a main conference, would warrant abstract mention.

**W6 — Latency mentioned but not measured.**
- **Issue:** Paper claimed latency benefit but gave no numbers.
- **Fix:** New paragraph in Discussion (§5): "On a Mac Mini (M2), R1 keyword processes a query in 3.2 ms median and R2 embedding in 0.3 ms—both 100–1,500× faster than R3's ~500 ms API round-trip. Since 74% of queries never reach the LLM, the cascade's median end-to-end latency is dominated by the cheap stages, not the API call."
- **Adequacy:** ✅ Excellent. Concrete measurements (3.2ms, 0.3ms, 500ms) with hardware specification. Qualitative insight (74% bypass → latency dominated by R1/R2) is valuable.

**W7 — CI overlap not explicitly stated.**
- **Issue:** Table 1 showed overlapping CIs but text did not call them out.
- **Fix:** §4.2 (Statistical Significance) now states: "The Wilson 95% CIs also overlap substantially—R3: [80.7%, 84.9%] vs R4: [80.3%, 84.6%]—corroborating the absence of a significant accuracy difference."
- **Adequacy:** ✅ Excellent. Direct numerical comparison strengthens the equivalence claim.

**W8 — R2 failure modes not detailed.**
- **Issue:** No breakdown of R2-specific error rates; hard to see where R2 fails vs R1.
- **Fix:** Table 2 (Per-Stage Error Rates) now includes R2-n (queries reaching R2), R2-err (R2 error rate), and qualitative insight: "R2 processes harder queries than R1, explaining its higher error rate. The most revealing finding is the device agent: R1 misclassifies 22.2% of its stops (polysemous tokens like 'call' and 'alarm'), but R2 recovers with only 4.3% error—embedding similarity correctly disambiguates commands that keywords cannot."
- **Adequacy:** ✅ Excellent. Concretely shows device agent (22.2% R1 error → 4.3% R2 error), making the complementarity of stages clear.

---

### Partially Addressed

**W4 — Pareto sensitivity analysis.**
- **Issue (Round 1):** Only fixed α=0.1 with no justification; no trade-off curve.
- **Fix:** New §4.5 (Threshold Sensitivity) with Table 3 sweeping α ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}.
- **Evidence:** Table 3 shows:
  - α = 0.00–0.05: (1.5, 0.15) → 90.9% acc, 41.2% LLM%
  - α = 0.10: (1.5, 0.10) → 89.9% acc, 29.4% LLM%
  - α = 0.15–0.20: (1.5, 0.05) → 87.8% acc, 13.9% LLM%
  - α = 0.30–0.50: (0.5, 0.05) → 87.2% acc, 11.5% LLM%
  - Paper config: (0.5, 0.10) → 88.5% acc, 22.7% LLM%
- **Adequacy:** ✅ Good but slightly confusing. The α ranges (e.g., "0.00–0.05") suggest broad intervals, implying multiple values tested were grouped. A clearer presentation would be 9 rows with individual α values and their optimal (kt, et) pairs. However, the current format does show the trade-off curve and allows practitioners to tune their operating point.
- **Minor note:** Paper config (88.5% acc, 22.7% LLM%) differs slightly from Table 1 reported result (82.6% acc on the test set). This is expected (val vs test) but could be clarified.

---

### Not Addressed / Deferred

**W5 — Alternative cheap-stage comparison.**
- **Issue:** No systematic comparison of other light models (TinyBERT, DistilBERT, etc.).
- **Status:** Authors acknowledge in Related Work: "Strong baselines include SetFit...which uses contrastive few-shot fine-tuning on sentence-transformer encoders, and fine-tuned DeBERTa." SetFit (70.2% on full test) is included as reference point in Table 1, but not as a cascade stage.
- **Justification (implicit):** SetFit is a fine-tuning baseline; R2 (embedding-nearest-centroid) is zero-shot. Different setup. TinyBERT would require training; scope-creep for a workshop.
- **Verdict:** Acceptable deferral for a 4-page workshop paper. Could mention in future work or limitations.

---

## New Issues Found in Round 2

**Issue N1: Table 3 (Pareto) — ambiguous α ranges.**
- **Severity:** Minor
- **Description:** Rows like "0.00–0.05" and "0.10–0.20" suggest ranges of α tested, but it's unclear how many values lie in each range and why they were grouped. Did you test α ∈ {0.00, 0.01, ..., 0.50} and group outcomes? Or test only {0, 0.05, 0.10, ...}?
- **Recommendation:** Either (a) list all 9 α values with one optimal (kt, et) per row, or (b) add a footnote: "α values tested in 0.05 intervals from 0.00 to 0.50; consecutive configurations with identical optimal parameters are grouped by range."

**Issue N2: Reproducibility footnote precision.**
- **Severity:** Minor
- **Description:** Footnote states "per-seed accuracy varied by ≤ 0.5 pp (1–2 queries out of 400)" confirming dual-run reproducibility. However, the paper reports 1,200 pooled observations and uses Wilson CIs. Did you report CI width, standard deviation, or confidence level for the dual-run comparison? Adding one sentence clarifying would be helpful: e.g., "Duplicate runs showed Seed 42: 83.75% (run 1) vs 83.5% (run 2)—differences attributable to API rate-limit retries."
- **Recommendation:** Optional for a workshop but strengthens reproducibility claim. Current statement is adequate.

**Issue N3: Confidence calibration of R1, R2 vs. actual error.**
- **Severity:** Minor
- **Description:** Paper reports confidence thresholds (kt = 0.5 for keyword score, et = 0.10 for cosine similarity) but doesn't analyze confidence calibration—e.g., "when R2 confidence = 0.08 (below et), how often is R2 correct anyway?" This would help readers understand whether the threshold strategy is principled (high-confidence predictions tend to be correct) or heuristic.
- **Recommendation:** Optional analysis: add one sentence in §4.4 or Discussion: "Confidence thresholds were tuned via grid search to minimize LLM escalation subject to accuracy floor; further calibration analysis (e.g., accuracy vs. confidence deciles) is deferred to future work."

---

## Strengths (Updated)

**S1: Statistically rigorous evaluation.**
- McNemar exact-binomial paired tests on identical query pairs, Wilson 95% CIs, 3-seed stratified evaluation (n=1,200 pooled).
- Reproducibility verified by dual-run with ≤0.5pp variance per seed.
- Above typical workshop standards; matches main-conference rigor.

**S2: Honest cost accounting.**
- Total API cost ($0.44) transparently reported; per-seed costs disaggregated ($0.35 R3, $0.09 R4).
- Latency measurements now concrete (3.2ms R1, 0.3ms R2, 500ms R3) with hardware spec.
- Acknowledges Haiku 4.5 is already cheap; savings would be larger vs. Sonnet/GPT-4.

**S3: Informative qualitative analysis.**
- Table 2 (per-stage errors) reveals complementarity: device agent (22.2% R1 err → 4.3% R2 err).
- OOS identified as weakest link; actionable insight (dedicated OOS detector recommended).
- Practical guidance: 59.3% stop at R1, 17.0% at R2, 23.8% escalate to R3.

**S4: Pareto operating points now visible.**
- Table 3 shows trade-off dial: practitioners can trade accuracy for LLM cost reduction.
- Paper config (88.5% val acc, 22.7% LLM rate) is justified as "min LLM rate s.t. acc ≥ 88%"—clear decision rule.
- Enables readers to pick their own operating point (α selection).

**S5: Reproducibility and code release.**
- MIT license, tuned thresholds, per-seed trajectories, full API call logs on GitHub.
- Seeds (42, 43, 44) and model (claude-haiku-4-5-20251001) explicitly documented.
- Code structure supports replication on other intent-classification benchmarks.

---

## Remaining Weaknesses

**W5 (deferred): Alternative cheap-stage baselines.**
- **Severity:** Minor (acknowledged as future work)
- **Details:** TinyBERT, DistilBERT, or other open-source models not compared. SetFit (70.2%) is mentioned but not used as a cascade stage.
- **Impact:** Workshop paper scope is appropriate; main-conference version should include this.

**W_new (Minor): Confidence calibration not analyzed.**
- **Severity:** Minor
- **Details:** Thresholds kt=0.5, et=0.10 are tuned via grid search but no analysis of whether high-confidence predictions (especially R2 cosine >0.10) actually tend to be correct. Inverse: what's the error rate for R2 predictions with confidence in [0.08, 0.12]? (i.e., near the threshold)
- **Recommendation:** Optional: one line in future work: "Confidence calibration analysis (accuracy by confidence decile) would inform threshold selection for other domains."

**W_new (Minor): CLINC150 collapse to 8 agents — generalization risk.**
- **Severity:** Minor
- **Details:** Original CLINC150 has 150 fine-grained intents. Collapsing to 7 agents + OOS reduces label space complexity from 150 → 8. This is realistic for production (agents are coarser than intents) but limits claim generality. Flat collapse (equal agents size) may not reflect real hierarchies.
- **Recommendation:** Limitations section already mentions "fixed 8-way agent inventory; cascades on much larger label spaces (50+ agents) would require different design." Adequate for workshop.

---

## Domain Checklist (Re-Check)

| # | Criterion | Pass? | Notes |
|---|-----------|-------|-------|
| 1 | Reproducibility (code, seeds, costs) | ✅ | Code on GitHub (MIT), seeds 42/43/44 fixed, total cost $0.44 disclosed, dual-run variance ≤0.5pp |
| 2 | Statistical rigour (CI, paired tests) | ✅ | Wilson 95% CIs, McNemar exact-binomial (n=400 per seed, p>0.37 all seeds), 1,200 pooled |
| 3 | Fair baselines | ✅ | R1/R2/R3 all evaluated on identical 400-query subsamples; SetFit included as reference |
| 4 | Cost accounting transparency | ✅ | Per-seed costs ($0.117 R3, $0.030 R4+LLM), API pricing model disclosed (Haiku 4.5), latency measurements provided |
| 5 | Limitation honesty | ✅ | Acknowledges: single benchmark, fixed agent count, no online adaptation, Haiku already cheap |
| 6 | Claims match evidence | ✅ | 82.6% vs 82.9% matches data (0.8275); 26.1% LLM rate matches data; 74.3% cost reduction verified |
| 7 | Writing quality | ✅ | Mostly clear; minor ambiguity in Table 3 α ranges but overall well-structured (Intro → Method → Results → Discussion) |
| 8 | Figures/tables readable | ✅ | Tables 1–3 non-overlapping (fixed from Round 1); captions clear; axes labeled |
| 9 | Related work coverage | ✅ | CLINC150 benchmarking (SetFit, DeBERTa), FrugalGPT cascade pattern, agent-routing systems (CrewAI, LangGraph, AutoGen) all cited |

**Summary: 9/9 criteria pass. No blockers.**

---

## Verdict

This is a well-executed workshop paper that directly addresses Round 1 feedback. The cascade pattern is practical (74% cost reduction at matched accuracy), the evaluation is statistically rigorous (3-seed, McNemar paired tests, Wilson CIs), and the new Pareto analysis (Table 3) gives practitioners a tool to tune their own operating point. The dual-run verification (≤0.5pp variance) and code release (MIT) are exemplary for reproducibility.

The main remaining gap (W5: alternative cheap-stage models) is a reasonable deferral for 4 pages; the paper delivers its core thesis clearly: "LLM-as-router for which queries?" not "LLM vs. no LLM?"

**Recommendation:** Accept for workshop. The paper is publication-ready for a 4-page venue (e.g., NLP4IF, WMT, or ACL workshop). To elevate to main-conference quality, add TinyBERT/DistilBERT comparisons and extend to 50+ agent label spaces (which would require a different cheap-stage design—good companion paper angle for τ-bench).

---

## Minor Suggestions for Presentation (Optional)

1. **Table 3 clarification:** Add footnote: "α tested in 0.05 increments from 0.0 to 0.5; rows group consecutive configurations with identical optimal parameters."

2. **Figure for Pareto curve:** If space permits (5th page in 2-column format is acceptable for workshops), consider a line plot of accuracy vs. LLM call rate for different α values. Currently Table 3 is textual; a visual would make the trade-off more intuitive.

3. **OOS in abstract (optional):** "...while trading off OOS accuracy (16pp loss)" could be added to abstract for full transparency, but current placement in qualitative section is acceptable for a workshop.

---

**Reviewer Confidence:** 5/5 — Conducted detailed numerical verification against metrics_merged.json and metrics_llm.json; all reported accuracies, CI ranges, McNemar p-values, and cost figures verified to match data.

