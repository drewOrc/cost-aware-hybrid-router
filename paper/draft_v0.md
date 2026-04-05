# Cost-Aware Hybrid Routing for Intent Classification: 74% LLM Cost Reduction at Equal Accuracy

> Paper 1 of the "cost-aware cascade" series. Workshop target: NLP4ConvAI / EMNLP Industry / arXiv preprint.
> Target length: 4-page short paper + references.
> Status: skeleton v0, 2026-04-05. All numbers filled from `results/metrics_merged.json`.

---

## Abstract (target: 150 words)

Routing user utterances to the correct task agent is a foundational step in multi-agent LLM systems, and the LLM-as-router pattern has become the default for its accuracy. We ask whether that accuracy is *worth* its cost. On CLINC150 (150 intents → 7 agents + OOS), we compare four routers spanning a 3-order-of-magnitude cost range: (R1) keyword, (R2) embedding-nearest-centroid, (R3) Claude Haiku 4.5 LLM-only, and (R4) a *hybrid cascade* that uses R1/R2 as a confidence-gated pre-filter and escalates to R3 only when low-confidence. Across 3 seeds × 400 pooled queries, R4 achieves **82.6% ± 1.2pp** accuracy versus R3's **82.9% ± 0.6pp** — statistically indistinguishable by McNemar paired test (p > 0.37) — while escalating to the LLM on only **26.1%** of queries, yielding a **74.3% reduction in LLM cost**. We release code, thresholds, and per-seed trajectories.

---

## 1. Introduction (~0.75 pages)

### Hook
In production multi-agent LLM systems, every user utterance first visits a *router*: a component that decides which downstream agent (finance, HR, travel, …) should handle the request. The industry default is to use an LLM as the router — a Haiku-scale model prompted with the agent inventory and an "answer with JSON" instruction. This default is convenient but expensive: at scale, every single user turn incurs an LLM API call *before* the real work begins.

### The question
Is that LLM call worth it? A keyword match or embedding lookup is 3-4 orders of magnitude cheaper. If a cheap method can handle the *easy* queries with confidence, we only need the LLM on the hard ones. This paper quantifies how much of the routing workload is "easy" on CLINC150, and measures the accuracy/cost trade-off of a confidence-gated cascade.

### Contributions
1. **Four routers on the same benchmark:** keyword (R1), embedding (R2), LLM (R3), hybrid cascade (R4), tuned and evaluated under identical protocol.
2. **3-seed paper-grade evaluation:** 400-query stratified subsample per seed (seeds 42/43/44), with McNemar paired test and Wilson 95% CIs.
3. **Empirical finding:** R4 matches R3 accuracy within 0.4pp at 26% LLM call rate → 74% cost reduction. Difference is not statistically significant at α=0.05.
4. **Open source:** code, tuned thresholds, per-seed trajectories, LLM call logs (`github.com/drewOrc/cost-aware-hybrid-router`, MIT).

### Roadmap
§2 positions against intent-classification and cost-cascade literature. §3 describes the four routers and tuning protocol. §4 presents the 3-seed results and significance tests. §5 discusses when cascades do/don't help, limitations, and extension to multi-turn agent tasks (our companion work on τ-bench).

---

## 2. Related Work (~0.5 pages)

### Intent classification on CLINC150
CLINC150 (Larson et al. 2019) has 150 in-domain intents + an OOS class, originally proposed to stress-test intent classifiers under distribution shift. Modern baselines: SetFit (Tunstall et al. 2022) few-shot contrastive, fine-tuned DeBERTa, and LLM zero-shot prompting. We use SetFit as a strong baseline (our reproduction: **70.2%**) and LLM (Haiku 4.5) as the cost ceiling.

### Cost-aware LLM cascades
FrugalGPT (Chen et al. 2023) proposed the cascade pattern — try a cheap model, escalate if low confidence — on single-turn Q&A, reporting up to 98% cost reduction at matched accuracy. We port this to *routing*, a narrower but very high-frequency task where cost savings compound multiplicatively across agent turns.

### Agent-routing systems
Most multi-agent frameworks (CrewAI, LangGraph, AutoGen) use an LLM supervisor for routing by default. We do not argue against LLM routers; we argue for *pre-filtering* the easy queries before the LLM sees them.

### Position
Our contribution is not a new routing algorithm. It is the empirical question: *what fraction of CLINC150 routing decisions can a keyword or embedding handle, at what accuracy cost?* The answer informs a practical production choice.

---

## 3. Method (~1 page)

### 3.1 Dataset and Labels
We use CLINC150's test split (5,500 queries total). The 150 intents are mapped to 7 task agents + one OOS class (auto, device, finance, kitchen, meta, productivity, travel, oos), giving the router a realistic multi-domain agent inventory. Class sizes range from 360 to 1,140.

### 3.2 The Four Routers
**R1 — Keyword router.** Per-agent keyword lists extracted from training-set intent names + common synonyms. Score = max tf-idf overlap. Fixed.

**R2 — Embedding router.** Per-agent centroid in MPNet-base (all-mpnet-base-v2, 768d). Score = cosine similarity to nearest centroid. Fixed at training time.

**R3 — LLM router.** Claude Haiku 4.5 with a few-line system prompt listing the 8 labels and a JSON-output instruction. Temperature 0. Single call per query.

**R4 — Hybrid cascade.** Two thresholds (kt, et) on R1/R2 scores. Decision rule:
```
if R1_score ≥ kt:          return R1_pred
elif R2_score ≥ et:        return R2_pred
else:                      call R3(LLM) → return R3_pred
```
Thresholds tuned on a held-out validation split by grid search (kt ∈ {0.5, 1.0, 1.5, 2.0}, et ∈ {0.05, 0.08, 0.10, 0.15}), maximizing **accuracy − 0.1·LLM_call_rate** to favor configurations that substitute cheap calls for LLM calls at equal accuracy. Final tuned values: `kt=0.5, et=0.10`.

### 3.3 Evaluation Protocol
Per-seed stratified subsample of n=400 (50 per class × 8 classes) to bound LLM spend. 3 seeds: 42, 43, 44. Report per-seed and pooled accuracy, Wilson 95% confidence intervals (pooled n=1,200), and McNemar paired test comparing R3 vs R4 on the *same* queries per seed.

### 3.4 Cost Accounting
LLM cost computed from measured Anthropic API token usage. R1/R2 costs amortized to ~$0 (local compute). Total API spend for the full 3-seed experiment: **$0.44** ($0.35 for R3, $0.09 for R4).

---

## 4. Results (~1.25 pages)

### 4.1 Headline Table

| Router | Accuracy (pooled) | Wilson 95% CI | LLM Call Rate | Cost/query |
|---|---|---|---|---|
| R1 (keyword, full test n=5,500) | 64.8% | — | 0% | $0 |
| R2 (embedding, full test n=5,500) | 74.0% | — | 0% | $0 |
| SetFit baseline (full test n=5,500) | 70.2% | — | 0% | $0 |
| R4 no-LLM (R1→R2 only, n=5,500) | 74.9% | — | 0% | $0 |
| **R3 (LLM, 3 seeds pooled n=1,200)** | **82.9%** | [80.7, 84.9] | 100% | $9.8×10⁻⁵ |
| **R4 hybrid with LLM (same n=1,200)** | **82.6%** | [80.3, 84.6] | 26.1% | $2.5×10⁻⁵ |

**Reading:** R3 and R4+LLM have overlapping 95% CIs. The full no-LLM baseline R4-no-LLM is only 75%, so the LLM escalation is doing *real work* — just on 26% of queries instead of 100%.

### 4.2 Statistical Significance (McNemar)

| Seed | Δ accuracy (R4 − R3) | McNemar p-value | Significant @ α=0.05 |
|---|---|---|---|
| 42 | +0.00pp | 1.000 | No |
| 43 | −1.75pp | 0.371 | No |
| 44 | +0.75pp | 0.755 | No |

Across all 3 seeds, R4 is statistically indistinguishable from R3.

### 4.3 Cost Savings
Per-seed R3 cost: $0.117 (pooled 400 LLM calls at Haiku-4.5 pricing).
Per-seed R4+LLM cost: $0.030 (pooled ~105 LLM calls).
**LLM cost reduction: 74.3%.**

### 4.4 Where R1/R2 Succeed vs Escalate
*(To add: confusion by agent — which agents R1 handles cleanly (finance 83.6% R1 alone), which require LLM (meta_agent, device_agent).)*
*(To add: escalation rate by ground-truth class.)*

### 4.5 Figure Plan
- **F1:** Accuracy vs LLM call rate Pareto (scatter of tuning grid + final R3/R4 points).
- **F2:** Per-seed accuracy bars with error bars, R1/R2/R3/R4 side-by-side.
- **F3:** Per-agent accuracy heatmap (router × agent).
- **F4:** Cost per query, log-scale bar (R1 ≈ $0, R2 ≈ $0, R3 $9.8e-5, R4 $2.5e-5).

---

## 5. Discussion (~0.5 pages)

### When does the cascade pay off?
The cascade wins when the query distribution is *bimodal*: many easy queries answerable from surface features, plus a tail of hard queries needing LLM. CLINC150 is bimodal — domain-level keywords (finance terms, travel verbs) carry most of the signal, while meta-intents and cross-domain queries are ambiguous. Benchmarks with flat difficulty (all queries equally hard) would show less benefit.

### When does it fail?
If the cheap router's confidence calibration is poor, the hybrid *loses accuracy without saving cost*. Our tuning grid penalizes this via `accuracy − 0.1·LLM_call_rate`. In production this knob should be set from real LLM cost.

### Limitations
(1) Single benchmark (CLINC150). (2) Fixed agent inventory (8 classes). (3) Static thresholds — no online adaptation. (4) Haiku 4.5 is itself cheap; savings against GPT-4 would be larger.

### Extension: multi-turn agent tasks
The cascade pattern extends naturally to *multi-turn* agent settings: instead of routing every user turn, use a cheap model for task *decomposition* and save the expensive model for *execution*. We evaluate this hypothesis on τ-bench in companion work (Paper 2).

---

## 6. Conclusion (~0.2 pages)

On CLINC150, a confidence-gated hybrid cascade matches an LLM-only router within 0.4pp accuracy while reducing LLM calls by 74%. The choice is not "LLM-as-router vs. not" — it is "LLM-as-router for *which* queries." We release the full pipeline as a template for production teams making this cost/accuracy trade-off.

---

## Reproducibility Statement

Code, tuned thresholds, per-seed trajectories, LLM call logs: `github.com/drewOrc/cost-aware-hybrid-router` (MIT). Model version: `claude-haiku-4-5-20251001`. Seeds: 42, 43, 44. Embedding model: `sentence-transformers/all-mpnet-base-v2`. Total API cost to reproduce: **$0.44**.

---

## Writing checklist (before submission)

- [ ] Run numbers one more time from `metrics_merged.json` to confirm no drift
- [ ] Generate F1-F4 figures with `matplotlib` (script under `paper/figures.py`)
- [ ] Write concrete numbers into §4.4 (confusion + escalation-rate-by-class)
- [ ] Add 1-2 example queries per (R1-only / R2-only / escalated to R3) category as qualitative Table 2
- [ ] Double-check Wilson CI formula and McNemar exact-binomial p-values
- [ ] Citation pass: Larson 2019, Tunstall 2022, Chen 2023 (FrugalGPT), anthropic pricing page
- [ ] Limitations section ≥ 3 items, explicit
- [ ] Ethics statement (no human subjects, no PII)

---

## Notes to self

- Tone: terse, data-first, no adjectives. Reviewer should be able to replicate headline table in 2 min from the repo.
- The companion τ-bench paper (Paper 2) should be cited here as "companion work" once it has a preprint ID; for now, footnote.
- This paper can stand alone (the headline number holds), but the *thesis* only lands once Paper 2 shows the pattern transfers to multi-turn agents. Frame conclusion accordingly.
