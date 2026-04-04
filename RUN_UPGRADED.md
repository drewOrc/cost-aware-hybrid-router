# Running the Upgraded Experiment (Paper-Grade)

This guide walks through reproducing the **n ≥ 1000, multi-seed, McNemar-tested,
SetFit-baselined** results locally on Mac.

> Expected wall-time on Mac M1/M2: ~15–25 minutes total (not counting SetFit
> training, which adds ~3–5 minutes).
> Expected Anthropic API cost: **~$2–4 USD** total (Claude Haiku at $0.80/M in,
> $4/M out, ~1000 queries × 2 runs × 3 seeds).

---

## 0. One-time setup

```bash
cd ~/path/to/cost-aware-hybrid-router

# Install dependencies
pip install -r requirements.txt
# For SetFit baseline (optional but recommended):
pip install setfit>=1.0.0 sentence-transformers>=3.0.0

# Download CLINC150 if not already
python download_data.py

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 1. Tune thresholds on validation set (no peeking at test)

```bash
PYTHONPATH=. python3 src/tune.py
```

Output → `results/tuned_thresholds.json`

This runs grid search over:
- `keyword_threshold ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}`
- `embed_threshold ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}`

on the **validation split** (3,100 queries), selecting:
- **no_llm config**: maximize accuracy
- **with_llm config**: minimize LLM call rate s.t. expected_acc ≥ 0.88

---

## 2. (Optional) Train SetFit baseline

```bash
PYTHONPATH=. python3 src/train_setfit.py
```

- Backbone: `sentence-transformers/paraphrase-MiniLM-L3-v2` (~60 MB)
- Few-shot: 16 examples × 8 classes = 128 training pairs
- Output → `models/setfit-agent-classifier/`

---

## 3. Run full evaluation (multi-seed + McNemar)

```bash
PYTHONPATH=. python3 src/evaluate.py \
    --llm-n 1000 \
    --seeds 42 43 44 \
    --use-tuned-thresholds \
    --setfit
```

Arguments:
- `--llm-n 1000` → 125/agent × 8 classes = 1000 queries per seed for LLM eval
- `--seeds 42 43 44` → 3 seeds, each with a different stratified sample
- `--use-tuned-thresholds` → load thresholds from step 1
- `--setfit` → include SetFit baseline on full test set

Output:
- `results/metrics.json` — raw per-seed metrics
- stdout — summary table with mean ± std and McNemar p-values

---

## 4. Regenerate figures

```bash
PYTHONPATH=. python3 src/analysis.py
```

Updates `results/figures/accuracy_vs_cost.png` with new Pareto plot including
SetFit baseline and error bars.

---

## 5. (Optional) Quick sanity-check run (no API, no SetFit)

```bash
PYTHONPATH=. python3 src/evaluate.py --llm-n 160 --seeds 42 --skip-full
```

Skips the deterministic full-test runs for fast iteration.

---

## Expected output shape

```
Summary
======
  R1_keyword              acc= 64.8%  (n=5500)
  R2_embedding            acc= 74.0%  (n=5500)
  R4_hybrid_no_llm        acc= ~75%   (n=5500)
  SetFit_baseline         acc= ~83-87% (n=5500)   ← strong baseline
  R3_llm (multi-seed)     acc= 85.6% ± X.Xpp  (n_seeds=3)
  R4_hybrid_with_llm (ms) acc= 88.1% ± X.Xpp
  McNemar significant in Y/3 seeds
```

The **key statistical claim** to support in the paper:
> Across 3 seeds (n=1000 each), R4+LLM outperforms R3 by X pp ± Y pp, with
> McNemar's test significant at p<0.05 in Y/3 seeds.

If Y = 0/3: claim falls apart → rewrite as "indistinguishable within CI,
cost advantage is the real win"
If Y = 2-3/3: claim holds → paper-ready

---

## Troubleshooting

- **`setfit` import fails**: `pip install setfit sentence-transformers`
- **scipy import fails**: `pip install scipy`
- **API rate limit**: add `time.sleep(0.2)` between calls in `llm_router.py` or
  reduce `--llm-n`
- **OOM during SetFit training**: reduce `N_PER_AGENT` in `train_setfit.py`
  from 16 → 8
