#!/usr/bin/env bash
# Run this locally to commit + push the paper-grade upgrade.
# (The sandbox cannot write to .git, so commit must run on your machine.)

set -e
cd "$(dirname "$0")"

# Clear any stale lock file
rm -f .git/index.lock

git add -A

git commit -m "Upgrade to paper-grade evaluation: 3 seeds + McNemar + SetFit + tuned thresholds

Major changes since initial commit:
- Grid-search tuned thresholds on validation set (src/tune.py)
  - no-LLM: kt=1.5, et=0.05 (74.9% on full test)
  - with-LLM: kt=0.5, et=0.10 (min LLM rate s.t. expected acc >= 88%)
- Multi-seed LLM evaluation across seeds 42/43/44, n=400 per seed (n=1200 pooled)
- Parallel evaluator with ThreadPoolExecutor + exponential backoff retry
  (src/evaluate_llm_parallel.py)
- Statistical validation (src/stats.py):
  - McNemar's exact binomial test between R4+LLM and R3 on shared samples
  - Wilson score interval for pooled accuracy
  - bootstrap CI utility
- SetFit 16-shot baseline (paraphrase-MiniLM-L3-v2): 70.2% on full test

Final validated results:
- R3 LLM:          82.9% +/- 0.6pp  (cost \$0.88/1k queries)
- R4 hybrid+LLM:   82.6% +/- 1.2pp  LLM-call rate 26.1% +/- 1.7pp  (cost \$0.23/1k)
- McNemar: p = 1.00 / 0.37 / 0.76 across 3 seeds -> not significant in 3/3
- Wilson 95% CI (pooled n=1200): intervals fully overlap
- Conclusion: R4+LLM matches R3 accuracy at ~4x lower cost, statistically
  indistinguishable.

README and DEVLOG rewritten to reflect validated multi-seed numbers."

git push origin main
echo "Done. Pushed to origin/main."
