# DEVLOG — Cost-Aware Hybrid Router

實驗進度與觀察。最新條目在最上面。

---

## 2026-04-05 (晚) — Full LLM evaluation 完成（3 seeds × n=400）

### 執行摘要
- 用 parallel evaluator (`src/evaluate_llm_parallel.py`) + ThreadPoolExecutor(6)
- 3 seeds × 400 queries = 1,200 pooled queries
- Anthropic Haiku 4.5 限速 ~50 RPM，實際 throughput ~1.0 req/s
- 每 seed ~7 分鐘，3 seeds 總計 ~22 分鐘 wall time
- **總 API 成本：$0.44 USD**（遠低於 $4 預算）

### 核心結果（tuned thresholds: kt=0.5, et=0.10）

| Router | Accuracy | LLM calls | Cost ($) |
|---|---|---|---|
| R1 keyword | 64.8% | 0 | 0 |
| R2 embedding (TF-IDF) | 74.0% | 0 | 0 |
| R4 hybrid no-LLM | 74.9% | 0 | 0 |
| SetFit baseline | 70.2% | 0 | 0 |
| **R3 LLM (Haiku)** | **82.9% ± 0.6pp** | 100% | $0.351 |
| **R4 hybrid + LLM** | **82.6% ± 1.2pp** | **26.1% ± 1.7pp** | **$0.090** |

### 🎯 關鍵發現 (headline)
> **R4+LLM 在只呼叫 26% LLM 的情況下，達到和 full-LLM 幾乎一樣的準確率
> (82.6% vs 82.9%)，成本降低 74%。McNemar test 在 3/3 seeds 都 not significant
> (p=1.00, 0.37, 0.76)，表示兩者 statistically indistinguishable。**

Wilson 95% CI (pooled n=1200):
- R3:     [80.7%, 84.9%]
- R4+LLM: [80.3%, 84.6%]
- 區間完全重疊 → 準確率差異無顯著性

### McNemar 逐 seed 結果
| seed | R3 acc | R4+LLM acc | delta | p-value | sig |
|---|---|---|---|---|---|
| 42 | 83.5% | 83.5% | +0.00pp | 1.0000 | No |
| 43 | 83.0% | 81.2% | -1.75pp | 0.3713 | No |
| 44 | 82.2% | 83.0% | +0.75pp | 0.7552 | No |

### 論文 claim (可 defensible)
1. **Cost reduction**：在不損失 accuracy 的前提下，hybrid router 把 LLM 呼叫
   從 100% 降到 26%（74% 成本節省）。
2. **No accuracy loss**：3/3 seeds McNemar not significant, Wilson CI 完全重疊。
3. **Cascade 的設計哲學有效**：keyword → embedding → LLM 的三階段篩選能把
   "確定的 query" 擋在便宜階段處理掉，只把真正困難的 case 送給 LLM。

### 仍存在的限制（要在 paper 裡誠實揭露）
1. **LLM ceiling 只有 83%**：即使呼叫 Claude Haiku zero-shot，CLINC150 8-class
   分類也只能到 83%。這個 ceiling 被 OOS 類別拖累（之前 DEVLOG 觀察）。
2. **R4-no-LLM 74.9% 已經接近 ceiling 的 90%**：即使沒有 LLM，hybrid 也能達
   到 R3 的 90%。LLM fallback 的邊際貢獻只有 +7.7pp（不大）。
3. **OOS 仍是共同瓶頸**：R4+LLM 對 OOS 的提升 gap 需要進一步分析
   （下一步 TODO）。

### Next
- [x] 3-seed LLM evaluation
- [x] McNemar + Wilson CI
- [ ] 重繪 accuracy-vs-cost scatter plot（加入 error bars + SetFit）
- [ ] 分析 R4+LLM 對 OOS 的具體提升
- [ ] 更新 research_statement / README 用新數據（82.6% ± 1.2pp）

### Files
- `src/evaluate_llm_parallel.py` — parallel eval with retry/backoff
- `src/merge_seeds.py` — 合併 3 個 seed 成 metrics_merged.json
- `results/metrics_seed{42,43,44}.json` — per-seed raw data
- `results/metrics_merged.json` — final aggregated metrics

---

## 2026-04-05 — Paper-grade upgrades (no-LLM portion)

### 本次變更
1. 加入 `src/stats.py`（McNemar test、Wilson CI、bootstrap CI、多 seed 聚合）
2. 加入 SetFit baseline：`src/routers/setfit_router.py` + `src/train_setfit.py`
3. 加入 `src/tune.py` 在 validation set 上 grid search thresholds
4. 重寫 `src/evaluate.py` 支援 `--llm-n`、`--seeds`、`--setfit`、`--use-tuned-thresholds`
5. 加入 `RUN_UPGRADED.md` 執行步驟

### Tuned thresholds（on validation set, n=3,100）
- **No-LLM 最佳配置**：kt=1.5, et=0.05 → val acc 83.2%
- **With-LLM 最佳配置**（min LLM calls s.t. expected_acc ≥ 0.88）：
  kt=0.5, et=0.10 → expected acc 88.5%, LLM call rate 22.7%

### SetFit 訓練
- Backbone: `paraphrase-MiniLM-L3-v2`（~60MB）
- Few-shot: 16/agent × 8 agents = 128 training pairs
- 1 epoch, 896 steps, train_loss 0.107
- Wall time: 103 秒 (CPU, Linux 沙箱)
- **Validation accuracy: 77.9% (2415/3100)**

### Full-test 評估（n=5,500, tuned thresholds）

| Router | Acc | Notes |
|---|---|---|
| R1 keyword | 64.8% | 15 秒 |
| R2 embedding (TF-IDF) | 74.0% | 1.7 秒 |
| R4 hybrid no-LLM | **74.9%** | kt=1.5, et=0.05 |
| **SetFit baseline** | 70.2% | paraphrase-MiniLM-L3-v2, 16-shot |

R4 hybrid 的 stage 分佈：
- Keyword accepted: 2,282 (41.5%)
- Embedding accepted: 3,218 (58.5%)
- LLM fallback: 0 (no-LLM 版本)

### 🚨 關鍵發現：OOS 是共同瓶頸

| Router | non-OOS (n=4,500) | OOS only (n=1,000) |
|---|---|---|
| R1 keyword | ~61% | **80.0%** ✓ |
| R2 embedding | ~85% | 26.4% ✗ |
| R4 hybrid no-LLM | ~86% | 26.2% ✗ |
| SetFit | ~78% | 39.3% ✗ |

解讀：
- **TF-IDF 和 SetFit 會強制把 OOS query 分到某個 domain**，因為它們被訓練成 8-class
  classifier（包含 oos 類）但 centroid/prototype 學不太出 "不屬於任何 domain"
  的特徵。
- **Keyword router 反而做對了**：在 intent-level keyword lookup 沒找到匹配時，
  預設為 oos 是個強 prior。
- R4 top confusion 前 7 名全是 `oos → 某 agent`（149、141、113、102、85、80、68 次）。
- **這是 R4+LLM fallback 應該要補的洞**：對於 R1+R2 都不確定的 query（多半是
  OOS），呼叫 LLM 判斷。

### 📊 SetFit 意外低於 TF-IDF 的解讀
- 原因猜測：
  1. backbone 太小（L3-v2 只有 3 層 transformer）
  2. 16-shot 可能不夠，should try 32 or 64
  3. OOS 類別的對比學習特別困難
- 這個結果本身是個可發表的 finding：**"fine-tuned SentenceTransformer ≠ 自動贏過
  TF-IDF"**，支持 hybrid cascade 的論點：用最便宜的方法處理多數，只在必要時
  加碼。
- 但要下這個結論前，應該先跑 L6-v2 或 L12-v2 backbone 試試看。

### 📋 還需要在 Mac 上跑（需要 API key）
1. `evaluate.py --llm-n 1000 --seeds 42 43 44 --use-tuned-thresholds --setfit`
2. 預期產出：
   - R3 LLM multi-seed mean ± std
   - R4+LLM multi-seed mean ± std
   - McNemar test：R4+LLM vs R3 是否 significant
   - Wilson CI for both
3. 預算：~$2-4 USD API cost, ~20-25 分鐘

### Next actions
- [ ] 本地跑 LLM 部分 → 取得 R3 / R4+LLM 完整數據
- [ ] 分析 R4+LLM 對 OOS 的提升幅度
- [ ] 考慮升級 SetFit backbone 到 L6/L12 看 baseline 是否翻盤
- [ ] 更新 README 加入 tuned thresholds 和 full-test SetFit 結果
- [ ] 更新 research_statement 引用新數據（移除 "~86%"，用實測值）

---

## 2026-04-04 — Initial experiment & honest audit

### 初版實驗（n=160 for LLM）
- R1 keyword: 64.8%
- R2 embedding: 74.0%
- R4 hybrid (no-LLM): 74.9%
- R3 LLM (n=160 sample): 85.6%
- R4+LLM (n=160 sample): 88.1%

### 自我審查發現的 5 個問題
1. Stale comment："每 agent 50 筆" 實際是 20 筆 → 已修
2. README 宣稱 "thresholds tuned on validation set" → 其實是手動 → 已修
3. R4 (n=5500) vs R3 (n=160) 直接比較 = apples-to-oranges → README 加註
4. R4 no-LLM 和 R4+LLM 用不同 embed_threshold → 已揭露
5. 缺 Limitations section → 已補

### 本次升級要解決的 5 項不足
（見上方 2026-04-05 條目）
