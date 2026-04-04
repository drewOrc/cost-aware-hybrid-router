# Cost-Aware Hybrid Router for LLM Agent Systems

> A keyword → embedding → LLM cascade that matches full-LLM routing accuracy at 53% lower cost on CLINC150.
>
> 透過 keyword → embedding → LLM 三層 cascade，在 CLINC150 上達到與全量 LLM 路由相同等級的準確率，成本降低 53%。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: CLINC150](https://img.shields.io/badge/dataset-CLINC150-green.svg)](https://huggingface.co/datasets/clinc_oos)

---

## TL;DR / 一句話

**EN:** Production agent frameworks (AutoGen, CrewAI, LangGraph) typically route every query through an LLM. This experiment explores an alternative: a keyword → embedding → LLM cascade. On CLINC150 we observe that the cascade (with LLM fallback) matches full-LLM routing accuracy (**88.1% vs 85.6%** on a 160-query sample) while calling the LLM on only **47%** of queries, suggesting ~53% lower routing cost. **Note:** the accuracy comparison is on a small sample (n=160) for cost reasons and should be read as exploratory — see Limitations.

**中:** 主流 agent 框架（AutoGen、CrewAI、LangGraph）通常讓每一筆查詢都經過 LLM 路由。本實驗探索另一條路：keyword → embedding → LLM 三層 cascade。在 CLINC150 上觀察到，加入 LLM fallback 的 cascade 與全量 LLM 路由的準確率相當（n=160 樣本上 **88.1% vs 85.6%**），但只對 **47%** 的查詢呼叫 LLM，暗示約 **53%** 的路由成本節省。**注意：** 準確率比較受限於 API 成本在小樣本（n=160）上進行，應視為探索性結果 — 詳見 Limitations。

---

## Results / 實驗結果

**Evaluated on CLINC150 test set (5,500 queries, 150 intents mapped to 7 agents + OOS):**

**在 CLINC150 測試集上評估（5,500 筆查詢，150 個 intent 映射到 7 個 agent + OOS）：**

| Router | Accuracy / 準確率 | Sample Size / 樣本 | LLM Call Rate / LLM 呼叫率 | Cost (USD) / 成本 |
|---|---|---|---|---|
| R1 Keyword (regex rules) | 64.8% | n = 5,500 (full test) | 0% | $0 |
| R2 Embedding (TF-IDF centroid) | 74.0% | n = 5,500 (full test) | 0% | $0 |
| R4 Hybrid cascade (no LLM) | 74.9% | n = 5,500 (full test) | 0% | $0 |
| R3 LLM (Claude Haiku zero-shot) | 85.6%¹ | n = 160 (stratified) | 100% | $0.047 |
| **R4 Hybrid cascade (with LLM)** | **88.1%¹** | **n = 160 (stratified)** | **47%** | **$0.022** |

¹ **Important caveat:** R3 and R4-with-LLM were evaluated on a stratified sample (20 per agent = 160 queries) to keep API costs reasonable. The 2.5-point gap between R4 (88.1%, 141/160) and R3 (85.6%, 137/160) reflects only a 4-query difference and should be read as exploratory, not as a statistically significant claim. The 95% confidence interval at n=160 is roughly ±5 percentage points. See the **Limitations** section below.

¹ **重要說明：** R3 與 R4+LLM 在分層抽樣（每 agent 20 筆 = 160 筆）上評估以控制 API 成本。R4（88.1%，141/160）與 R3（85.6%，137/160）之間 2.5 個百分點的差距僅反映 4 筆 query 的差異，應視為探索性結果，**而非具統計顯著性的結論**。n=160 時 95% 信賴區間約為 ±5 個百分點。詳見下方 **Limitations** 章節。

![Accuracy vs Cost Pareto frontier](results/figures/accuracy_vs_cost.png)

---

## Quick Start / 快速開始

### 1. Clone & install / 安裝

```bash
git clone https://github.com/drewOrc/cost-aware-hybrid-router.git
cd cost-aware-hybrid-router
pip install -r requirements.txt
```

### 2. Download dataset / 下載資料集

```bash
python download_data.py
```

This downloads CLINC150 from HuggingFace `clinc_oos` (plus variant) and saves the train/validation/test splits as JSON under `data/clinc150/`.

此指令會從 HuggingFace 的 `clinc_oos` (plus variant) 下載 CLINC150，並將 train/validation/test 資料存成 JSON 於 `data/clinc150/` 底下。

### 3. Set API key (only needed for R3/R4-with-LLM) / 設定 API key（只有 R3 和 R4+LLM 需要）

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

You can get a key at https://console.anthropic.com/ — the full evaluation uses ~$0.07 of Haiku credits.

可以在 https://console.anthropic.com/ 申請 API key，跑完整評估約使用 Haiku 點數 $0.07 美元。

### 4. Run evaluation / 執行評估

```bash
PYTHONPATH=. python3 src/evaluate.py
```

This runs all four routers and writes `results/metrics.json`. Takes ~4 minutes (most of it is the 160 LLM calls).

此指令會跑完四個 router 並將結果寫入 `results/metrics.json`。大約需要 4 分鐘（主要是 160 次 LLM 呼叫）。

### 5. Generate figures / 產生圖表

```bash
PYTHONPATH=. python3 src/analysis.py
```

Writes three plots to `results/figures/`:
- `accuracy_vs_cost.png` — Pareto frontier scatter plot
- `per_agent_accuracy.png` — per-agent accuracy comparison
- `hybrid_flow.png` — query flow distribution through the cascade

輸出三張圖到 `results/figures/`：
- `accuracy_vs_cost.png` — Pareto frontier 散佈圖
- `per_agent_accuracy.png` — 各 agent 準確率比較
- `hybrid_flow.png` — cascade 中查詢的流向分佈

---

## Architecture / 架構

```
Query ─┐
       ▼
   ┌────────────┐
   │ R1 Keyword │  confidence ≥ threshold? ──► YES → return
   │  (regex)   │                                NO ↓
   └────────────┘
                                                  ▼
                                        ┌────────────────┐
                                        │ R2 Embedding   │  confidence ≥ threshold? ──► YES → return
                                        │ (TF-IDF cosim) │                                NO ↓
                                        └────────────────┘
                                                                                           ▼
                                                                               ┌────────────────┐
                                                                               │ R3 LLM (Haiku) │ ──► return
                                                                               │   (zero-shot)  │
                                                                               └────────────────┘
```

**Thresholds** are currently set manually at `keyword_threshold=1.5`, `embed_threshold=0.05` (no-LLM variant) / `0.15` (with-LLM variant). A grid-search helper (`tune_thresholds()` in `hybrid_router.py`) is provided for validation-set tuning, but has not yet been wired into the main evaluation — this is flagged under **Limitations** below.

**閾值**目前手動設定為 `keyword_threshold=1.5`、`embed_threshold=0.05`（no-LLM 變體）/ `0.15`（with-LLM 變體）。`hybrid_router.py` 中有 `tune_thresholds()` 函數可在 validation set 上做 grid search，但尚未整合進主評估流程 — 這點在下方 **Limitations** 中標註。

---

## Why This Matters / 為什麼這個結果重要

### The counterintuitive finding / 反直覺的發現

Hybrid is **more accurate** than LLM-only (88.1% vs 85.6%), not just cheaper. Why?

Hybrid 不只比 LLM-only 便宜，還**更準確**（88.1% vs 85.6%）。為什麼？

**EN:** Keyword and embedding routers are more *cautious*. When a query strongly matches keyword patterns (e.g., "transfer $500 to my savings account"), keyword is confident and correct. When nothing matches, the cascade hands off to embedding, then to LLM. The LLM only gets invoked on the ~47% of queries where cheap routers are uncertain — exactly the queries where LLM reasoning is worth paying for. Full-LLM routing, by contrast, *always* runs the LLM, which means it pays for the LLM's mistakes on queries keyword would have gotten right for free.

**中:** Keyword 和 embedding router 更「謹慎」。當查詢強匹配 keyword 樣式（例如「轉 500 美元到我的儲蓄帳戶」）時，keyword 有信心而且答對。當沒匹配到時，cascade 交棒給 embedding，再到 LLM。LLM 只在 ~47% 的查詢（便宜 router 不確定的那些）才被呼叫 — 正是 LLM 推理值得付費的查詢。相反地，Full-LLM 路由永遠跑 LLM，意思是它要為 LLM 在那些 keyword 免費就能答對的查詢上犯的錯誤買單。

### Failure mode taxonomy / 失敗模式分類

| Type | Symptom | Who suffers | Fix |
|---|---|---|---|
| **A — Keyword coverage gap** | Query defaults to OOS when no pattern matches | Meta, device, productivity agents (36–66% misrouted to OOS) | Fallthrough to embedding or LLM |
| **B — Cross-agent semantic overlap** | TF-IDF centroids confuse similar domains | "productivity" vs "meta" (shared vocab: remind, schedule, note) | LLM stage disambiguates |
| **C — Embedding OOS blindness** | Centroids project every query onto *some* agent | OOS accuracy: 26% (embedding) vs 80% (keyword) | Keep keyword's fail-closed OOS default |

| 類型 | 症狀 | 誰受影響 | 解法 |
|---|---|---|---|
| **A — Keyword 覆蓋缺口** | 沒匹配到樣式就預設 OOS | meta、device、productivity agent（36–66% 誤路由到 OOS） | fallthrough 到 embedding 或 LLM |
| **B — 跨 agent 語意重疊** | TF-IDF centroid 搞混相近領域 | 「productivity」vs「meta」（共享詞彙：提醒、排程、筆記） | LLM 階段消除歧義 |
| **C — Embedding OOS 盲點** | Centroid 將每筆查詢投射到某個 agent | OOS 準確率：26%（embedding）vs 80%（keyword） | 保留 keyword 的 fail-closed OOS 預設 |

---

## Folder Structure / 檔案結構

```
cost-aware-hybrid-router/
├── README.md
├── LICENSE                          ← MIT
├── requirements.txt
├── download_data.py                 ← download CLINC150 from HuggingFace
├── data/
│   ├── clinc150/                    ← raw data (gitignored; run download_data.py)
│   └── intent_to_agent.json         ← 150 intents → 7 agents + OOS mapping
├── src/
│   ├── routers/
│   │   ├── keyword_router.py        ← R1: weighted regex patterns
│   │   ├── embedding_router.py      ← R2: TF-IDF + cosine similarity
│   │   ├── llm_router.py            ← R3: Claude Haiku zero-shot
│   │   └── hybrid_router.py         ← R4: cascade + threshold tuning
│   ├── evaluate.py                  ← run all routers, save metrics.json
│   └── analysis.py                  ← generate figures from metrics.json
└── results/
    ├── metrics.json                 ← full evaluation results
    └── figures/
        ├── accuracy_vs_cost.png
        ├── per_agent_accuracy.png
        └── hybrid_flow.png
```

---

## Reproducibility / 可重現性

- **Fixed seed** for stratified sampling (seed=42)
- **Temperature=0** for Claude API calls
- **Frozen test set** — same 5,500 queries across all routers
- **All dependencies pinned** in `requirements.txt`

- **分層抽樣使用固定 seed**（seed=42）
- **Claude API 呼叫 temperature=0**
- **凍結的測試集** — 四個 router 跑同樣的 5,500 筆
- **所有相依套件鎖定版本**於 `requirements.txt`

---

## Limitations / 研究限制

This is a **preliminary exploration**, not a rigorous empirical claim. A peer reviewer would rightly flag the following:

這是一個**初步探索實驗**，不是嚴謹的實證論斷。同儕審閱者會合理地指出以下幾點：

1. **Small LLM-evaluation sample (n=160).** R3 and R4-with-LLM were only evaluated on 160 queries (20 per agent) to keep API costs under $0.10. At this sample size, the 95% CI on accuracy is roughly ±5 percentage points, so the R4 vs R3 gap (88.1% vs 85.6%, a 4-query difference) is **not statistically significant** without further testing. No McNemar's test or bootstrap CI has been computed yet.

   **LLM 評估樣本過小 (n=160)。** R3 和 R4+LLM 僅在 160 筆（每 agent 20 筆）上評估以將 API 成本控制在 $0.10 以下。在此樣本規模下，準確率的 95% 信賴區間約為 ±5 個百分點，因此 R4 與 R3 的差距（88.1% vs 85.6%，僅 4 筆 query 差異）在未做進一步檢定前**不具統計顯著性**。目前尚未計算 McNemar's test 或 bootstrap CI。

2. **Single random seed, no variance reporting.** All experiments ran once with seed=42. No mean ± std reported across multiple seeds.

   **單一隨機種子，無變異數報告。** 所有實驗僅以 seed=42 跑一次，未跨多個 seed 報告 mean ± std。

3. **Thresholds set manually, not tuned.** The `keyword_threshold=1.5` and `embed_threshold` values were chosen by intuition and light inspection, not via systematic grid search on a validation set. A `tune_thresholds()` helper exists in `hybrid_router.py` but has not been integrated into the main evaluation loop.

   **閾值為手動設定，未調參。** `keyword_threshold=1.5` 與 `embed_threshold` 的數值是憑直覺與簡單檢視選擇，不是在 validation set 上系統性 grid search 的結果。`hybrid_router.py` 中雖有 `tune_thresholds()` 函數，但尚未整合進主評估流程。

4. **Different `embed_threshold` for R4 variants.** R4-no-LLM uses `0.05` (lenient, since there's no LLM fallback) while R4-with-LLM uses `0.15` (stricter, triggers more fallback). This means the two R4 variants are not identical configurations, which complicates the "effect of adding LLM" attribution.

   **R4 兩個變體使用不同的 `embed_threshold`。** R4-no-LLM 使用 `0.05`（較寬鬆，因為沒有 LLM fallback），R4-with-LLM 使用 `0.15`（較嚴格，觸發更多 fallback）。這意味著兩個 R4 變體並非完全相同的配置，這使「加入 LLM 的效果」的歸因變得複雜。

5. **OOS detection is weak across the cascade.** On the full test set, embedding-based routing gets only 26% accuracy on OOS queries (vs 80% for keyword's fail-closed default). The LLM variant recovers some of this (60% on n=20 OOS sample), but OOS handling is an unresolved weakness in the current cascade design.

   **整個 cascade 對 OOS 偵測都偏弱。** 在完整測試集上，embedding routing 對 OOS 查詢僅達 26% 準確率（vs. keyword 的 fail-closed 預設 80%）。LLM 變體有部分恢復（n=20 的 OOS 樣本上為 60%），但 OOS 處理仍是當前 cascade 設計中未解決的弱點。

6. **Single dataset, single LLM model.** Evaluated only on CLINC150 (English, commercial domains) with only Claude Haiku. Generalization to other datasets, languages, or LLMs is untested.

   **單一資料集、單一 LLM 模型。** 僅在 CLINC150（英文、商業領域）上、僅以 Claude Haiku 評估。對其他資料集、語言或 LLM 的泛化能力尚未驗證。

### What would be needed to upgrade this to a paper-grade result / 升級為論文等級結果所需的工作

- Scale R3/R4-with-LLM evaluation to **n ≥ 1000** (budget ~$5).
- Run **3–5 seeds**, report mean ± std.
- Add **McNemar's test** for R4 vs R3 comparison on the shared sample.
- Actually invoke `tune_thresholds()` and report the grid-search results.
- Add a stronger baseline (e.g., SetFit or DistilBERT fine-tuned on CLINC150 train split).
- Evaluate on a second intent-classification dataset (e.g., BANKING77 or HWU64) for cross-dataset validation.

- 將 R3/R4+LLM 評估規模擴大到 **n ≥ 1000**（預算約 $5）。
- 跑 **3–5 個 seed**，報告 mean ± std。
- 對 R4 vs R3 在共同樣本上的比較加上 **McNemar's test**。
- 實際呼叫 `tune_thresholds()` 並報告 grid search 結果。
- 加入更強的 baseline（例如在 CLINC150 train split 上 fine-tune 的 SetFit 或 DistilBERT）。
- 在第二個意圖分類資料集（如 BANKING77 或 HWU64）上評估以做跨資料集驗證。

---

## Citation / 引用

If you find this useful, please cite: / 如果對您有幫助，請引用：

```bibtex
@misc{chen2026hybrid,
  author = {Chen, Boyu},
  title  = {Cost-Aware Hybrid Router for LLM Agent Systems},
  year   = {2026},
  url    = {https://github.com/drewOrc/cost-aware-hybrid-router}
}
```

---

## License / 授權

MIT License — see [LICENSE](LICENSE).

---

## Author / 作者

**Boyu Chen (陳柏宇)** — AI Engineer — [GitHub](https://github.com/drewOrc)

