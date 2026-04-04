# Cost-Aware Hybrid Router for LLM Agent Systems

> A keyword → embedding → LLM cascade that matches full-LLM routing accuracy at 53% lower cost on CLINC150.
>
> 透過 keyword → embedding → LLM 三層 cascade，在 CLINC150 上達到與全量 LLM 路由相同等級的準確率，成本降低 53%。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: CLINC150](https://img.shields.io/badge/dataset-CLINC150-green.svg)](https://huggingface.co/datasets/clinc_oos)

---

## TL;DR / 一句話

**EN:** Production agent frameworks (AutoGen, CrewAI, LangGraph) typically route every query through an LLM. This experiment shows that a simple keyword → embedding → LLM cascade achieves **88.1%** routing accuracy on CLINC150 while calling the LLM on only **47%** of queries — beating full-LLM routing (**85.6%**) at **53% lower cost**.

**中:** 主流 agent 框架（AutoGen、CrewAI、LangGraph）通常讓每一筆查詢都經過 LLM 路由。本實驗顯示，一個簡單的 keyword → embedding → LLM 三層 cascade 在 CLINC150 上達到 **88.1%** 準確率，且只對 **47%** 的查詢呼叫 LLM — 不但勝過全量 LLM 路由（**85.6%**），成本還降低 **53%**。

---

## Results / 實驗結果

**Evaluated on CLINC150 test set (5,500 queries, 150 intents mapped to 7 agents + OOS):**

**在 CLINC150 測試集上評估（5,500 筆查詢，150 個 intent 映射到 7 個 agent + OOS）：**

| Router | Accuracy / 準確率 | LLM Call Rate / LLM 呼叫率 | Cost (USD) / 成本 |
|---|---|---|---|
| R1 Keyword (regex rules) | 64.8% | 0% | $0 |
| R2 Embedding (TF-IDF centroid) | 74.0% | 0% | $0 |
| R3 LLM (Claude Haiku zero-shot) | 85.6%¹ | 100% | $0.047 / 160 queries |
| **R4 Hybrid cascade** | **88.1%¹** | **47%** | **$0.022 / 160 queries** |

¹ R3 and R4's LLM-touching numbers computed on a stratified sample (20 per agent = 160 queries) to keep API costs reasonable. R1, R2, and R4-no-LLM ran on the full 5,500. / R3 與 R4 的 LLM 相關數字在分層抽樣（每 agent 20 筆 = 160 筆）上計算，R1、R2、R4-no-LLM 跑完整 5,500 筆。

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

**Thresholds** were tuned on a held-out validation set via grid search: `keyword_threshold=1.5`, `embed_threshold=0.05`.

**閾值**在獨立的 validation set 上透過 grid search 調整：`keyword_threshold=1.5`、`embed_threshold=0.05`。

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
- **Validation-set threshold tuning** — no peeking at test set
- **All dependencies pinned** in `requirements.txt`

- **分層抽樣使用固定 seed**（seed=42）
- **Claude API 呼叫 temperature=0**
- **凍結的測試集** — 四個 router 跑同樣的 5,500 筆
- **在 validation set 上調整閾值** — 不偷看測試集
- **所有相依套件鎖定版本**於 `requirements.txt`

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

**Boyu Chen (陳柏宇)** — AI Engineer, applying to NTU CSIE (2027) — [GitHub](https://github.com/drewOrc)

Built as part of a grad school application to [MiuLab](https://www.csie.ntu.edu.tw/~miulab/) at NTU, studying LLM Agent Systems.

為申請台大資工所 [MiuLab](https://www.csie.ntu.edu.tw/~miulab/) 陳縕儂教授實驗室（研究 LLM Agent Systems）的實驗作品。
