# 成本感知混合路由器：LLM Agent 系統

關鍵詞 → 嵌入向量 → LLM 級聯架構，在 CLINC150 上達到與純 LLM 路由相同的準確率，同時僅對 ~26% 的查詢呼叫 LLM。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: CLINC150](https://img.shields.io/badge/dataset-CLINC150-green.svg)](https://huggingface.co/datasets/clinc_oos)

[English README](README.md)

---

## 核心結果

現有的 Agent 框架（AutoGen、CrewAI、LangGraph）通常將每個查詢都送到 LLM 處理。本專案評估一種更低成本的替代方案：基於信心度閘門的關鍵詞 → 嵌入向量 → LLM 級聯路由。在 CLINC150 上以 3 個隨機種子評估，級聯路由的準確率與純 LLM 路由無顯著差異（**82.6% ± 1.2pp vs 82.9% ± 0.6pp**），同時僅使用 **26.1%** 的 LLM 呼叫——**節省 74% 的 LLM 成本**。McNemar 精確檢定在所有種子上均不顯著（p = 1.000, 0.371, 0.755）。

| 路由器 | 準確率 | LLM 呼叫率 | 每種子成本 |
|---|---|---|---|
| R1 關鍵詞 | 64.8% | 0% | $0 |
| R2 嵌入向量（MPNet 質心） | 74.0% | 0% | $0 |
| SetFit 基線（16-shot） | 70.2% | 0% | $0 |
| R4 級聯（無 LLM） | 74.9% | 0% | $0 |
| R3 純 LLM（Haiku 4.5） | **82.9% ± 0.6pp** | 100% | $0.117 |
| **R4 級聯 + LLM** | **82.6% ± 1.2pp** | **26.1%** | **$0.030** |

評估資料集：CLINC150 測試集（5,500 查詢，150 意圖 → 7 agent + OOS）。含 LLM 行：3 種子 × 400 分層採樣查詢（合計 n=1,200）。

![Accuracy vs Cost](paper/figures/F1_accuracy_vs_llm_rate.png)

---

## 架構

```
查詢 ──► R1 關鍵詞 ──[高信心度]──► 回傳預測
              │
              ▼ (低信心度)
         R2 嵌入向量 ──[高信心度]──► 回傳預測
              │
              ▼ (低信心度)
         R3 LLM (Haiku 4.5) ──► 回傳預測
```

閾值透過 CLINC150 驗證集的網格搜索調整（`src/tune.py`）。含 LLM 級聯參數：`kt=0.5, et=0.10`。結果存於 `results/tuned_thresholds.json`。

---

## 快速開始

```bash
# 複製 & 安裝
git clone https://github.com/drewOrc/cost-aware-hybrid-router.git
cd cost-aware-hybrid-router
pip install -r requirements.txt

# 下載 CLINC150
python download_data.py

# 執行零成本路由器（R1、R2、R4 無 LLM、SetFit）
PYTHONPATH=. python3 src/evaluate.py --no-llm

# 執行 LLM 評估（需要 Anthropic API key，總計約 $0.44）
export ANTHROPIC_API_KEY=sk-ant-...
PYTHONPATH=. python3 src/evaluate_llm_parallel.py \
  --llm-n 400 --seeds 42 43 44 --workers 6 --use-tuned-thresholds

# 合併種子 → 平均 ± 標準差、Wilson CI、McNemar
PYTHONPATH=. python3 src/merge_seeds.py

# 產生論文等級圖表
python paper/figures.py
```

---

## 統計驗證

- **Wilson 95% 信賴區間** — R3: [80.7%, 84.9%]，R4+LLM: [80.3%, 84.6%]（重疊）
- **McNemar 精確二項檢定** — 0/3 種子在 α=0.05 下顯著
- **配對評估** — R3 和 R4 在每個種子中評估相同的查詢
- **閾值僅在驗證集上調整** — 無測試集洩漏

---

## 論文

Workshop 論文草稿及出版品質圖表位於 [`paper/`](paper/)：

- [`Cost-Aware_Hybrid_Routing_Paper.pdf`](paper/Cost-Aware_Hybrid_Routing_Paper.pdf) — 可提交 PDF
- [`draft_v0.md`](paper/draft_v0.md) — Markdown 原始稿（含內嵌圖表）
- [`figures.py`](paper/figures.py) — 可重現圖表產生器（4 張圖，300 DPI）

![Per-agent heatmap](paper/figures/F3_per_agent_heatmap.png)

---

## 資料夾結構

```
cost-aware-hybrid-router/
├── README.md / README_zh.md
├── LICENSE (MIT)
├── requirements.txt
├── download_data.py
├── data/
│   ├── clinc150/              ← 原始資料（gitignore；執行 download_data.py）
│   └── intent_to_agent.json   ← 150 意圖 → 7 agent + OOS
├── src/
│   ├── routers/               ← R1 關鍵詞、R2 嵌入向量、R3 LLM、R4 級聯、SetFit
│   ├── tune.py                ← 驗證集閾值網格搜索
│   ├── evaluate.py            ← 零成本路由器評估
│   ├── evaluate_llm_parallel.py ← LLM + 級聯（平行、多種子）
│   ├── merge_seeds.py         ← 聚合 → 平均 ± 標準差、Wilson CI、McNemar
│   └── stats.py               ← Wilson CI + McNemar 精確二項檢定
├── results/
│   ├── tuned_thresholds.json
│   ├── metrics_seed{42,43,44}.json
│   └── metrics_merged.json    ← 主要結果檔案
└── paper/
    ├── Cost-Aware_Hybrid_Routing_Paper.docx/pdf
    ├── draft_v0.md
    ├── figures.py
    └── figures/F1-F4.png
```

---

## 限制

1. **單一基準測試**（CLINC150）。尚未在 BANKING77、HWU64 或非英語資料集上驗證泛化能力。
2. **LLM 上限約 83%。** 級聯匹配但無法超越 Haiku 零樣本表現。微調或少樣本 LLM 會提高上限。
3. **靜態閾值。** 無線上適應或個人化機制。
4. **OOS 偵測脆弱。** 依賴關鍵詞失敗時預設為 OOS；校準過的 OOS 偵測器會更穩健。
5. **閾值敏感度未分析。** 完整 `(kt, et)` 網格上的成本/準確率曲線未報告。

---

## 可重現性

- 固定種子：42, 43, 44
- 所有 LLM 呼叫溫度 = 0
- 閾值僅在驗證集上調整
- 所有依賴套件已釘選於 `requirements.txt`
- 重現總 API 成本：**$0.44**

---

## 引用

```bibtex
@misc{chen2026hybrid,
  author = {Chen, Bo-Yu},
  title  = {Cost-Aware Hybrid Router for LLM Agent Systems},
  year   = {2026},
  url    = {https://github.com/drewOrc/cost-aware-hybrid-router}
}
```

---

## 授權

MIT — 見 [LICENSE](LICENSE)。

**Bo-Yu Chen** — University of Texas at San Antonio — [GitHub](https://github.com/drewOrc)
