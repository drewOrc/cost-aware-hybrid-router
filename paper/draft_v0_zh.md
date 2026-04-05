# 成本感知混合路由在意圖分類上的應用：在維持相同準確率下降低 74% LLM 成本

> 「Cost-Aware Cascade」系列 Paper 1 的中文版。英文版見 `draft_v0.md`。
> 這份中文版用來幫助用中文思考論點和結構，實際投稿用英文版。
> Workshop 目標：NLP4ConvAI / EMNLP Industry / arXiv preprint。
> 預期長度：4 頁 short paper + references。
> 狀態：skeleton v0，2026-04-05。所有數字來自 `results/metrics_merged.json`。

---

## Abstract（中文版，約 200 字）

在多代理 LLM 系統中，把用戶輸入路由到正確的 task agent 是一個基礎步驟。目前業界的預設做法是用 LLM 作為路由器，因為準確率高。但我們要問：這個準確率是否值這個價？

在 CLINC150（150 個 intent 映射到 7 個 agent + OOS）上，我們比較四個成本範圍涵蓋 3 個數量級的路由器：(R1) keyword、(R2) embedding nearest-centroid、(R3) Claude Haiku 4.5 LLM-only、(R4) 以 R1/R2 作為 confidence-gated 前置過濾、只在低信心時升級到 R3 的**混合 cascade**。

跨 3 個 seed × 400 個 pooled queries 的實驗結果：R4 達到 **82.6% ± 1.2pp** 的準確率 vs R3 的 **82.9% ± 0.6pp** — McNemar 配對檢定顯示統計上無顯著差異（p > 0.37）— 同時只有 **26.1%** 的 query 升級到 LLM，帶來 **74.3% 的 LLM 成本降低**。我們開源完整程式碼、tuned thresholds 和 per-seed trajectory。

---

## 1. Introduction（約 0.75 頁）

### 問題切入點
在生產環境的多代理 LLM 系統裡，每個用戶的每一句話都會先經過一個**路由器（router）**：這個元件決定下游哪個 agent（finance、HR、travel……）來處理這個請求。業界預設做法是用 LLM 當路由器 —— 一個 Haiku 等級的模型，prompt 裡放 agent 清單和「用 JSON 回答」的指令。這種做法方便，但**很貴**：規模化時，每一個用戶 turn 都要先呼叫一次 LLM API，才能進入真正的工作。

### 核心問題
這個 LLM 呼叫值不值？keyword 比對或 embedding 查找便宜 3-4 個數量級。如果便宜方法可以**有信心地**處理簡單 query，我們就只需要在困難 query 上用 LLM。這篇論文量化 CLINC150 裡有多少路由工作是「簡單的」，並測量 confidence-gated cascade 的準確率/成本 trade-off。

### 貢獻
1. **同一 benchmark 上的四個路由器**：keyword (R1)、embedding (R2)、LLM (R3)、hybrid cascade (R4)，在相同 protocol 下 tune 和 evaluate。
2. **3-seed paper-grade 評估**：每個 seed 400 query 分層抽樣（seed 42/43/44），包含 McNemar 配對檢定和 Wilson 95% CI。
3. **實證發現**：R4 與 R3 在 26% 的 LLM 呼叫率下準確率相差 0.4pp 以內 → 74% 成本降低。差異在 α=0.05 下不具統計顯著性。
4. **開源**：程式碼、tuned thresholds、per-seed trajectory、LLM 呼叫紀錄（`github.com/drewOrc/cost-aware-hybrid-router`，MIT）。

### 全文架構
§2 在意圖分類和成本 cascade 文獻中定位本工作。§3 描述四個路由器和 tuning protocol。§4 呈現 3-seed 結果和顯著性檢定。§5 討論 cascade 在什麼情況下 work / 不 work、limitation、以及延伸到 multi-turn agent task 的方向（我們的 companion work 在 τ-bench 上）。

---

## 2. Related Work（約 0.5 頁）

### CLINC150 上的意圖分類
CLINC150（Larson et al. 2019）有 150 個 in-domain intent + 一個 OOS class，原本被提出來 stress-test 分類器在 distribution shift 下的表現。現代 baseline 包括：SetFit（Tunstall et al. 2022，few-shot contrastive）、fine-tuned DeBERTa、LLM zero-shot prompting。我們用 SetFit 當強 baseline（我們的重製結果：**70.2%**）和 LLM（Haiku 4.5）當成本天花板。

### 成本感知 LLM cascade
FrugalGPT（Chen et al. 2023）提出 cascade 模式 —— 試便宜模型，低信心就升級 —— 在 single-turn Q&A 上報告最多 98% 成本降低、準確率維持。我們把這個模式移植到**路由**這個更窄但頻率極高的 task，因為成本降低會在 agent turn 之間**乘法式累積**。

### Agent 路由系統
大部分多代理框架（CrewAI、LangGraph、AutoGen）預設用 LLM supervisor 做路由。我們不是要反對 LLM 路由器，而是要主張：**在 LLM 看到 query 之前，先用便宜方法過濾掉簡單的**。

### 定位
我們的貢獻不是新的路由演算法。我們的貢獻是一個實證問題：**CLINC150 上有多少比例的路由決策可以靠 keyword 或 embedding 處理？代價是多少準確率？** 這個答案影響的是實際的生產選擇。

---

## 3. Method（約 1 頁）

### 3.1 Dataset 與 Label
我們用 CLINC150 的 test split（總共 5,500 個 query）。150 個 intent 被映射到 7 個 task agent + 一個 OOS class（auto、device、finance、kitchen、meta、productivity、travel、oos），給路由器一個符合現實的多領域 agent inventory。每個 class 的大小從 360 到 1,140 不等。

### 3.2 四個路由器
**R1 — Keyword 路由器。** 每個 agent 有一份 keyword list，從訓練集 intent 名稱 + 常見同義詞抽出來。Score = 最大 tf-idf overlap。固定不 tune。

**R2 — Embedding 路由器。** 每個 agent 有一個 centroid，用 MPNet-base（all-mpnet-base-v2，768 維）。Score = 到最近 centroid 的 cosine similarity。訓練時就固定。

**R3 — LLM 路由器。** Claude Haiku 4.5，system prompt 列出 8 個 label 和 JSON 輸出指令。Temperature 0。每個 query 一次呼叫。

**R4 — 混合 Cascade。** 兩個 threshold (kt, et) 作用在 R1/R2 score 上。決策規則：
```
if R1_score ≥ kt:          回傳 R1_pred
elif R2_score ≥ et:        回傳 R2_pred
else:                      呼叫 R3(LLM) → 回傳 R3_pred
```
Threshold 在 held-out validation split 上做 grid search tune（kt ∈ {0.5, 1.0, 1.5, 2.0}，et ∈ {0.05, 0.08, 0.10, 0.15}），目標函數是 **accuracy − 0.1·LLM_call_rate**，用這個目標函數偏好「在相同準確率下用便宜呼叫替代 LLM 呼叫」的配置。最終 tuned 值：`kt=0.5, et=0.10`。

### 3.3 Evaluation Protocol
每個 seed 做 n=400 的分層抽樣（每 class 50 × 8 class）來控制 LLM 花費。3 個 seed：42、43、44。報告 per-seed 和 pooled accuracy、Wilson 95% confidence interval（pooled n=1,200）、以及 R3 vs R4 在**同一批 query** 上的 McNemar 配對檢定。

### 3.4 成本計算
LLM 成本從實際 Anthropic API token 使用量計算。R1/R2 成本 amortize 到 ~$0（本地計算）。整個 3-seed 實驗的總 API 花費：**$0.44**（R3 $0.35，R4 $0.09）。

---

## 4. Results（約 1.25 頁）

### 4.1 主要結果表

| Router | Accuracy (pooled) | Wilson 95% CI | LLM Call Rate | Cost/query |
|---|---|---|---|---|
| R1（keyword，full test n=5,500） | 64.8% | — | 0% | $0 |
| R2（embedding，full test n=5,500） | 74.0% | — | 0% | $0 |
| SetFit baseline（full test n=5,500） | 70.2% | — | 0% | $0 |
| R4 no-LLM（R1→R2 only，n=5,500） | 74.9% | — | 0% | $0 |
| **R3（LLM，3 seeds pooled n=1,200）** | **82.9%** | [80.7, 84.9] | 100% | $9.8×10⁻⁵ |
| **R4 hybrid with LLM（同一批 n=1,200）** | **82.6%** | [80.3, 84.6] | 26.1% | $2.5×10⁻⁵ |

**閱讀方式**：R3 和 R4+LLM 的 95% CI 重疊。完整 no-LLM baseline R4-no-LLM 只有 75%，所以 LLM 升級**確實在做事** —— 只是只在 26% 的 query 上做，而不是 100%。

### 4.2 統計顯著性（McNemar）

| Seed | Δ accuracy (R4 − R3) | McNemar p-value | 在 α=0.05 下顯著? |
|---|---|---|---|
| 42 | +0.00pp | 1.000 | 否 |
| 43 | −1.75pp | 0.371 | 否 |
| 44 | +0.75pp | 0.755 | 否 |

跨 3 個 seed，R4 和 R3 在統計上無法區分。

### 4.3 成本節省
Per-seed R3 成本：$0.117（pooled 400 次 LLM 呼叫，Haiku-4.5 定價）。
Per-seed R4+LLM 成本：$0.030（pooled ~105 次 LLM 呼叫）。
**LLM 成本降低：74.3%。**

### 4.4 R1/R2 在哪裡成功 vs 需要升級
*（待補：各 agent 的 confusion — 哪些 agent R1 能清楚處理（finance 單獨用 R1 就 83.6%），哪些需要 LLM（meta_agent、device_agent）。）*
*（待補：各 ground-truth class 的升級率。）*

### 4.5 圖表計畫
- **F1**：Accuracy vs LLM call rate Pareto（tuning grid 的 scatter + 最終 R3/R4 點）。
- **F2**：per-seed accuracy bar + error bar，R1/R2/R3/R4 並列。
- **F3**：per-agent accuracy heatmap（router × agent）。
- **F4**：per query 成本，log-scale bar（R1 ≈ $0、R2 ≈ $0、R3 $9.8e-5、R4 $2.5e-5）。

---

## 5. Discussion（約 0.5 頁）

### Cascade 在什麼時候有效？
當 query 分佈是**雙峰（bimodal）**時，cascade 會 work：大量簡單 query 可以從表面特徵回答，加上一個需要 LLM 的困難 tail。CLINC150 是 bimodal 的 —— 領域層級的 keyword（finance 術語、travel 動詞）帶大部分訊號，而 meta-intent 和跨領域 query 是 ambiguous 的。難度平坦（所有 query 一樣難）的 benchmark 就看不出好處。

### 什麼時候會失敗？
如果便宜路由器的信心校準（confidence calibration）不好，hybrid **會在不省成本的情況下掉準確率**。我們的 tuning grid 用 `accuracy − 0.1·LLM_call_rate` 懲罰這種情況。在生產環境裡，這個旋鈕應該根據實際 LLM 成本來設。

### Limitation
(1) 單一 benchmark（CLINC150）。(2) 固定 agent inventory（8 class）。(3) 靜態 threshold，沒有 online adaptation。(4) Haiku 4.5 本身已經便宜了，如果和 GPT-4 比，節省會更多。

### 延伸：multi-turn agent task
Cascade 模式自然可以延伸到 **multi-turn** agent 場景：不是每個 user turn 都做路由，而是用便宜模型做 task **decomposition**，把昂貴模型留給 **execution**。我們在 companion work 裡用 τ-bench 驗證這個假設（Paper 2）。

---

## 6. Conclusion（約 0.2 頁）

在 CLINC150 上，confidence-gated hybrid cascade 以 0.4pp 的準確率差距追平 LLM-only 路由器，同時降低 74% 的 LLM 呼叫。選擇不是「LLM-as-router 還是不用」，而是「LLM-as-router 要用在**哪些** query 上」。我們開源完整 pipeline，作為生產團隊做這個成本/準確率 trade-off 的模板。

---

## Reproducibility Statement

程式碼、tuned thresholds、per-seed trajectory、LLM 呼叫紀錄：`github.com/drewOrc/cost-aware-hybrid-router`（MIT）。Model 版本：`claude-haiku-4-5-20251001`。Seeds：42、43、44。Embedding 模型：`sentence-transformers/all-mpnet-base-v2`。完整重製的 API 成本：**$0.44**。

---

## 投稿前 checklist

- [ ] 從 `metrics_merged.json` 再跑一次數字確認沒 drift
- [ ] 用 `matplotlib` 生成 F1-F4 圖（script 放在 `paper/figures.py`）
- [ ] 把具體數字寫進 §4.4（confusion + 各 class 升級率）
- [ ] 在 §4.4 加 1-2 個 qualitative Table 2 example（R1-only / R2-only / 升級到 R3 各挑一個）
- [ ] 再檢查 Wilson CI 公式和 McNemar exact-binomial p-value
- [ ] Citation 檢查：Larson 2019、Tunstall 2022、Chen 2023 (FrugalGPT)、anthropic pricing page
- [ ] Limitation section 至少 3 項，明寫
- [ ] Ethics statement（無人類受試者，無 PII）

---

## 給自己的筆記

- **Tone**：精簡、以數據為主、不要形容詞。reviewer 應該要能在 2 分鐘內從 repo 重現 headline table。
- **Companion τ-bench paper (Paper 2)** 等有 preprint ID 就要在這裡引用為 "companion work"；目前先放 footnote。
- 這篇 paper 可以獨立站立（headline 數字成立），但**整個 thesis 只有在 Paper 2 證明 pattern 可以延伸到 multi-turn agent 時才完整落地**。conclusion 要這樣 frame。

---

## 中文版 writing tips（寫英文正式稿時參考）

- 中文版是**思考工具**，不是翻譯目標。英文正式稿要保持學術簡潔，不要把中文這邊的口語直接譯過去。
- 中文版用來 check 論點邏輯：如果一段論述中文講不清楚，英文也一定講不清楚。
- 術語對照：
  - 路由器 → router
  - 混合 cascade → hybrid cascade
  - 信心門檻 → confidence threshold
  - 升級（到 LLM）→ escalate (to LLM)
  - 成本感知 → cost-aware
  - 前置過濾 → pre-filter
  - 配對檢定 → paired test
