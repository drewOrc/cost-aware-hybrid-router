"""
train_setfit.py — 訓練 SetFit agent classifier (baseline)

跑法：
    cd cost-aware-hybrid-router/
    PYTHONPATH=. python3 src/train_setfit.py

輸出：
    models/setfit-agent-classifier/  ← SetFit checkpoint
    models/setfit-agent-classifier/labels.json

說明：
- 用 CLINC150 train split 中每個 agent 的 N 筆 few-shot 範例做對比學習
- Backbone: sentence-transformers/paraphrase-MiniLM-L3-v2 (~60MB, CPU-friendly)
- N_PER_AGENT=16 是 SetFit 論文推薦的 few-shot 規模，已足夠達到強 baseline
- 訓練時間: Mac M1/M2 約 2-5 分鐘
"""

import json
import random
from pathlib import Path

# ─────────────────────────────────────────────
# 參數
# ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "setfit-agent-classifier"
MODEL_DIR.parent.mkdir(exist_ok=True)

N_PER_AGENT = 16          # few-shot 每 agent 筆數
BACKBONE = "sentence-transformers/paraphrase-MiniLM-L3-v2"
SEED = 42
NUM_EPOCHS = 1
BATCH_SIZE = 16


def main():
    # 依賴檢查
    try:
        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "需要安裝: pip install setfit sentence-transformers datasets"
        ) from e

    # ── 載入資料 ──
    with open(DATA_DIR / "clinc150" / "train.json") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "clinc150" / "validation.json") as f:
        val_data = json.load(f)
    with open(DATA_DIR / "clinc150" / "intent_names.json") as f:
        intent_names = json.load(f)
    with open(DATA_DIR / "intent_to_agent.json") as f:
        raw = json.load(f)
        mapping = {k: v for k, v in raw.items() if k != "_meta"}

    def to_agent(item):
        return mapping[intent_names[item["intent"]]]

    # ── Few-shot sampling: 每 agent N_PER_AGENT 筆 ──
    rng = random.Random(SEED)
    by_agent: dict[str, list[str]] = {}
    for item in train_data:
        agent = to_agent(item)
        by_agent.setdefault(agent, []).append(item["text"])

    labels = sorted(by_agent.keys())
    label2id = {lbl: i for i, lbl in enumerate(labels)}

    train_texts = []
    train_labels = []
    for agent, texts in by_agent.items():
        sampled = rng.sample(texts, min(N_PER_AGENT, len(texts)))
        train_texts.extend(sampled)
        train_labels.extend([label2id[agent]] * len(sampled))

    # Validation set (完整 val split)
    val_texts = [item["text"] for item in val_data]
    val_labels = [label2id[to_agent(item)] for item in val_data]

    print(f"Training set: {len(train_texts)} examples ({N_PER_AGENT}/agent × {len(labels)} agents)")
    print(f"Val set: {len(val_texts)} examples")
    print(f"Labels: {labels}\n")

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

    # ── 訓練 ──
    print(f"Loading backbone: {BACKBONE}")
    model = SetFitModel.from_pretrained(BACKBONE, labels=labels)

    args = TrainingArguments(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Training SetFit (contrastive phase + classifier head)...")
    trainer.train()

    # ── 評估（手動，避免依賴 evaluate package） ──
    preds = model.predict(val_texts)
    # SetFitModel.predict returns label strings when labels= was passed
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    pred_ids = [label2id[p] if isinstance(p, str) else int(p) for p in preds]
    correct = sum(1 for p, y in zip(pred_ids, val_labels) if p == y)
    val_acc = correct / len(val_labels)
    print(f"\nValidation accuracy: {val_acc*100:.2f}% ({correct}/{len(val_labels)})")

    # ── 存檔 ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    with open(MODEL_DIR / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\n✅ Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
