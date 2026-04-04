"""
SetFit Router (Baseline) — Few-shot fine-tuned sentence transformer

作為 "paper-grade baseline" 對照：比 TF-IDF (R2) 強、比 LLM (R3) 便宜。
SetFit 用對比學習 (contrastive learning) 微調 sentence-transformer，
在 intent classification 上是 NLP 社群 2024-2025 的標準 strong baseline。

使用方式：
    # 訓練 (一次就好，會存 checkpoint 到 models/setfit-agent-classifier/)
    python src/train_setfit.py

    # 推論
    from src.routers import setfit_router
    setfit_router.route("what's my credit score")
    → {"agent": "finance", "confidence": 0.87, "method": "setfit"}

設計：
- 用 CLINC150 train set + intent→agent mapping 訓練
- Backbone: sentence-transformers/paraphrase-MiniLM-L3-v2（小而快）
- Few-shot: 每 agent 取 N 筆（default N=16）做對比學習，其餘當 validation
- 輸出: 7 agents + oos 的分類機率

研究意義：
- 若 R4+LLM > SetFit，證明 LLM fallback 在高不確定性 query 上仍有價值
- 若 SetFit > R4+LLM，則 cost-aware cascade 可能被 "single strong cheap model" 取代
- 無論結果如何，這都是 paper 必要的對照
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "setfit-agent-classifier"
_LABEL_PATH = _MODEL_DIR / "labels.json"

_model = None
_labels: Optional[list[str]] = None


def _load():
    """Lazy-load SetFit model from disk."""
    global _model, _labels
    if _model is not None:
        return

    if not _MODEL_DIR.exists():
        raise RuntimeError(
            f"SetFit model not found at {_MODEL_DIR}. "
            f"Train it first: `python src/train_setfit.py`"
        )

    try:
        from setfit import SetFitModel
    except ImportError as e:
        raise ImportError(
            "setfit package not installed. Run: pip install setfit sentence-transformers"
        ) from e

    _model = SetFitModel.from_pretrained(str(_MODEL_DIR))
    with open(_LABEL_PATH, encoding="utf-8") as f:
        _labels = json.load(f)


def route(query: str) -> dict:
    """
    用 SetFit 分類 query 到某個 agent。

    Returns:
        {
            "agent": str,
            "confidence": float,    # softmax probability of top class
            "scores": dict,          # 各 agent 的 probability
            "method": "setfit"
        }
    """
    _load()

    # SetFit returns label index; use predict_proba for confidence
    probs = _model.predict_proba([query.lower().strip()])
    # probs is a torch tensor or numpy array, shape (1, n_classes)
    import numpy as np
    probs = np.asarray(probs).flatten()

    best_idx = int(np.argmax(probs))
    best_label = _labels[best_idx]
    best_score = float(probs[best_idx])

    scores = {
        _labels[i]: round(float(probs[i]), 4)
        for i in range(len(_labels))
    }

    return {
        "agent": best_label,
        "confidence": round(best_score, 4),
        "scores": scores,
        "method": "setfit",
    }


if __name__ == "__main__":
    test_queries = [
        "what's my credit score",
        "book a flight to tokyo",
        "when should i get my oil changed",
        "tell me a joke",
        "asdfghjkl random gibberish xyz",
    ]
    print("Loading SetFit model...")
    for q in test_queries:
        r = route(q)
        print(f"  [{r['agent']:20s}] (conf={r['confidence']:.3f})  {q}")
