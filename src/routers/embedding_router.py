"""
Embedding Router (R2) — TF-IDF + Cosine Similarity

設計原則：
- 用 CLINC150 train set 建立每個 agent 的 TF-IDF centroid
- 新 query 轉 TF-IDF vector 後，與各 agent centroid 做 cosine similarity
- 取最高 similarity 的 agent
- 零 LLM call，延遲 ~ms 級

技術選擇：
- 用 scikit-learn TF-IDF 而非 sentence-transformers
  原因：(1) 磁碟空間限制不允許安裝 PyTorch (2) TF-IDF 在 intent classification
  任務上已有不錯表現 (3) 可重現性更高（無 model checkpoint 依賴）
- 如果需要更強的 embedding，可以之後在有 GPU 的環境換成 SBERT

研究意義：
- 代表「中等成本」的路由策略：比 keyword 準但比 LLM 便宜
- 與 keyword router 比較可看出「語意理解」帶來多少提升
"""

import json
import os
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# 資料路徑
# ─────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_TRAIN_PATH = _DATA_DIR / "clinc150" / "train.json"
_INTENT_NAMES_PATH = _DATA_DIR / "clinc150" / "intent_names.json"
_MAPPING_PATH = _DATA_DIR / "intent_to_agent.json"


# ─────────────────────────────────────────────
# 模型（lazy init，第一次 route() 時建立）
# ─────────────────────────────────────────────

_vectorizer: TfidfVectorizer = None
_agent_centroids: dict[str, np.ndarray] = None
_agent_names: list[str] = None
_centroid_matrix: np.ndarray = None  # shape: (n_agents, n_features)


def _load_and_build():
    """用 train set 建立 TF-IDF vectorizer 和 agent centroid"""
    global _vectorizer, _agent_centroids, _agent_names, _centroid_matrix

    # 載入資料
    with open(_TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(_INTENT_NAMES_PATH, encoding="utf-8") as f:
        intent_names = json.load(f)
    with open(_MAPPING_PATH, encoding="utf-8") as f:
        raw_mapping = json.load(f)
        mapping = {k: v for k, v in raw_mapping.items() if k != "_meta"}

    # 按 agent 分組收集 train queries
    agent_queries: dict[str, list[str]] = {}
    for item in train_data:
        intent_name = intent_names[item["intent"]]
        agent = mapping[intent_name]
        if agent not in agent_queries:
            agent_queries[agent] = []
        agent_queries[agent].append(item["text"])

    # 建立 TF-IDF（用所有 train queries）
    all_queries = []
    query_agents = []
    for agent, queries in agent_queries.items():
        all_queries.extend(queries)
        query_agents.extend([agent] * len(queries))

    _vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),     # unigram + bigram
        sublinear_tf=True,      # 用 1 + log(tf) 而非 raw tf
        min_df=2,
        stop_words="english",
    )
    tfidf_matrix = _vectorizer.fit_transform(all_queries)

    # 計算每個 agent 的 centroid（取該 agent 所有 query 的平均 TF-IDF vector）
    _agent_names = sorted(agent_queries.keys())
    centroids = []
    for agent in _agent_names:
        indices = [i for i, a in enumerate(query_agents) if a == agent]
        agent_vectors = tfidf_matrix[indices]
        centroid = agent_vectors.mean(axis=0)
        centroids.append(np.asarray(centroid).flatten())

    _centroid_matrix = np.array(centroids)  # (n_agents, n_features)
    _agent_centroids = {name: centroids[i] for i, name in enumerate(_agent_names)}


# ─────────────────────────────────────────────
# Router 實作
# ─────────────────────────────────────────────

def route(query: str) -> dict:
    """
    用 TF-IDF cosine similarity 路由 query 到某個 agent。

    Returns:
        {
            "agent": str,
            "confidence": float,    # cosine similarity（0-1）
            "scores": dict,         # 各 agent 的 similarity
            "method": "embedding"
        }
    """
    global _vectorizer, _centroid_matrix, _agent_names

    # Lazy init
    if _vectorizer is None:
        _load_and_build()

    # 轉換 query
    q_vec = _vectorizer.transform([query.lower().strip()])

    # 與所有 agent centroid 做 cosine similarity
    similarities = cosine_similarity(q_vec, _centroid_matrix).flatten()

    scores = {
        _agent_names[i]: round(float(similarities[i]), 4)
        for i in range(len(_agent_names))
    }

    best_idx = int(np.argmax(similarities))
    best_agent = _agent_names[best_idx]
    best_score = float(similarities[best_idx])

    return {
        "agent": best_agent,
        "confidence": round(best_score, 4),
        "scores": scores,
        "method": "embedding",
    }


# ─────────────────────────────────────────────
# 直接執行時做簡單測試
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "what's my credit score",
        "book a flight to tokyo",
        "when should i get my oil changed",
        "how many calories in an avocado",
        "set a reminder for 3pm",
        "play some jazz music",
        "tell me a joke",
        "i need to transfer money to my savings",
        "what's the weather like tomorrow",
        "do i need a visa for japan",
        "asdfghjkl random gibberish xyz",
    ]

    print("Building TF-IDF model from train set...")
    for q in test_queries:
        r = route(q)
        print(f"  [{r['agent']:20s}] (conf={r['confidence']:.3f})  {q}")
