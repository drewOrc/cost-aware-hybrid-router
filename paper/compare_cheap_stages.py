"""
compare_cheap_stages.py — W5: Alternative cheap-stage comparison

比較不同 R2 候選方案在 CLINC150 上的效果：
  1. TF-IDF nearest-centroid（目前 R2）
  2. all-MiniLM-L6-v2 sentence-transformer（384d, 22M params）
  3. all-mpnet-base-v2 sentence-transformer（768d, 109M params）

每個候選跑兩件事：
  A. Standalone R2 accuracy（full test set, n=5500）
  B. Cascade simulation（R1 → R2 → oracle-R3）用 tuned thresholds

跑法：
    cd cost-aware-hybrid-router/
    pip3 install sentence-transformers
    PYTHONPATH=. python3 paper/compare_cheap_stages.py

輸出：
    results/cheap_stage_comparison.json

知識點：
  sentence-transformers 是一個建立在 HuggingFace transformers 上的套件，
  專門做句子級別的 embedding。跟 TF-IDF 的差別在於：
  - TF-IDF 是 bag-of-words 模型，只看「哪些字出現過」+ 加權
  - Sentence-transformers 是 neural model，理解語意（"book a flight" ≈ "reserve a plane ticket"）
  - 代價是需要 GPU/CPU 跑推理，模型檔比較大（~100MB-400MB）
  MiniLM 是 Microsoft 的輕量模型（22M params），速度比 mpnet（109M）快 5 倍左右。
"""

import json
import time
import sys
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Paths ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT))
from src.routers import keyword_router


# ─── Load data ─────────────────────────────────────────
def load_clinc150():
    """Load train + test data with intent→agent mapping."""
    with open(DATA_DIR / "clinc150" / "train.json") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "clinc150" / "test.json") as f:
        test_data = json.load(f)
    with open(DATA_DIR / "clinc150" / "intent_names.json") as f:
        intent_names = json.load(f)
    with open(DATA_DIR / "intent_to_agent.json") as f:
        raw = json.load(f)
        mapping = {k: v for k, v in raw.items() if k != "_meta"}

    # Group train queries by agent
    train_by_agent = {}
    for item in train_data:
        intent = intent_names[item["intent"]]
        agent = mapping[intent]
        train_by_agent.setdefault(agent, []).append(item["text"])

    # Test set with labels
    test_items = []
    for item in test_data:
        intent = intent_names[item["intent"]]
        agent = mapping[intent]
        test_items.append({"text": item["text"], "agent": agent})

    return train_by_agent, test_items


# ─── R2 Candidate: TF-IDF ──────────────────────────────
class TfidfR2:
    """Current R2: TF-IDF nearest-centroid."""
    name = "TF-IDF"
    params = "10K features, 1-2gram, sublinear_tf"

    def __init__(self, train_by_agent):
        all_queries, query_agents = [], []
        for agent, queries in train_by_agent.items():
            all_queries.extend(queries)
            query_agents.extend([agent] * len(queries))

        self.vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2, stop_words="english"
        )
        tfidf_matrix = self.vectorizer.fit_transform(all_queries)

        self.agent_names = sorted(train_by_agent.keys())
        centroids = []
        for agent in self.agent_names:
            indices = [i for i, a in enumerate(query_agents) if a == agent]
            centroid = tfidf_matrix[indices].mean(axis=0)
            centroids.append(np.asarray(centroid).flatten())
        self.centroid_matrix = np.array(centroids)

    def route(self, query: str) -> dict:
        q_vec = self.vectorizer.transform([query.lower().strip()])
        sims = cosine_similarity(q_vec, self.centroid_matrix).flatten()
        best_idx = int(np.argmax(sims))
        return {
            "agent": self.agent_names[best_idx],
            "confidence": float(sims[best_idx]),
        }


# ─── R2 Candidate: Sentence-Transformer ────────────────
class SentenceTransformerR2:
    """Sentence-transformer nearest-centroid."""

    def __init__(self, model_name, train_by_agent):
        from sentence_transformers import SentenceTransformer

        self.name = model_name.split("/")[-1]
        self.model = SentenceTransformer(model_name)

        self.agent_names = sorted(train_by_agent.keys())
        centroids = []
        for agent in self.agent_names:
            queries = train_by_agent[agent]
            embeddings = self.model.encode(queries, show_progress_bar=False,
                                           batch_size=256, normalize_embeddings=True)
            centroid = embeddings.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # L2 normalize
            centroids.append(centroid)
        self.centroid_matrix = np.array(centroids)
        self.params = f"{self.centroid_matrix.shape[1]}d, {sum(p.numel() for p in self.model[0].auto_model.parameters()) / 1e6:.0f}M params"

    def route(self, query: str) -> dict:
        q_vec = self.model.encode([query], normalize_embeddings=True)
        sims = (q_vec @ self.centroid_matrix.T).flatten()
        best_idx = int(np.argmax(sims))
        return {
            "agent": self.agent_names[best_idx],
            "confidence": float(sims[best_idx]),
        }


# ─── Cascade simulation ───────────────────────────────
def simulate_cascade(test_items, r2_model, kt=0.5, et=0.10):
    """
    Simulate R1→R2→oracle-R3 cascade.
    Oracle-R3 = assumes LLM is correct (upper bound on cascade accuracy).

    知識點：
    Oracle analysis 是一種常見的實驗手法。
    我們不實際呼叫 LLM（太貴），而是假設 LLM 全部答對。
    這樣可以看出「如果 LLM 是完美的，不同 R2 在 cascade 裡的效果差多少」。
    實際差異 = oracle_acc × real_R3_accuracy_on_escalated_queries.
    """
    correct = 0
    r1_stops = 0
    r2_stops = 0
    llm_calls = 0

    for item in test_items:
        text, true_agent = item["text"], item["agent"]

        # Stage 1: R1 keyword
        r1 = keyword_router.route(text)
        if r1["confidence"] >= kt:
            r1_stops += 1
            if r1["agent"] == true_agent:
                correct += 1
            continue

        # Stage 2: R2 (candidate model)
        r2 = r2_model.route(text)
        if r2["confidence"] >= et:
            r2_stops += 1
            if r2["agent"] == true_agent:
                correct += 1
            continue

        # Stage 3: Oracle LLM (always correct)
        llm_calls += 1
        correct += 1

    n = len(test_items)
    return {
        "cascade_oracle_acc": round(correct / n * 100, 2),
        "r1_stop_rate": round(r1_stops / n * 100, 1),
        "r2_stop_rate": round(r2_stops / n * 100, 1),
        "llm_call_rate": round(llm_calls / n * 100, 1),
    }


# ─── Evaluate standalone R2 accuracy ───────────────────
def eval_standalone(test_items, r2_model):
    """Evaluate R2 model standalone on full test set."""
    correct = 0
    confidences = []
    t0 = time.time()
    for item in test_items:
        r = r2_model.route(item["text"])
        if r["agent"] == item["agent"]:
            correct += 1
        confidences.append(r["confidence"])
    elapsed = time.time() - t0

    n = len(test_items)
    return {
        "accuracy": round(correct / n * 100, 2),
        "mean_confidence": round(np.mean(confidences), 4),
        "median_confidence": round(float(np.median(confidences)), 4),
        "latency_per_query_ms": round(elapsed / n * 1000, 2),
        "total_time_s": round(elapsed, 1),
    }


# ─── Main ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("W5: Alternative Cheap-Stage Comparison")
    print("=" * 60)

    train_by_agent, test_items = load_clinc150()
    print(f"Train: {sum(len(v) for v in train_by_agent.values())} queries, "
          f"{len(train_by_agent)} agents")
    print(f"Test:  {len(test_items)} queries\n")

    # Load thresholds
    with open(RESULTS_DIR / "tuned_thresholds.json") as f:
        thresholds = json.load(f)
    kt = thresholds["with_llm"]["keyword_threshold"]
    et = thresholds["with_llm"]["embed_threshold"]
    print(f"Cascade thresholds: kt={kt}, et={et}\n")

    results = []

    # ── Model 1: TF-IDF (current R2) ──
    print("─" * 40)
    print("Building TF-IDF model...")
    tfidf = TfidfR2(train_by_agent)
    standalone = eval_standalone(test_items, tfidf)
    cascade = simulate_cascade(test_items, tfidf, kt, et)
    entry = {
        "model": "TF-IDF (current R2)",
        "params": tfidf.params,
        "standalone": standalone,
        "cascade": cascade,
    }
    results.append(entry)
    print(f"  TF-IDF standalone: {standalone['accuracy']}% "
          f"({standalone['latency_per_query_ms']}ms/q)")
    print(f"  TF-IDF cascade:   oracle_acc={cascade['cascade_oracle_acc']}%, "
          f"LLM_rate={cascade['llm_call_rate']}%")

    # ── Model 2 & 3: Sentence-Transformers ──
    st_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]

    try:
        import sentence_transformers
        print(f"\nsentence-transformers v{sentence_transformers.__version__} found")
    except ImportError:
        print("\n⚠ sentence-transformers not installed!")
        print("  Run: pip3 install sentence-transformers")
        print("  Skipping neural embedding models.\n")
        st_models = []

    for model_name in st_models:
        short_name = model_name.split("/")[-1]
        print(f"\n{'─' * 40}")
        print(f"Building {short_name} centroids...")
        t0 = time.time()
        st_model = SentenceTransformerR2(model_name, train_by_agent)
        build_time = time.time() - t0
        print(f"  Centroids built in {build_time:.1f}s ({st_model.params})")

        standalone = eval_standalone(test_items, st_model)
        cascade = simulate_cascade(test_items, st_model, kt, et)
        entry = {
            "model": short_name,
            "params": st_model.params,
            "standalone": standalone,
            "cascade": cascade,
        }
        results.append(entry)
        print(f"  {short_name} standalone: {standalone['accuracy']}% "
              f"({standalone['latency_per_query_ms']}ms/q)")
        print(f"  {short_name} cascade:   oracle_acc={cascade['cascade_oracle_acc']}%, "
              f"LLM_rate={cascade['llm_call_rate']}%")

    # ── Summary table ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<28} {'R2 Acc%':>8} {'ms/q':>6} {'Cascade':>8} {'LLM%':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<28} {r['standalone']['accuracy']:>7.1f}% "
              f"{r['standalone']['latency_per_query_ms']:>5.1f} "
              f"{r['cascade']['cascade_oracle_acc']:>7.1f}% "
              f"{r['cascade']['llm_call_rate']:>5.1f}%")

    # ── Save ──
    out_path = RESULTS_DIR / "cheap_stage_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"thresholds": {"kt": kt, "et": et}, "results": results},
                  f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
