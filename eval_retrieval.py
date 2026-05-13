"""
Evaluation 1: Retrieval Accuracy (RAG Component)
=================================================
Measures Hit Rate @K and Mean Reciprocal Rank (MRR) using the 41 Q&A
pairs as ground truth. For each question in the dataset, we check
whether the correct answer text appears in the top-K retrieved chunks.
"""
import os, json, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
import chromadb

# Load ground truth
with open("nsbm_rag_dataset.json") as f:
    qa_data = json.load(f)

# Connect to KB
client = chromadb.PersistentClient(path="./unified_chroma_store")
kb = client.get_or_create_collection(name="knowledge_base")

print(f"Evaluating retrieval on {len(qa_data)} Q&A pairs against {kb.count()} KB chunks...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

K_VALUES = [1, 3, 5]
results = {f"hit@{k}": 0 for k in K_VALUES}
mrr_sum = 0
per_category = {}

detailed_results = []

for i, item in enumerate(qa_data):
    question = item["question"]
    answer = item["answer"]
    category = item["category"]

    # Retrieve top-5
    query_emb = embedder.encode(question).tolist()
    retrieved = kb.query(query_embeddings=[query_emb], n_results=max(K_VALUES))

    docs = retrieved["documents"][0] if retrieved["documents"] else []

    # Check if the answer appears in retrieved chunks
    # Use substring matching since the exact Q&A pair is stored as a chunk
    hit_rank = None
    for rank, doc in enumerate(docs, 1):
        # Match if significant portion of the answer is in the retrieved chunk
        answer_words = set(answer.lower().split()[:15])  # First 15 words
        doc_words = set(doc.lower().split())
        overlap = len(answer_words & doc_words) / max(len(answer_words), 1)
        if overlap >= 0.6:  # 60% word overlap = hit
            hit_rank = rank
            break

    # Record hits at each K
    for k in K_VALUES:
        if hit_rank is not None and hit_rank <= k:
            results[f"hit@{k}"] += 1

    # MRR
    if hit_rank:
        mrr_sum += 1.0 / hit_rank

    # Per-category tracking
    if category not in per_category:
        per_category[category] = {"total": 0, "hit@3": 0}
    per_category[category]["total"] += 1
    if hit_rank and hit_rank <= 3:
        per_category[category]["hit@3"] += 1

    detailed_results.append({
        "question": question[:80],
        "category": category,
        "hit_rank": hit_rank,
        "retrieved_preview": docs[0][:100] if docs else "NONE"
    })

    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(qa_data)}...")

del embedder
gc.collect()

# Calculate final metrics
n = len(qa_data)
metrics = {
    "total_queries": n,
    "kb_chunks": kb.count(),
    "hit_rate_at_1": round(results["hit@1"] / n * 100, 1),
    "hit_rate_at_3": round(results["hit@3"] / n * 100, 1),
    "hit_rate_at_5": round(results["hit@5"] / n * 100, 1),
    "mrr": round(mrr_sum / n, 4),
    "per_category": {
        cat: round(v["hit@3"] / v["total"] * 100, 1)
        for cat, v in sorted(per_category.items())
    },
    "detailed_results": detailed_results
}

# Save
with open("eval_results/retrieval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'='*50}")
print(f"  RETRIEVAL EVALUATION RESULTS")
print(f"{'='*50}")
print(f"  Total Queries:    {n}")
print(f"  KB Chunks:        {kb.count()}")
print(f"  Hit Rate @1:      {metrics['hit_rate_at_1']}%")
print(f"  Hit Rate @3:      {metrics['hit_rate_at_3']}%")
print(f"  Hit Rate @5:      {metrics['hit_rate_at_5']}%")
print(f"  MRR:              {metrics['mrr']}")
print(f"\n  Per-Category Hit Rate @3:")
for cat, rate in metrics["per_category"].items():
    print(f"    {cat:25s} {rate}%")
print(f"\nSaved to eval_results/retrieval_metrics.json")
