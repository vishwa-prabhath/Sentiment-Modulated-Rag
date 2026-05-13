"""
Evaluation 4: End-to-End Response Quality (BLEU + ROUGE + Grounding)
=====================================================================
Uses the 41 Q&A pairs as reference answers and measures how closely
the generated responses match using BLEU and ROUGE-L scores.
Also measures the factual grounding rate.
"""
import os, json, time, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests
from collections import Counter

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"

# Load ground truth
with open("nsbm_rag_dataset.json") as f:
    qa_data = json.load(f)

# ---- BLEU Score (simplified, no external deps) ----
def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def bleu_score(reference, hypothesis, max_n=4):
    """Compute BLEU score (simplified) between reference and hypothesis."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ngrams(ref_tokens, n))
        hyp_ngrams = Counter(ngrams(hyp_tokens, n))

        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())

        if total == 0:
            scores.append(0)
        else:
            scores.append(clipped / total)

    # Geometric mean
    import math
    if any(s == 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / len(scores)

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))

    return round(bp * math.exp(log_avg), 4)


# ---- ROUGE-L Score ----
def lcs_length(x, y):
    """Longest Common Subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(reference, hypothesis):
    """Compute ROUGE-L F1 score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = lcs_length(ref_tokens[:200], hyp_tokens[:200])  # Limit for performance
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0
    recall = lcs / len(ref_tokens) if ref_tokens else 0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


# ---- Factual Grounding Check ----
def check_grounding(reference, response):
    """Check if key facts from reference appear in the response."""
    ref_words = set(reference.lower().split())
    resp_words = set(response.lower().split())
    # Remove common stop words
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
                 "of", "and", "or", "but", "not", "with", "by", "from", "that", "this", "it",
                 "be", "has", "have", "had", "do", "does", "did", "will", "would", "can", "could"}
    ref_content = ref_words - stopwords
    resp_content = resp_words - stopwords

    if not ref_content:
        return 0.0
    overlap = len(ref_content & resp_content) / len(ref_content)
    return round(overlap, 4)


# ---- Use Unified Pipeline ----
from unified_system import generate_unified_response


# ---- Run evaluation ----
# Use a subset (every 2nd question) to keep runtime manageable
eval_subset = qa_data[::2]  # ~20 queries
print(f"{'='*55}")
print(f"  END-TO-END QUALITY EVALUATION")
print(f"  Evaluating {len(eval_subset)} queries (subset of 41)")
print(f"{'='*55}\n")

results = []
bleu_scores = []
rouge_scores = []
grounding_scores = []

for i, item in enumerate(eval_subset, 1):
    question = item["question"]
    reference = item["answer"]
    category = item["category"]

    print(f"[{i}/{len(eval_subset)}] {question[:65]}...", flush=True)

    start = time.time()
    response, task_type, emotion, confidence, hyperparams = generate_unified_response(question)
    elapsed = time.time() - start

    # Calculate metrics
    b = bleu_score(reference, response)
    r = rouge_l(reference, response)
    g = check_grounding(reference, response)

    bleu_scores.append(b)
    rouge_scores.append(r)
    grounding_scores.append(g)

    results.append({
        "question": question[:80],
        "category": category,
        "bleu": b,
        "rouge_l": r,
        "grounding": g,
        "response_preview": response[:200],
        "time_seconds": round(elapsed, 1)
    })

    print(f"  BLEU={b:.4f}  ROUGE-L={r:.4f}  Grounding={g:.4f}  ({elapsed:.1f}s)")

# Calculate averages
n = len(results)
output = {
    "total_queries": n,
    "avg_bleu": round(sum(bleu_scores) / n, 4),
    "avg_rouge_l": round(sum(rouge_scores) / n, 4),
    "avg_grounding": round(sum(grounding_scores) / n, 4),
    "grounding_rate_above_30pct": round(sum(1 for g in grounding_scores if g >= 0.3) / n * 100, 1),
    "per_category": {},
    "detailed_results": results
}

# Per-category averages
cat_scores = {}
for r in results:
    cat = r["category"]
    if cat not in cat_scores:
        cat_scores[cat] = {"bleu": [], "rouge": [], "grounding": []}
    cat_scores[cat]["bleu"].append(r["bleu"])
    cat_scores[cat]["rouge"].append(r["rouge_l"])
    cat_scores[cat]["grounding"].append(r["grounding"])

for cat, scores in cat_scores.items():
    n_cat = len(scores["bleu"])
    output["per_category"][cat] = {
        "avg_bleu": round(sum(scores["bleu"]) / n_cat, 4),
        "avg_rouge_l": round(sum(scores["rouge"]) / n_cat, 4),
        "avg_grounding": round(sum(scores["grounding"]) / n_cat, 4),
        "count": n_cat
    }

with open("eval_results/quality_metrics.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*55}")
print(f"  QUALITY EVALUATION RESULTS")
print(f"{'='*55}")
print(f"  Average BLEU:         {output['avg_bleu']}")
print(f"  Average ROUGE-L:      {output['avg_rouge_l']}")
print(f"  Average Grounding:    {output['avg_grounding']}")
print(f"  Grounding Rate (>30%): {output['grounding_rate_above_30pct']}%")
print(f"\nSaved to eval_results/quality_metrics.json")
