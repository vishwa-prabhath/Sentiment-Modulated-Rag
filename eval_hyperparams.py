"""
Evaluation 3: Hyperparameter Effect Analysis (HAG Novelty)
===========================================================
Proves that dynamic hyperparameters actually change response characteristics.
Runs the same 5 queries at 3 different temperature settings and measures
lexical diversity, response length, and tone word frequencies.
"""
import os, json, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"

# 5 test queries that work across all task types
test_queries = [
    "What degree programs does the Faculty of Computing offer?",
    "Tell me about the departments in the computing faculty.",
    "What is the enrollment process at NSBM?",
    "Describe the academic support services available.",
    "What are the facilities at NSBM Green University?",
]

# The 3 hyperparameter profiles from the system
profiles = {
    "factual": {"temperature": 0.3, "top_p": 0.85, "top_k": 40, "repetition_penalty": 1.2},
    "emotional": {"temperature": 0.6, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1},
    "creative": {"temperature": 0.85, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.0},
}

# Simple context (same for all to isolate the effect of hyperparameters)
CONTEXT = """Faculty of Computing (FOC) provides world-class education and training in Computing and 
Information Technology, both at the undergraduate as well as postgraduate levels.
FOC offers University Grants Commission's approved degree programmes in multiple disciplines.
The Faculty of Computing offers a plethora of pathways and specializations for its undergraduates.
Department of Computer and Data Science, Department of Computer Security and Network Systems,
Department of Software Engineering & Information Systems."""

# Tone word lists for analysis
EMPATHETIC_WORDS = {"understand", "sorry", "help", "support", "concern", "worry", "feel", "care",
                    "encourage", "assist", "guide", "comfort", "assure", "hope", "wish"}
ENTHUSIASTIC_WORDS = {"amazing", "exciting", "great", "wonderful", "fantastic", "excellent",
                      "incredible", "awesome", "brilliant", "outstanding", "thrilling", "superb",
                      "remarkable", "extraordinary", "impressive"}
FORMAL_WORDS = {"therefore", "however", "furthermore", "additionally", "specifically",
                "accordingly", "consequently", "regarding", "pertaining", "pursuant"}


def generate(prompt, params):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "top_k": params["top_k"],
            "repeat_penalty": params["repetition_penalty"],
            "num_predict": 200,
            "seed": 42  # Fixed seed for reproducibility
        }
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


def analyze_response(text):
    words = text.lower().split()
    unique_words = set(words)
    total = len(words)
    unique = len(unique_words)

    return {
        "word_count": total,
        "unique_words": unique,
        "type_token_ratio": round(unique / max(total, 1), 4),
        "avg_word_length": round(sum(len(w) for w in words) / max(total, 1), 2),
        "sentence_count": text.count('.') + text.count('!') + text.count('?'),
        "empathetic_words": len(unique_words & EMPATHETIC_WORDS),
        "enthusiastic_words": len(unique_words & ENTHUSIASTIC_WORDS),
        "formal_words": len(unique_words & FORMAL_WORDS),
    }


print(f"{'='*55}")
print(f"  HYPERPARAMETER EFFECT ANALYSIS")
print(f"  Running {len(test_queries)} queries × {len(profiles)} profiles")
print(f"{'='*55}\n")

all_results = []
profile_aggregates = {p: {"word_count": [], "ttr": [], "empathetic": [], "enthusiastic": [], "formal": []}
                      for p in profiles}

for qi, query in enumerate(test_queries, 1):
    print(f"Query {qi}/{len(test_queries)}: {query[:60]}...")

    prompt = (
        f"You are a helpful university assistant for NSBM Green University. "
        f"Answer based on the context below.\n\n"
        f"Context:\n{CONTEXT}\n\n"
        f"Question: {query}\nAnswer:"
    )

    for profile_name, params in profiles.items():
        print(f"  Profile: {profile_name} (temp={params['temperature']})...", end=" ", flush=True)
        start = time.time()
        response = generate(prompt, params)
        elapsed = time.time() - start

        analysis = analyze_response(response)
        analysis["profile"] = profile_name
        analysis["query"] = query
        analysis["response_preview"] = response[:150]
        analysis["time_seconds"] = round(elapsed, 1)
        all_results.append(analysis)

        # Aggregate
        profile_aggregates[profile_name]["word_count"].append(analysis["word_count"])
        profile_aggregates[profile_name]["ttr"].append(analysis["type_token_ratio"])
        profile_aggregates[profile_name]["empathetic"].append(analysis["empathetic_words"])
        profile_aggregates[profile_name]["enthusiastic"].append(analysis["enthusiastic_words"])
        profile_aggregates[profile_name]["formal"].append(analysis["formal_words"])

        print(f"{analysis['word_count']} words, TTR={analysis['type_token_ratio']}, {elapsed:.1f}s")

# Calculate averages
summary = {}
for profile, agg in profile_aggregates.items():
    n = len(agg["word_count"])
    summary[profile] = {
        "temperature": profiles[profile]["temperature"],
        "avg_word_count": round(sum(agg["word_count"]) / n, 1),
        "avg_type_token_ratio": round(sum(agg["ttr"]) / n, 4),
        "avg_empathetic_words": round(sum(agg["empathetic"]) / n, 2),
        "avg_enthusiastic_words": round(sum(agg["enthusiastic"]) / n, 2),
        "avg_formal_words": round(sum(agg["formal"]) / n, 2),
    }

output = {
    "summary": summary,
    "detailed_results": all_results,
    "profiles_tested": profiles,
    "num_queries": len(test_queries),
}

with open("eval_results/hyperparameter_analysis.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*55}")
print(f"  RESULTS SUMMARY")
print(f"{'='*55}")
print(f"\n  {'Profile':12s} {'Temp':>6s} {'AvgWords':>10s} {'TTR':>8s} {'Empath':>8s} {'Enthus':>8s} {'Formal':>8s}")
print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for profile, s in summary.items():
    print(f"  {profile:12s} {s['temperature']:>6.2f} {s['avg_word_count']:>10.1f} {s['avg_type_token_ratio']:>8.4f} "
          f"{s['avg_empathetic_words']:>8.2f} {s['avg_enthusiastic_words']:>8.2f} {s['avg_formal_words']:>8.2f}")

print(f"\nSaved to eval_results/hyperparameter_analysis.json")
