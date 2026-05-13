"""
Evaluation 2: Sentiment Classification Accuracy (HAG Component)
================================================================
Evaluates the DistilBERT classifier + override heuristics against
a manually labelled test set. Produces confusion matrix and F1 scores.
Compares performance WITH and WITHOUT the override heuristics.
"""
import os, json, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline as hf_pipeline
import re

# ---- Override heuristics (same as in unified_system.py) ----
FACTUAL_KEYWORDS = re.compile(
    r'\b(what|when|where|which|how\s+(long|many|much|do|does|is|are|can)|'
    r'tell me|list|explain|describe|define|duration|deadline|policy|fee|'
    r'program|course|degree|subject|department|schedule|requirement|enroll|'
    r'admission|credit|semester|gpa|grade|transcript|certificate)\b',
    re.IGNORECASE
)

GREETING_PATTERN = re.compile(
    r'^\s*(hi|hello|hey|hii+|helo+|good\s*(morning|afternoon|evening|day)|'
    r'greetings|sup|yo|howdy|thanks|thank\s*you|ok|okay|bye|goodbye|'
    r'see\s*you|cheers|welcome|please\s*help)\s*[!?.]*\s*$',
    re.IGNORECASE
)

def is_greeting(text):
    return bool(GREETING_PATTERN.match(text.strip()))

def is_likely_factual(text):
    matches = FACTUAL_KEYWORDS.findall(text)
    if len(matches) >= 2:
        return True
    if len(text.split()) <= 12 and FACTUAL_KEYWORDS.search(text):
        return True
    return False

# ---- Test dataset with manually labelled expected task types ----
labelled_tests = [
    # Factual queries
    {"query": "What are the core subjects offered in the Computer Science degree?", "expected": "factual"},
    {"query": "How long is the Software Engineering degree?", "expected": "factual"},
    {"query": "What is the dean's message?", "expected": "factual"},
    {"query": "What are the degree transfer policies?", "expected": "factual"},
    {"query": "How can I get a student ID card?", "expected": "factual"},
    {"query": "What courses does NSBM offer?", "expected": "factual"},
    {"query": "Where is NSBM located?", "expected": "factual"},
    {"query": "What is the late payment penalty?", "expected": "factual"},
    {"query": "Tell me about the Faculty of Computing departments", "expected": "factual"},
    {"query": "What are the enrollment requirements?", "expected": "factual"},

    # Emotional queries (sadness, fear, stress)
    {"query": "I am really stressed out because I don't know the late payment penalty policy. I'm afraid I might get dropped.", "expected": "emotional"},
    {"query": "I failed my last module and I am very depressed. What is the policy for retaking?", "expected": "emotional"},
    {"query": "I'm so worried about my grades this semester, I feel like giving up.", "expected": "emotional"},
    {"query": "I feel so alone and lost on campus. Is there counselling support?", "expected": "emotional"},
    {"query": "I am scared I won't be able to afford next semester's fees.", "expected": "emotional"},

    # Creative / Joyful queries
    {"query": "I just got my acceptance letter and I'm so excited! What facilities does computing have?", "expected": "creative"},
    {"query": "Wow, I didn't expect to get a scholarship! What are the next steps?", "expected": "creative"},
    {"query": "I am so happy I got into NSBM! This is amazing!", "expected": "creative"},
    {"query": "I love this university! The campus is beautiful!", "expected": "creative"},
    {"query": "Just passed all my exams with flying colors! I'm thrilled!", "expected": "creative"},

    # Greetings / Neutral
    {"query": "Hi", "expected": "creative"},  # greeting → creative
    {"query": "Hello!", "expected": "creative"},
    {"query": "Good morning", "expected": "creative"},
    {"query": "Hey there", "expected": "creative"},
    {"query": "Thanks", "expected": "creative"},

    # Edge cases
    {"query": "I am happy I passed but anxious about the tuition fee deadline.", "expected": "emotional"},
    {"query": "Oh great, another fee increase. How much now?", "expected": "factual"},
    {"query": "ok", "expected": "factual"},
    {"query": "hmm", "expected": "factual"},
    {"query": "yes", "expected": "factual"},
]

print(f"Evaluating sentiment classification on {len(labelled_tests)} labelled queries...\n")

# Load classifier once
classifier = hf_pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)

results_raw = []     # Without overrides
results_fixed = []   # With overrides

for i, test in enumerate(labelled_tests):
    query = test["query"]
    expected = test["expected"]

    result = classifier(query)[0]
    emotion = result["label"]
    confidence = result["score"]

    # --- RAW classification (no overrides) ---
    if emotion in ["joy", "surprise"]:
        raw_task = "creative"
    elif emotion in ["sadness", "fear", "love"]:
        raw_task = "emotional"
    else:
        raw_task = "factual"

    # --- WITH overrides ---
    if is_greeting(query):
        fixed_task = "creative"
        fixed_emotion = "neutral"
    elif is_likely_factual(query):
        fixed_task = "factual"
        fixed_emotion = emotion
    elif len(query.split()) <= 3 and emotion == "anger":
        fixed_task = "factual"
        fixed_emotion = "neutral"
    elif confidence < 0.5:
        fixed_task = "factual"
        fixed_emotion = emotion
    elif emotion in ["joy", "surprise"]:
        fixed_task = "creative"
        fixed_emotion = emotion
    elif emotion in ["sadness", "fear", "love"]:
        fixed_task = "emotional"
        fixed_emotion = emotion
    else:
        fixed_task = "factual"
        fixed_emotion = emotion

    results_raw.append({"query": query, "expected": expected, "predicted": raw_task, "emotion": emotion, "confidence": confidence})
    results_fixed.append({"query": query, "expected": expected, "predicted": fixed_task, "emotion": fixed_emotion, "confidence": confidence})

del classifier
gc.collect()

# --- Calculate metrics ---
def calc_metrics(results):
    classes = ["factual", "creative", "emotional"]
    correct = sum(1 for r in results if r["predicted"] == r["expected"])
    accuracy = round(correct / len(results) * 100, 1)

    # Per-class precision, recall, F1
    per_class = {}
    for cls in classes:
        tp = sum(1 for r in results if r["predicted"] == cls and r["expected"] == cls)
        fp = sum(1 for r in results if r["predicted"] == cls and r["expected"] != cls)
        fn = sum(1 for r in results if r["predicted"] != cls and r["expected"] == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[cls] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for r in results if r["expected"] == cls)
        }

    # Confusion matrix
    confusion = {actual: {pred: 0 for pred in classes} for actual in classes}
    for r in results:
        if r["expected"] in classes and r["predicted"] in classes:
            confusion[r["expected"]][r["predicted"]] += 1

    return {
        "accuracy": accuracy,
        "total": len(results),
        "correct": correct,
        "per_class": per_class,
        "confusion_matrix": confusion
    }

raw_metrics = calc_metrics(results_raw)
fixed_metrics = calc_metrics(results_fixed)

output = {
    "without_overrides": raw_metrics,
    "with_overrides": fixed_metrics,
    "improvement": round(fixed_metrics["accuracy"] - raw_metrics["accuracy"], 1),
    "detailed_raw": results_raw,
    "detailed_fixed": results_fixed,
}

with open("eval_results/sentiment_metrics.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"{'='*55}")
print(f"  SENTIMENT CLASSIFICATION EVALUATION")
print(f"{'='*55}")
print(f"\n  WITHOUT Overrides:")
print(f"    Accuracy: {raw_metrics['accuracy']}% ({raw_metrics['correct']}/{raw_metrics['total']})")
for cls, m in raw_metrics["per_class"].items():
    print(f"    {cls:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (n={m['support']})")

print(f"\n  WITH Overrides (greeting + factual + short-msg):")
print(f"    Accuracy: {fixed_metrics['accuracy']}% ({fixed_metrics['correct']}/{fixed_metrics['total']})")
for cls, m in fixed_metrics["per_class"].items():
    print(f"    {cls:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (n={m['support']})")

print(f"\n  Improvement: +{output['improvement']}%")

print(f"\n  Confusion Matrix (WITH overrides):")
classes = ["factual", "creative", "emotional"]
print(f"  {'':15s} {'Pred_Fact':>10s} {'Pred_Crea':>10s} {'Pred_Emot':>10s}")
for actual in classes:
    row = [str(fixed_metrics["confusion_matrix"][actual][pred]) for pred in classes]
    print(f"  {'Act_'+actual:15s} {row[0]:>10s} {row[1]:>10s} {row[2]:>10s}")

print(f"\nSaved to eval_results/sentiment_metrics.json")
