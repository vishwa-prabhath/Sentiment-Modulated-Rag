"""
Generate Publication-Quality Charts for Final Report
=====================================================
Reads evaluation JSON files and produces charts saved as PNG images.
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTPUT_DIR = "eval_results/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'primary': '#6366f1',     # Indigo
    'secondary': '#06b6d4',   # Cyan
    'accent': '#f59e0b',      # Amber
    'success': '#10b981',     # Emerald
    'danger': '#ef4444',      # Red
    'neutral': '#64748b',     # Slate
    'factual': '#3b82f6',     # Blue
    'emotional': '#ec4899',   # Pink
    'creative': '#f59e0b',    # Amber
}

# ============================================================
# Chart 1: Sentiment Classification — Before vs After Overrides
# ============================================================
def chart_sentiment_comparison():
    with open("eval_results/sentiment_metrics.json") as f:
        data = json.load(f)

    classes = ["factual", "creative", "emotional"]
    without = data["without_overrides"]["per_class"]
    with_ov = data["with_overrides"]["per_class"]

    x = np.arange(len(classes))
    width = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # F1 Scores
    ax = axes[0]
    bars1 = ax.bar(x - width/2, [without[c]["f1"] for c in classes], width,
                   label='Without Overrides', color=COLORS['neutral'], alpha=0.7, edgecolor='white')
    bars2 = ax.bar(x + width/2, [with_ov[c]["f1"] for c in classes], width,
                   label='With Overrides', color=COLORS['primary'], edgecolor='white')
    ax.set_xlabel('Task Type')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes])
    ax.set_ylim(0, 1.1)
    ax.legend()
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    # Precision
    ax = axes[1]
    bars1 = ax.bar(x - width/2, [without[c]["precision"] for c in classes], width,
                   label='Without Overrides', color=COLORS['neutral'], alpha=0.7, edgecolor='white')
    bars2 = ax.bar(x + width/2, [with_ov[c]["precision"] for c in classes], width,
                   label='With Overrides', color=COLORS['secondary'], edgecolor='white')
    ax.set_xlabel('Task Type')
    ax.set_ylabel('Precision')
    ax.set_title('Precision Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes])
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Recall
    ax = axes[2]
    bars1 = ax.bar(x - width/2, [without[c]["recall"] for c in classes], width,
                   label='Without Overrides', color=COLORS['neutral'], alpha=0.7, edgecolor='white')
    bars2 = ax.bar(x + width/2, [with_ov[c]["recall"] for c in classes], width,
                   label='With Overrides', color=COLORS['success'], edgecolor='white')
    ax.set_xlabel('Task Type')
    ax.set_ylabel('Recall')
    ax.set_title('Recall Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes])
    ax.set_ylim(0, 1.1)
    ax.legend()

    fig.suptitle('Sentiment Classification: Effect of Override Heuristics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_sentiment_comparison.png")
    plt.close()
    print("  ✅ 1_sentiment_comparison.png")


# ============================================================
# Chart 2: Confusion Matrix Heatmap
# ============================================================
def chart_confusion_matrix():
    with open("eval_results/sentiment_metrics.json") as f:
        data = json.load(f)

    classes = ["factual", "creative", "emotional"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (key, title) in enumerate([
        ("without_overrides", "Without Override Heuristics"),
        ("with_overrides", "With Override Heuristics")
    ]):
        cm = data[key]["confusion_matrix"]
        matrix = np.array([[cm[a][p] for p in classes] for a in classes])

        ax = axes[idx]
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues' if idx == 0 else 'Purples',
                    xticklabels=[c.capitalize() for c in classes],
                    yticklabels=[c.capitalize() for c in classes],
                    ax=ax, cbar=False, linewidths=1, linecolor='white',
                    annot_kws={"size": 14, "fontweight": "bold"})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{title}\n(Accuracy: {data[key]["accuracy"]}%)', fontweight='bold')

    fig.suptitle('Confusion Matrix: Sentiment Classification', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_confusion_matrix.png")
    plt.close()
    print("  ✅ 2_confusion_matrix.png")


# ============================================================
# Chart 3: Overall Accuracy Improvement (Bar)
# ============================================================
def chart_accuracy_improvement():
    with open("eval_results/sentiment_metrics.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ['Without Overrides', 'With Overrides']
    values = [data["without_overrides"]["accuracy"], data["with_overrides"]["accuracy"]]
    colors = [COLORS['neutral'], COLORS['primary']]

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='white', linewidth=2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Draw improvement arrow
    ax.annotate('', xy=(1, values[1] - 2), xytext=(0, values[0] + 2),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2.5))
    ax.text(0.5, (values[0] + values[1]) / 2, f'+{data["improvement"]}%',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=COLORS['success'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['success']))

    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_title('Impact of Override Heuristics on Classification Accuracy',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='Baseline (50%)')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_accuracy_improvement.png")
    plt.close()
    print("  ✅ 3_accuracy_improvement.png")


# ============================================================
# Chart 4: Hyperparameter Effect — Multi-metric radar/bar chart
# ============================================================
def chart_hyperparameter_effect():
    with open("eval_results/hyperparameter_analysis.json") as f:
        data = json.load(f)

    summary = data["summary"]
    profiles = list(summary.keys())
    colors_list = [COLORS['factual'], COLORS['emotional'], COLORS['creative']]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Chart 4a: Average Word Count
    ax = axes[0]
    vals = [summary[p]["avg_word_count"] for p in profiles]
    bars = ax.bar([p.capitalize() for p in profiles], vals, color=colors_list, edgecolor='white', width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Word Count')
    ax.set_title('Response Length by Profile')
    ax.set_ylim(0, max(vals) * 1.2)

    # Chart 4b: Type-Token Ratio
    ax = axes[1]
    vals = [summary[p]["avg_type_token_ratio"] for p in profiles]
    bars = ax.bar([p.capitalize() for p in profiles], vals, color=colors_list, edgecolor='white', width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Type-Token Ratio (TTR)')
    ax.set_title('Lexical Diversity by Profile')
    ax.set_ylim(0, 1.0)

    # Chart 4c: Empathetic Word Frequency
    ax = axes[2]
    emp = [summary[p]["avg_empathetic_words"] for p in profiles]
    ent = [summary[p]["avg_enthusiastic_words"] for p in profiles]
    x = np.arange(len(profiles))
    width = 0.3
    ax.bar(x - width/2, emp, width, label='Empathetic', color=COLORS['emotional'], edgecolor='white')
    ax.bar(x + width/2, ent, width, label='Enthusiastic', color=COLORS['creative'], edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in profiles])
    ax.set_ylabel('Avg Word Count')
    ax.set_title('Tone Words by Profile')
    ax.legend()

    fig.suptitle('Dynamic Hyperparameter Effect on Response Characteristics',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_hyperparameter_effect.png")
    plt.close()
    print("  ✅ 4_hyperparameter_effect.png")


# ============================================================
# Chart 5: Response Quality — BLEU & ROUGE scores
# ============================================================
def chart_quality_scores():
    with open("eval_results/quality_metrics.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Chart 5a: Overall scores bar chart
    ax = axes[0]
    metrics = ['BLEU', 'ROUGE-L', 'Grounding']
    values = [data["avg_bleu"], data["avg_rouge_l"], data["avg_grounding"]]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success']]

    bars = ax.bar(metrics, values, color=colors, width=0.5, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
                f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_title('Average Quality Metrics (n=21)')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.text(2.6, 0.51, 'Baseline', fontsize=9, color='gray')

    # Chart 5b: Per-query BLEU distribution
    ax = axes[1]
    bleu_scores = [r["bleu"] for r in data["detailed_results"]]
    rouge_scores = [r["rouge_l"] for r in data["detailed_results"]]

    ax.hist(bleu_scores, bins=10, alpha=0.7, label='BLEU', color=COLORS['primary'], edgecolor='white')
    ax.hist(rouge_scores, bins=10, alpha=0.5, label='ROUGE-L', color=COLORS['secondary'], edgecolor='white')
    ax.set_xlabel('Score')
    ax.set_ylabel('Number of Queries')
    ax.set_title('Score Distribution Across 21 Queries')
    ax.legend()
    ax.axvline(x=data["avg_bleu"], color=COLORS['primary'], linestyle='--', alpha=0.5)
    ax.axvline(x=data["avg_rouge_l"], color=COLORS['secondary'], linestyle='--', alpha=0.5)

    fig.suptitle('End-to-End Response Quality Evaluation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_quality_scores.png")
    plt.close()
    print("  ✅ 5_quality_scores.png")


# ============================================================
# Chart 6: Per-Category Quality Breakdown
# ============================================================
def chart_category_quality():
    with open("eval_results/quality_metrics.json") as f:
        data = json.load(f)

    cats = data["per_category"]
    # Sort by BLEU score
    sorted_cats = sorted(cats.items(), key=lambda x: x[1]["avg_bleu"], reverse=True)
    names = [c[0] for c in sorted_cats]
    bleu = [c[1]["avg_bleu"] for c in sorted_cats]
    rouge = [c[1]["avg_rouge_l"] for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.35

    ax.barh(x - width/2, bleu, width, label='BLEU', color=COLORS['primary'], edgecolor='white')
    ax.barh(x + width/2, rouge, width, label='ROUGE-L', color=COLORS['secondary'], edgecolor='white')

    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.set_xlabel('Score')
    ax.set_title('Response Quality by Category', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_category_quality.png")
    plt.close()
    print("  ✅ 6_category_quality.png")


# ============================================================
# Chart 7: System Architecture Pipeline Diagram
# ============================================================
def chart_pipeline_flow():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Pipeline boxes
    boxes = [
        (1, 3, 'Student\nQuery', COLORS['neutral']),
        (3.5, 3, 'RAG\nRetrieval', COLORS['factual']),
        (6, 3, 'HAG\nClassification', COLORS['emotional']),
        (8.5, 3, 'Dynamic\nHyperparams', COLORS['creative']),
        (11, 3, 'LLM\nGeneration', COLORS['primary']),
        (13, 3, 'Response', COLORS['success']),
    ]

    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x - 0.9, y - 0.7), 1.8, 1.4,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.85,
                                        edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.9
        x2 = boxes[i+1][0] - 0.9
        ax.annotate('', xy=(x2, 3), xytext=(x1, 3),
                    arrowprops=dict(arrowstyle='->', color='#334155', lw=2))

    # Sub-labels
    sublabels = [
        (1, 1.8, 'User Input'),
        (3.5, 1.8, 'ChromaDB\n64 chunks'),
        (6, 1.8, 'DistilBERT\n+ Overrides'),
        (8.5, 1.8, 'temp / top_p\ntop_k'),
        (11, 1.8, 'Llama 3.1\nvia Ollama'),
        (13, 1.8, 'Grounded\n+ Toned'),
    ]
    for x, y, text in sublabels:
        ax.text(x, y, text, ha='center', va='center', fontsize=8, color='#64748b', style='italic')

    # Memory feedback loop
    ax.annotate('', xy=(11, 4.5), xytext=(3.5, 4.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5, linestyle='--'))
    ax.text(7.25, 4.8, 'Conversational Memory (ChromaDB)', ha='center', va='center',
            fontsize=9, color=COLORS['accent'], fontweight='bold')

    ax.set_title('Sentiment Modulated RAG — Pipeline Architecture', fontsize=14, fontweight='bold', y=1.0)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/7_pipeline_architecture.png")
    plt.close()
    print("  ✅ 7_pipeline_architecture.png")


# ============================================================
# Chart 8: Temperature vs Response Characteristics (Line)
# ============================================================
def chart_temperature_line():
    with open("eval_results/hyperparameter_analysis.json") as f:
        data = json.load(f)

    summary = data["summary"]
    temps = [summary[p]["temperature"] for p in ["factual", "emotional", "creative"]]
    ttr = [summary[p]["avg_type_token_ratio"] for p in ["factual", "emotional", "creative"]]
    words = [summary[p]["avg_word_count"] for p in ["factual", "emotional", "creative"]]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = COLORS['primary']
    color2 = COLORS['accent']

    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Type-Token Ratio (TTR)', color=color1)
    line1 = ax1.plot(temps, ttr, 'o-', color=color1, linewidth=2.5, markersize=10, label='TTR (Lexical Diversity)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.5, 1.0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Word Count', color=color2)
    line2 = ax2.plot(temps, words, 's--', color=color2, linewidth=2.5, markersize=10, label='Word Count')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(100, 170)

    # Labels
    for t, v in zip(temps, ttr):
        ax1.annotate(f'{v:.4f}', (t, v), textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=9, color=color1, fontweight='bold')
    for t, v in zip(temps, words):
        ax2.annotate(f'{v:.0f}', (t, v), textcoords="offset points", xytext=(0, -15),
                     ha='center', fontsize=9, color=color2, fontweight='bold')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')

    ax1.set_xticks(temps)
    ax1.set_xticklabels(['0.3\n(Factual)', '0.6\n(Emotional)', '0.85\n(Creative)'])
    ax1.set_title('Effect of Temperature on Response Characteristics',
                  fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/8_temperature_effect.png")
    plt.close()
    print("  ✅ 8_temperature_effect.png")


# ============================================================
# RUN ALL
# ============================================================
print("=" * 55)
print("  Generating Charts for Final Report")
print("=" * 55)
print()

chart_sentiment_comparison()
chart_confusion_matrix()
chart_accuracy_improvement()
chart_hyperparameter_effect()
chart_quality_scores()
chart_category_quality()
chart_pipeline_flow()
chart_temperature_line()

print(f"\n{'='*55}")
print(f"  All 8 charts saved to: {OUTPUT_DIR}/")
print(f"{'='*55}")
