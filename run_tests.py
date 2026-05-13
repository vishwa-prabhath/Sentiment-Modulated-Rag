"""
Automated Test Runner for Sentiment Modulated RAG
Uses Ollama for generation, subprocess isolation for classifier/embedder.
"""
import subprocess
import sys
import json
import time
import os
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"
PYTHON = sys.executable  # Use whatever Python is running this script

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"

test_questions = [
    # Category A: Factual (Neutral Sentiment)
    "What are the core subjects offered in the Computer Science degree program at NSBM?",
    "Can you tell me the duration of the Software Engineering degree?",

    # Category B: Emotional (Sadness / Fear)
    "I am really stressed out because I don't know the late payment penalty policy. I'm afraid I might get dropped from my classes.",
    "I failed my last module and I am very depressed. What is the policy for retaking a module?",

    # Category C: Joyful (Joy / Surprise)
    "I just got my acceptance letter for the IT program and I'm so excited! What amazing facilities does the computing faculty have?",
    "Wow, I didn't expect to get a scholarship! What are the next steps to enroll?",

    # Category D: Edge Cases (Mixed / Sarcasm)
    "I am so happy that I passed my exam, but I am extremely anxious about the upcoming tuition fee payment. What is the deadline?",
    "Oh great, another fee increase. How much do I have to pay for the semester now?"
]

from unified_system import generate_unified_response


def run_all_tests():
    print("=" * 60)
    print("  Automated Test Runner - Sentiment Modulated RAG")
    print("  Backend: Ollama + ChromaDB + DistilBERT")
    print("=" * 60)

    # Quick health check
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
        print("  ✅ Ollama is running\n")
    except Exception:
        print("  ❌ Ollama is not running! Start it with: ollama serve")
        return

    with open("test_results.md", "w") as f:
        f.write("# Automated Test Results: Sentiment Modulated RAG\n\n")
        f.write(f"**Model:** `{OLLAMA_MODEL}` (via Ollama)  \n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}  \n\n---\n\n")
        f.flush()

        for i, query in enumerate(test_questions, 1):
            print(f"--- Test {i}/{len(test_questions)} ---")
            print(f"  Q: {query[:65]}...")
            f.write(f"## Test Case {i}\n")
            f.write(f"**Query:** `{query}`\n\n")
            f.flush()

            try:
                start = time.time()

                # Run Unified Pipeline
                print("  [1/1] Generating unified response...")
                response, task_type, emotion, confidence, hyperparams = generate_unified_response(query)

                elapsed = time.time() - start

                f.write(f"**Detected Emotion:** `{emotion}` (confidence: {confidence:.2f})  \n")
                f.write(f"**Task Type:** `{task_type.upper()}`\n\n")
                f.write(f"**Applied Hyperparameters:** `{hyperparams}`\n\n")
                f.write(f"**Retrieved Context Snippet:**\n```text\n(Retrieved context is internal to unified pipeline)\n```\n\n")
                f.write(f"**AI Response:**\n> {response}\n\n")
                f.write(f"*Total Time: {elapsed:.1f}s*\n\n---\n\n")
                f.flush()

                print(f"  ✅ Done! Emotion={emotion}, Task={task_type.upper()}, Time={elapsed:.1f}s")
                print(f"  Response: {response[:100]}...\n")

            except Exception as e:
                print(f"  ❌ FAILED: {str(e)[:200]}\n")
                f.write(f"**Error:** `{str(e)[:300]}`\n\n---\n\n")
                f.flush()

    print("=" * 60)
    print("All tests completed! Results saved to 'test_results.md'")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
