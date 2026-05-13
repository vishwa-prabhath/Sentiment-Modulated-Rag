# Testing Plan: Sentiment Modulated RAG Framework

## 1. Testing Objectives
The primary objective of this testing phase is to validate that the hybrid architecture successfully bridges the gap between factual rigidity and emotional tone-deafness. Specifically, we need to prove:
- **Factual Accuracy:** The model correctly retrieves and adheres to NSBM university policies without hallucinating.
- **Affective Alignment:** The model correctly identifies user sentiment and adjusts its hyperparameter tuning (tone) accordingly.
- **Progressive Parameter Evolution:** The system handles sudden emotional shifts smoothly without abrupt tonal breaks (this will be tested iteratively).

## 2. Test Case Categories

### Category A: Pure Factual Queries (Neutral Sentiment)
*Goal: Ensure RAG retrieval is accurate and generation uses low-temperature for strict policy adherence.*
1. "What are the core subjects offered in the Computer Science degree program at NSBM?"
2. "Can you tell me the duration of the Software Engineering degree?"

### Category B: Emotional Queries (Anxiety / Sadness / Fear)
*Goal: Ensure the sentiment classifier detects the emotion and applies empathetic, moderate hyperparameters, while still providing factual answers.*
3. "I am really stressed out because I don't know the late payment penalty policy. I'm afraid I might get dropped from my classes."
4. "I failed my last module and I am very depressed. What is the policy for retaking a module?"

### Category C: Creative / Joyful Queries (Joy / Surprise)
*Goal: Ensure the model applies higher temperature for a more enthusiastic and creative tone without deviating from facts.*
5. "I just got my acceptance letter for the IT program and I'm so excited! What amazing facilities does the computing faculty have?"
6. "Wow, I didn't expect to get a scholarship! What are the next steps to enroll?"

### Category D: Edge Cases (Mixed Emotions & Sarcasm)
*Goal: Stress test the system against ambiguous inputs as outlined in Section 9.3 of the interim report.*
7. **Mixed Emotion:** "I am so happy that I passed my exam, but I am extremely anxious about the upcoming tuition fee payment. What is the deadline?"
8. **Sarcasm:** "Oh great, another fee increase. How much do I have to pay for the semester now?"

## 3. Testing Execution Strategy (Automated Script)
We will use an automated script (`run_tests.py`) that feeds these predefined questions into the `unified_system.py` pipeline.
- The script will capture the **Detected Intent**, the **Dynamic Parameters Applied**, and the **AI's Generated Response**.
- Results will be logged into a file (`test_results.md`) for manual qualitative review.

## 4. Evaluation Metrics
For each test case, you should evaluate:
1. **Retrieval Success:** Did it find the right context from the PDF?
2. **Sentiment Accuracy:** Did the DistilBERT model classify the emotion correctly?
3. **Response Quality:** Is the tone appropriate? Is the information factually correct based on the PDF?
