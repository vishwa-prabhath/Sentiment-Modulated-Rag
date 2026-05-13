# Automated Test Results: Sentiment Modulated RAG

**Model:** `qwen2.5:1.5b` (via Ollama)  
**Date:** 2026-05-12 10:48  

---

## Test Case 1
**Query:** `What are the core subjects offered in the Computer Science degree program at NSBM?`

**Detected Emotion:** `joy` (confidence: 0.79)  
**Task Type:** `FACTUAL`

**Applied Hyperparameters:** `{'temperature': 0.36, 'top_p': 0.85, 'top_k': 46, 'repetition_penalty': 1.17}`

**Retrieved Context Snippet:**
```text
(Retrieved context is internal to unified pipeline)
```

**AI Response:**
> Unfortunately, I couldn't find any information on the specific core subjects offered in the BSc (Hons) Computer Science – (UGC Approved – Offered By NSBM) program. The context only mentions that it's a UGC-approved course and is offered by NSBM, but doesn't provide details on its curriculum.

However, I can suggest checking with the Department of Computer and Data Science or contacting them directly for more information on this specific degree program.

*Total Time: 15.4s*

---

## Test Case 2
**Query:** `Can you tell me the duration of the Software Engineering degree?`

**Detected Emotion:** `joy` (confidence: 0.75)  
**Task Type:** `FACTUAL`

**Applied Hyperparameters:** `{'temperature': 0.33899999999999997, 'top_p': 0.85, 'top_k': 43, 'repetition_penalty': 1.1804999999999999}`

**Retrieved Context Snippet:**
```text
(Retrieved context is internal to unified pipeline)
```

**AI Response:**
> Unfortunately, I don't have that information. The provided context only lists the available degrees offered by NSBM's Faculty of Computing but does not mention anything about the duration of each program. If you need to know the exact duration, it would be best to contact the university directly or check their official website for more details.

*Total Time: 12.9s*

---

## Test Case 3
**Query:** `I am really stressed out because I don't know the late payment penalty policy. I'm afraid I might get dropped from my classes.`

**Detected Emotion:** `fear` (confidence: 0.90)  
**Task Type:** `EMOTIONAL`

**Applied Hyperparameters:** `{'temperature': 0.5398999999999999, 'top_p': 0.885, 'top_k': 49, 'repetition_penalty': 1.11505}`

**Retrieved Context Snippet:**
```text
(Retrieved context is internal to unified pipeline)
```

**AI Response:**
> I apologize, but based on the provided context, I do not have information about a late payment penalty policy or any consequences related to it. The context only mentions the university's contact details and some general information about the Faculty of Computing, but does not provide specific policies regarding payments or penalties. It would be best to consult with technical staff at NSBM Green University for accurate guidance on this matter.

*Total Time: 14.6s*

---

## Test Case 4
**Query:** `I failed my last module and I am very depressed. What is the policy for retaking a module?`

**Detected Emotion:** `sadness` (confidence: 1.00)  
**Task Type:** `FACTUAL`

**Applied Hyperparameters:** `{'temperature': 0.34041749999999993, 'top_p': 0.852625, 'top_k': 42, 'repetition_penalty': 1.18241625}`

**Retrieved Context Snippet:**
```text
(Retrieved context is internal to unified pipeline)
```

**AI Response:**
> I'm so sorry to hear you're feeling down about failing your module! Unfortunately, our context doesn't contain any information on policies regarding retaking modules or academic support services that might be helpful in this situation.

However, I can suggest reaching out to the technical staff mentioned at the beginning of the document. They may be able to provide guidance or direct you to someone who can help with your query about retaking a module.

*Total Time: 14.1s*

---

## Test Case 5
**Query:** `I just got my acceptance letter for the IT program and I'm so excited! What amazing facilities does the computing faculty have?`

**Detected Emotion:** `joy` (confidence: 0.99)  
**Task Type:** `FACTUAL`

**Applied Hyperparameters:** `{'temperature': 0.33475904999999995, 'top_p': 0.8522574999999999, 'top_k': 42, 'repetition_penalty': 1.184877975}`

**Retrieved Context Snippet:**
```text
(Retrieved context is internal to unified pipeline)
```

**AI Response:**
> Unfortunately, I couldn't find any information about the specific facilities available to students within the Faculty of Computing. The provided context mentions that the Faculty provides "world-class education and training" in Computing and Information Technology, but it doesn't specify what kind of facilities are available.

However, based on the general tone of the text, it's likely that NSBM has invested in providing a good learning environment for its students. If you're interested in knowing more about the specific facilities, I recommend reaching out to the Faculty directly or checking their official website (if there is one) for more information!

*Total Time: 15.0s*

---

## Test Case 6
**Query:** `Wow, I didn't expect to get a scholarship! What are the next steps to enroll?`

**Detected Emotion:** `joy` (confidence: 0.98)  
**Task Type:** `FACTUAL`

**Applied Hyperparameters:** `{'temperature': 0.32484459299999996, 'top_p': 0.8523929499999999, 'top_k': 41, 'repetition_penalty': 1.1899706535}`

**Retrieved Context Snippet:**
```text
(Retrieved context is internal to unified pipeline)
```

**AI Response:**
> I'm glad you're excited about the possibility of getting a scholarship. However, there's no information provided in this context regarding scholarships or their application process.

If you'd like to inquire about enrolling at NSBM Green University and potentially applying for a scholarship, I recommend contacting them directly through one of the phone numbers (Tel:+94 (11) 544 5000 for new enrollments or Tel+94 (11) 544 6000 for current students) or via email (inquiries@nsbm.ac.lk). They would be able to provide you with more information on their enrollment process and any available scholarships.

*Total Time: 15.5s*

---

## Test Case 7
**Query:** `I am so happy that I passed my exam, but I am extremely anxious about the upcoming tuition fee payment. What is the deadline?`

**Detected Emotion:** `fear` (confidence: 0.99)  
**Task Type:** `FACTUAL`

**Applied Hyperparameters:** `{'temperature': 0.32273526857999996, 'top_p': 0.8525365269999999, 'top_k': 41, 'repetition_penalty': 1.19116889271}`

**Retrieved Context Snippet:**
```text
(Retrieved context is internal to unified pipeline)
```

**AI Response:**
> I'm glad you're excited about passing your exam! Unfortunately, there's no information provided in the context regarding tuition fees or deadlines for payments. It seems to be focused on introducing the Faculty of Computing and its degree programs rather than discussing financial matters like fee payment schedules. I recommend contacting the university directly via their phone numbers (New Enrollments: +94 (11) 544 5000, Current Students: +94 (11) 544 6000) or emails (inquiries@nsbm.ac.lk for new enrollments and ar.foc@nsbm.ac.lk for current students) to inquire about the tuition fee payment deadline.

*Total Time: 17.6s*

---

## Test Case 8
**Query:** `Oh great, another fee increase. How much do I have to pay for the semester now?`

**Detected Emotion:** `joy` (confidence: 0.88)  
**Task Type:** `FACTUAL`

**Applied Hyperparameters:** `{'temperature': 0.3217593846948, 'top_p': 0.8526887186200001, 'top_k': 40, 'repetition_penalty': 1.1918090262726}`

**Retrieved Context Snippet:**
```text
(Retrieved context is internal to unified pipeline)
```

**AI Response:**
> I'm sorry but there is no information provided about fees or payment amounts related to the Faculty of Computing at NSBM Green University in the context given.

*Total Time: 11.8s*

---

