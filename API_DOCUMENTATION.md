# Sentiment Modulated RAG — REST API Documentation

> **Version:** 1.0.0  
> **Base URL:** `http://localhost:5001/api/v1`  
> **Interactive Docs (Swagger UI):** `http://localhost:5001/docs`

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [Authentication](#authentication)
5. [API Endpoints](#api-endpoints)
   - [Health Check](#1-health-check)
   - [Submit Query](#2-submit-query)
   - [Knowledge Base Stats](#3-knowledge-base-stats)
   - [Ingest PDF](#4-ingest-pdf)
   - [Get Memory](#5-get-conversational-memory)
   - [Clear Memory](#6-clear-conversational-memory)
6. [Data Models](#data-models)
7. [Error Handling](#error-handling)
8. [Configuration](#configuration)
9. [Examples](#examples)

---

## Overview

The **Sentiment Modulated RAG API** exposes the hybrid RAG+HAG pipeline as a set of RESTful endpoints. It combines:

- **RAG (Retrieval-Augmented Generation):** Retrieves factual context from a ChromaDB knowledge base populated with NSBM university data.
- **HAG (Hyperparameter-Augmented Generation):** Uses DistilBERT emotion classification to dynamically tune LLM generation parameters (temperature, top_p, top_k, repetition_penalty).

The system generates contextually accurate and emotionally appropriate responses to student queries.

---

## Architecture

```
┌─────────────┐    HTTP POST     ┌─────────────────────────────────────────────┐
│   Client    │ ──────────────►  │              Flask REST API                 │
│  (Browser,  │                  │                                             │
│   Postman,  │  ◄────────────── │  ┌─────────┐  ┌─────────┐  ┌───────────┐  │
│   curl)     │    JSON Response │  │ ChromaDB │  │DistilBERT│  │  Ollama   │  │
└─────────────┘                  │  │  (RAG)   │  │  (HAG)   │  │  (LLM)   │  │
                                 │  └─────────┘  └─────────┘  └───────────┘  │
                                 └─────────────────────────────────────────────┘
```

### Pipeline Flow

```
User Query
    │
    ▼
┌──────────────────────────┐
│ 1. RAG: Context Retrieval│  ← ChromaDB + SentenceTransformer
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 2. HAG: Sentiment        │  ← DistilBERT emotion classifier
│    Classification         │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 3. HAG: Hyperparameter   │  ← Emotion → {temperature, top_p, top_k, ...}
│    Selection              │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 4. Generate Response     │  ← Ollama API (qwen2.5:1.5b)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 5. Log to Memory         │  ← ChromaDB conversational_memory
└──────────┬───────────────┘
           ▼
      JSON Response
```

---

## Getting Started

### Prerequisites

1. **Python 3.9+** with a virtual environment
2. **Ollama** running locally with `qwen2.5:1.5b` model pulled
3. Required Python packages installed

### Installation

```bash
# Activate your virtual environment
source venv312/bin/activate

# Install dependencies
pip install flask flask-cors flask-restx

# Make sure Ollama is running
ollama serve

# Pull the model (if not already done)
ollama pull qwen2.5:1.5b
```

### Starting the API Server

```bash
python api.py
```

The server starts at `http://localhost:5001`. Visit `http://localhost:5001/docs` for the interactive Swagger UI.

---

## Authentication

Currently, the API does **not** require authentication. It is designed for local development and testing. For production deployment, consider adding:
- API key authentication via headers
- OAuth2 / JWT token-based auth
- Rate limiting

---

## API Endpoints

### 1. Health Check

Check the system's health status including Ollama connectivity, ChromaDB status, and model availability.

| Property | Value |
|----------|-------|
| **URL** | `GET /api/v1/health` |
| **Auth** | None |
| **Rate Limit** | None |

#### Response

```json
{
  "status": "healthy",
  "ollama": "connected (model available)",
  "ollama_model": "qwen2.5:1.5b",
  "chromadb": "connected",
  "knowledge_base_docs": 42,
  "memory_entries": 5
}
```

#### Status Values

| Status | Meaning |
|--------|---------|
| `healthy` | All systems operational |
| `degraded` | Ollama connected but model not found |
| `unhealthy` | Critical component (Ollama or ChromaDB) is down |

---

### 2. Submit Query

Submit a student query to the full Sentiment Modulated RAG pipeline.

| Property | Value |
|----------|-------|
| **URL** | `POST /api/v1/query` |
| **Content-Type** | `application/json` |
| **Auth** | None |

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | — | The student's question or message |
| `top_k` | integer | ❌ | 3 | Number of context documents to retrieve (1–10) |

```json
{
  "query": "I am really stressed about the late payment penalty. What happens if I miss the deadline?",
  "top_k": 3
}
```

#### Response (200 OK)

```json
{
  "response": "I understand your concern about the late payment penalty. Based on the university policy...",
  "emotion": "fear",
  "confidence": 0.8734,
  "task_type": "emotional",
  "hyperparameters": {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
  },
  "context_retrieved": true,
  "processing_time_seconds": 3.42
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | The AI-generated answer |
| `emotion` | string | Detected emotion: `joy`, `sadness`, `fear`, `anger`, `surprise`, `love` |
| `confidence` | float | Emotion detection confidence (0.0–1.0) |
| `task_type` | string | Mapped task type: `factual`, `creative`, `emotional` |
| `hyperparameters` | object | The dynamically selected generation parameters |
| `context_retrieved` | boolean | Whether relevant context was found in the KB |
| `processing_time_seconds` | float | Total pipeline execution time |

#### Emotion → Task Type Mapping

| Emotions | Task Type | Temperature | Behavior |
|----------|-----------|-------------|----------|
| `anger`, low confidence | `factual` | 0.3 | Strict, precise answers |
| `joy`, `surprise` | `creative` | 0.85 | Enthusiastic, creative tone |
| `sadness`, `fear`, `love` | `emotional` | 0.6 | Empathetic, warm tone |

---

### 3. Knowledge Base Stats

Get statistics about the ChromaDB knowledge base.

| Property | Value |
|----------|-------|
| **URL** | `GET /api/v1/knowledge-base/stats` |
| **Auth** | None |

#### Response

```json
{
  "total_documents": 42,
  "collection_name": "knowledge_base"
}
```

---

### 4. Ingest PDF

Upload and ingest a PDF document into the knowledge base. Each page is embedded and stored as a separate document.

| Property | Value |
|----------|-------|
| **URL** | `POST /api/v1/knowledge-base/ingest` |
| **Content-Type** | `application/json` |
| **Auth** | None |

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pdf_path` | string | ✅ | File system path to the PDF |

```json
{
  "pdf_path": "RAG_V2/nsbm_foc_data.pdf"
}
```

#### Response (200 OK)

```json
{
  "message": "Successfully ingested 15 pages from 'RAG_V2/nsbm_foc_data.pdf'.",
  "pages_ingested": 15
}
```

---

### 5. Get Conversational Memory

Retrieve stored conversation history from the ChromaDB memory collection.

| Property | Value |
|----------|-------|
| **URL** | `GET /api/v1/memory` |
| **Auth** | None |

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 20 | Max entries to return (1–100) |

#### Response

```json
{
  "total_entries": 8,
  "entries": [
    {
      "id": "a1b2c3d4-...",
      "document": "Prompt: What are the core subjects...\nResponse: The core subjects include...",
      "task_type": "factual"
    }
  ]
}
```

---

### 6. Clear Conversational Memory

Delete all entries from the conversational memory store.

| Property | Value |
|----------|-------|
| **URL** | `DELETE /api/v1/memory` |
| **Auth** | None |

#### Response (200 OK)

```json
{
  "message": "Conversational memory cleared successfully."
}
```

---

## Data Models

### Hyperparameters

| Field | Type | Description | Range |
|-------|------|-------------|-------|
| `temperature` | float | Controls randomness in generation | 0.0–1.0 |
| `top_p` | float | Nucleus sampling probability threshold | 0.0–1.0 |
| `top_k` | integer | Limits token selection to top K candidates | 1–100 |
| `repetition_penalty` | float | Penalizes repeated tokens | 1.0–2.0 |

### Emotion Labels

The DistilBERT classifier (`bhadresh-savani/distilbert-base-uncased-emotion`) detects the following emotions:

| Label | Description |
|-------|-------------|
| `joy` | Happy, excited, positive |
| `sadness` | Sad, depressed, disappointed |
| `anger` | Angry, frustrated, annoyed |
| `fear` | Anxious, worried, scared |
| `surprise` | Surprised, shocked, amazed |
| `love` | Affectionate, caring, grateful |

---

## Error Handling

All errors follow a consistent JSON format:

```json
{
  "error": "Brief error description",
  "details": "Detailed information about what went wrong"
}
```

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| `200` | Success | Request processed successfully |
| `400` | Bad Request | Missing/invalid fields, empty query |
| `404` | Not Found | Invalid endpoint URL |
| `500` | Internal Server Error | Pipeline failure, model error |
| `503` | Service Unavailable | Ollama server not running |

---

## Configuration

The API can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama model name for generation |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `CHROMA_PATH` | `./unified_chroma_store` | Path to ChromaDB persistent storage |
| `API_PORT` | `5000` | Port for the Flask server |

Example:

```bash
OLLAMA_MODEL=llama3.2:3b API_PORT=8080 python api.py
```

---

## Examples

### cURL Examples

#### Health Check
```bash
curl http://localhost:5000/api/v1/health
```

#### Submit a Factual Query
```bash
curl -X POST http://localhost:5000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the duration of the Software Engineering degree?"}'
```

#### Submit an Emotional Query
```bash
curl -X POST http://localhost:5000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "I failed my last module and I am very depressed. What is the policy for retaking a module?", "top_k": 5}'
```

#### Check KB Stats
```bash
curl http://localhost:5000/api/v1/knowledge-base/stats
```

#### Ingest a PDF
```bash
curl -X POST http://localhost:5000/api/v1/knowledge-base/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "RAG_V2/nsbm_foc_data.pdf"}'
```

#### Get Memory (last 10 entries)
```bash
curl "http://localhost:5000/api/v1/memory?limit=10"
```

#### Clear Memory
```bash
curl -X DELETE http://localhost:5000/api/v1/memory
```

### Python `requests` Example

```python
import requests

BASE_URL = "http://localhost:5000/api/v1"

# Check health
health = requests.get(f"{BASE_URL}/health").json()
print(f"Status: {health['status']}")

# Submit query
response = requests.post(f"{BASE_URL}/query", json={
    "query": "I just got accepted and I'm so excited! What facilities does the campus have?",
    "top_k": 3
}).json()

print(f"Emotion: {response['emotion']} ({response['confidence']:.2f})")
print(f"Task Type: {response['task_type']}")
print(f"Response: {response['response']}")
print(f"Time: {response['processing_time_seconds']}s")
```

### JavaScript `fetch` Example

```javascript
const BASE_URL = "http://localhost:5000/api/v1";

// Submit query
const response = await fetch(`${BASE_URL}/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "What are the core subjects in Computer Science?",
    top_k: 3
  })
});

const data = await response.json();
console.log(`Emotion: ${data.emotion} (${data.confidence})`);
console.log(`Response: ${data.response}`);
```

---

## Rate Limiting & Performance Notes

- **First request** may be slower (~10-15s) due to model loading (DistilBERT + SentenceTransformer).
- Subsequent requests are faster (~2-5s) as models are cached in memory.
- The Ollama generation step typically takes 1-3s depending on response length.
- Memory usage increases with model loading. Consider subprocess isolation for production deployments (see `run_tests.py` for an example).

---

## Project Structure

```
.
├── api.py                    # ← REST API server (this documentation)
├── unified_system.py         # Original interactive CLI system
├── run_tests.py              # Automated test runner
├── testing_plan.md           # Test case specifications
├── requirements.txt          # Python dependencies
├── API_DOCUMENTATION.md      # This file
├── RAG_V2/
│   └── nsbm_foc_data.pdf     # Knowledge base source PDF
└── unified_chroma_store/     # ChromaDB persistent storage
    └── chroma.sqlite3
```
