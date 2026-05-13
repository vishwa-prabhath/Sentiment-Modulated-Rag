"""
REST API for Sentiment Modulated RAG System
============================================
Provides a Flask-based REST API with Swagger/OpenAPI documentation
for the Sentiment Modulated RAG Framework.

Endpoints:
  - POST /api/v1/query           → Main query pipeline
  - GET  /api/v1/health          → Health check (Ollama + ChromaDB)
  - POST /api/v1/ingest          → Ingest a PDF into the knowledge base
  - GET  /api/v1/knowledge-base/stats → Knowledge base statistics
  - GET  /api/v1/memory          → Retrieve conversational memory entries
  - DELETE /api/v1/memory        → Clear conversational memory
"""

import os
import json
import uuid
import gc
import re
import time
import traceback
import collections

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests as http_requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace

from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# Configuration
# ==========================================

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./unified_chroma_store")
API_PORT = int(os.environ.get("API_PORT", 5001))

# Chunking configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Keywords that strongly indicate a neutral/factual query (no emotion)
FACTUAL_KEYWORDS = re.compile(
    r'\b(what|when|where|which|how\s+(long|many|much|do|does|is|are|can)|'
    r'tell me|list|explain|describe|define|duration|deadline|policy|fee|'
    r'program|course|degree|subject|department|schedule|requirement|enroll|'
    r'admission|credit|semester|gpa|grade|transcript|certificate)\b',
    re.IGNORECASE
)

# Greetings and casual phrases that DistilBERT misclassifies as "anger"
GREETING_PATTERN = re.compile(
    r'^\s*(hi|hello|hey|hii+|helo+|good\s*(morning|afternoon|evening|day)|'
    r'greetings|sup|yo|howdy|thanks|thank\s*you|ok|okay|bye|goodbye|'
    r'see\s*you|cheers|welcome|please\s*help)\s*[!?.]*\s*$',
    re.IGNORECASE
)

def is_greeting(text):
    """Detect greetings and casual messages that have no real emotion."""
    return bool(GREETING_PATTERN.match(text.strip()))

def is_likely_factual(text):
    """Heuristic: if query is dominated by question words and academic terms, it's factual."""
    matches = FACTUAL_KEYWORDS.findall(text)
    if len(matches) >= 2:
        return True
    if len(text.split()) <= 12 and FACTUAL_KEYWORDS.search(text):
        return True
    return False

# ==========================================
# Flask App & Swagger Setup
# ==========================================

app = Flask(__name__, template_folder="templates")
CORS(app)

# Configure Swagger / OpenAPI documentation
api = Api(
    app,
    version="1.0.0",
    title="Sentiment Modulated RAG API",
    description=(
        "A REST API for the Sentiment Modulated RAG Framework.\n\n"
        "This system combines **RAG** (Retrieval-Augmented Generation) for factual grounding "
        "with **HAG** (Hyperparameter-Augmented Generation) for sentiment-driven response tuning.\n\n"
        "The pipeline:\n"
        "1. **Retrieve** relevant context from a ChromaDB knowledge base\n"
        "2. **Classify** user sentiment using DistilBERT emotion detection\n"
        "3. **Select** dynamic hyperparameters based on detected emotion\n"
        "4. **Generate** a response via Ollama with tuned parameters\n"
        "5. **Log** the interaction to conversational memory"
    ),
    doc="/docs",
    prefix="/api/v1",
)

# Namespaces for logical grouping
ns_query = Namespace("query", description="Main query pipeline operations")
ns_health = Namespace("health", description="System health checks")
ns_kb = Namespace("knowledge-base", description="Knowledge base management")
ns_memory = Namespace("memory", description="Conversational memory operations")

api.add_namespace(ns_query)
api.add_namespace(ns_health)
api.add_namespace(ns_kb)
api.add_namespace(ns_memory)

# ==========================================
# Request/Response Models (for Swagger docs)
# ==========================================

query_input = ns_query.model("QueryInput", {
    "query": fields.String(
        required=True,
        description="The student's question or message",
        example="What are the core subjects offered in the Computer Science degree program at NSBM?"
    ),
    "top_k": fields.Integer(
        required=False,
        description="Number of context documents to retrieve (default: 3)",
        default=3,
        example=3
    ),
})

hyperparams_model = api.model("Hyperparameters", {
    "temperature": fields.Float(description="Sampling temperature", example=0.3),
    "top_p": fields.Float(description="Nucleus sampling threshold", example=0.85),
    "top_k": fields.Integer(description="Top-K sampling", example=40),
    "repetition_penalty": fields.Float(description="Repetition penalty", example=1.2),
})

query_response = ns_query.model("QueryResponse", {
    "response": fields.String(description="Generated AI response"),
    "emotion": fields.String(description="Detected emotion label", example="joy"),
    "confidence": fields.Float(description="Emotion detection confidence", example=0.92),
    "task_type": fields.String(
        description="Mapped task type (factual, creative, emotional)",
        example="factual",
        enum=["factual", "creative", "emotional"]
    ),
    "hyperparameters": fields.Nested(hyperparams_model, description="Applied generation hyperparameters"),
    "context_retrieved": fields.Boolean(description="Whether relevant context was found"),
    "processing_time_seconds": fields.Float(description="Total processing time in seconds", example=2.5),
})

error_model = api.model("Error", {
    "error": fields.String(description="Error message"),
    "details": fields.String(description="Detailed error information"),
})

ingest_input = ns_kb.model("IngestInput", {
    "pdf_path": fields.String(
        required=True,
        description="Path to the PDF file to ingest",
        example="RAG_V2/nsbm_foc_data.pdf"
    ),
})

ingest_response = ns_kb.model("IngestResponse", {
    "message": fields.String(description="Status message"),
    "pages_ingested": fields.Integer(description="Number of pages ingested"),
})

kb_stats_response = ns_kb.model("KBStatsResponse", {
    "total_documents": fields.Integer(description="Total documents in the knowledge base"),
    "collection_name": fields.String(description="ChromaDB collection name"),
})

memory_entry = ns_memory.model("MemoryEntry", {
    "id": fields.String(description="Memory entry ID"),
    "document": fields.String(description="Stored prompt/response text"),
    "task_type": fields.String(description="Task type classification"),
})

memory_response = ns_memory.model("MemoryResponse", {
    "total_entries": fields.Integer(description="Total memory entries"),
    "entries": fields.List(fields.Nested(memory_entry)),
})

health_response = ns_health.model("HealthResponse", {
    "status": fields.String(description="Overall system status", example="healthy"),
    "ollama": fields.String(description="Ollama server status"),
    "ollama_model": fields.String(description="Configured Ollama model"),
    "chromadb": fields.String(description="ChromaDB status"),
    "knowledge_base_docs": fields.Integer(description="Number of documents in KB"),
    "memory_entries": fields.Integer(description="Number of memory entries"),
})

# ==========================================
# ChromaDB Initialization
# ==========================================

print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
kb_collection = chroma_client.get_or_create_collection(name="knowledge_base")
memory_collection = chroma_client.get_or_create_collection(name="conversational_memory")
print(f"ChromaDB ready. KB docs: {kb_collection.count()}, Memory entries: {memory_collection.count()}")

# ==========================================
# Core Pipeline Functions
# ==========================================

def infer_task_type(prompt):
    """HAG: Classify the user's emotion and map to a task type.
    Includes override heuristics to fix DistilBERT's anger misclassification."""
    # OVERRIDE 1: Greetings → treat as friendly/creative (not anger)
    if is_greeting(prompt):
        return "creative", "neutral", 1.0

    classifier = hf_pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion"
    )
    result = classifier(prompt)[0]
    emotion = result['label']
    confidence = result['score']
    del classifier
    gc.collect()

    # OVERRIDE 2: Factual keyword detection
    if is_likely_factual(prompt):
        return "factual", emotion, confidence

    # OVERRIDE 3: Very short messages (≤3 words) with no strong emotion → neutral
    if len(prompt.split()) <= 3 and emotion == "anger":
        return "factual", "neutral", confidence

    # Low confidence = ambiguous → default to factual
    if confidence < 0.5:
        return "factual", emotion, confidence

    if emotion in ["joy", "surprise"]:
        return "creative", emotion, confidence
    elif emotion in ["sadness", "fear", "love"]:
        return "emotional", emotion, confidence
    else:
        return "factual", emotion, confidence


def dynamic_hyperparameter_selector(task_type: str):
    """HAG: Map task type to generation hyperparameters."""
    if task_type == "factual":
        return {"temperature": 0.3, "top_p": 0.85, "top_k": 40, "repetition_penalty": 1.2}
    elif task_type == "creative":
        return {"temperature": 0.85, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.0}
    elif task_type == "emotional":
        return {"temperature": 0.6, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1}
    else:
        return {"temperature": 0.5, "top_p": 0.85, "top_k": 60, "repetition_penalty": 1.1}


CURRENT_TASK_WEIGHT = 0.7
PREVIOUS_MEMORY_WEIGHT = 0.3
hyperparameter_memory = collections.deque(maxlen=5)
hyperparameter_memory.append({"temperature": 0.5, "top_p": 0.85, "top_k": 60, "repetition_penalty": 1.1})

def smooth_hyperparameters(current_task_params: dict, memory: collections.deque) -> dict:
    if not memory:
        return current_task_params

    avg_memory_params = {
        "temperature": 0.0,
        "top_p": 0.0,
        "top_k": 0.0,
        "repetition_penalty": 0.0
    }

    for past_params in memory:
        for key in avg_memory_params:
            avg_memory_params[key] += past_params[key]

    for key in avg_memory_params:
        avg_memory_params[key] /= len(memory)

    # Apply the weighted average to smooth the parameters
    smoothed_params = {
        "temperature": (current_task_params["temperature"] * CURRENT_TASK_WEIGHT) +
                       (avg_memory_params["temperature"] * PREVIOUS_MEMORY_WEIGHT),
        "top_p": (current_task_params["top_p"] * CURRENT_TASK_WEIGHT) +
                 (avg_memory_params["top_p"] * PREVIOUS_MEMORY_WEIGHT),
        "top_k": int((current_task_params["top_k"] * CURRENT_TASK_WEIGHT) +
                     (avg_memory_params["top_k"] * PREVIOUS_MEMORY_WEIGHT)), 
        "repetition_penalty": (current_task_params["repetition_penalty"] * CURRENT_TASK_WEIGHT) +
                              (avg_memory_params["repetition_penalty"] * PREVIOUS_MEMORY_WEIGHT)
    }

    # Add constraints to ensure parameters stay within reasonable bounds
    smoothed_params["temperature"] = max(0.1, min(smoothed_params["temperature"], 1.0))
    smoothed_params["top_p"] = max(0.1, min(smoothed_params["top_p"], 1.0))
    smoothed_params["top_k"] = max(1, min(smoothed_params["top_k"], 100)) # bounded for Ollama
    smoothed_params["repetition_penalty"] = max(1.0, min(smoothed_params["repetition_penalty"], 2.0))

    return smoothed_params


def retrieve_context(query, top_k=5):
    """RAG: Retrieve relevant chunks from the knowledge base via cosine similarity."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedder.encode(query).tolist()
    del embedder
    gc.collect()

    results = kb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if not results['documents'] or not results['documents'][0]:
        return ""

    context = "\n---\n".join(results['documents'][0])
    return context


def log_to_memory(prompt, response, task_type):
    """Save the interaction to the conversational memory vector store."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    text_to_embed = f"Prompt: {prompt}\nResponse: {response}"
    embedding = embedder.encode(text_to_embed).tolist()
    del embedder
    gc.collect()

    memory_collection.add(
        documents=[text_to_embed],
        embeddings=[embedding],
        metadatas=[{"task_type": task_type}],
        ids=[str(uuid.uuid4())]
    )


def generate_with_ollama(prompt, hyperparams):
    """Generate a response using Ollama's local API with dynamic hyperparameters."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": hyperparams["temperature"],
            "top_p": hyperparams["top_p"],
            "top_k": hyperparams["top_k"],
            "repeat_penalty": hyperparams["repetition_penalty"],
            "num_predict": 300
        }
    }

    response = http_requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]


def generate_unified_response(user_query: str, top_k: int = 3):
    """
    The main Sentiment Modulated RAG pipeline:
      1. RAG: Retrieve factual context from ChromaDB
      2. HAG: Classify sentiment → select dynamic hyperparameters
      3. Generate response via Ollama with dynamic tuning
      4. Log to conversational memory
    """
    context = retrieve_context(user_query, top_k=top_k)
    task_type, emotion, confidence = infer_task_type(user_query)
    target_hyperparams = dynamic_hyperparameter_selector(task_type)

    global hyperparameter_memory
    hyperparams = smooth_hyperparameters(target_hyperparams, hyperparameter_memory)
    hyperparameter_memory.append(hyperparams)

    context_snippet = context[:2000]
    
    if task_type == "emotional":
        system_instruction = (
            "You are a warmly empathetic, supportive student guidance counselor for NSBM Green University. "
            "The student is experiencing distress, disappointment, or anxiety. "
            "First, acknowledge their emotions kindly and provide a comforting, supportive validation statement. "
            "Then, clearly present the factual procedures or options extracted ONLY from the context below."
        )
    elif task_type == "creative":
        system_instruction = (
            "You are an upbeat, enthusiastic student life advisor for NSBM Green University. "
            "Provide an engaging, welcoming answer to the student's inquiry using strictly the factual context below."
        )
    else:
        system_instruction = (
            "You are an official academic assistant for NSBM Green University. "
            "Provide a highly professional, direct, and clearly structured answer based strictly on the context below."
        )

    prompt = (
        f"{system_instruction}\n"
        f"Do not invent policies or halluncinate details outside the provided text. If the answer is not present, state so honestly.\n\n"
        f"Context:\n{context_snippet}\n\n"
        f"Student Question: {user_query}\n\n"
        f"Response:"
    )

    response = generate_with_ollama(prompt, hyperparams)
    log_to_memory(user_query, response, task_type)

    return response, task_type, emotion, confidence, hyperparams, bool(context.strip())


# ==========================================
# API Endpoints
# ==========================================

@ns_health.route("")
class HealthCheck(Resource):
    @ns_health.doc("health_check")
    @ns_health.marshal_with(health_response)
    def get(self):
        """Check system health (Ollama server, ChromaDB, model availability)"""
        result = {
            "status": "healthy",
            "ollama": "unknown",
            "ollama_model": OLLAMA_MODEL,
            "chromadb": "unknown",
            "knowledge_base_docs": 0,
            "memory_entries": 0,
        }

        # Check Ollama
        try:
            resp = http_requests.get("http://localhost:11434/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                result["ollama"] = "connected (model available)"
            else:
                result["ollama"] = f"connected (model '{OLLAMA_MODEL}' not found, available: {models})"
                result["status"] = "degraded"
        except Exception as e:
            result["ollama"] = f"disconnected: {str(e)}"
            result["status"] = "unhealthy"

        # Check ChromaDB
        try:
            result["knowledge_base_docs"] = kb_collection.count()
            result["memory_entries"] = memory_collection.count()
            result["chromadb"] = "connected"
        except Exception as e:
            result["chromadb"] = f"error: {str(e)}"
            result["status"] = "unhealthy"

        return result


@ns_query.route("")
class QueryPipeline(Resource):
    @ns_query.doc("submit_query")
    @ns_query.expect(query_input, validate=True)
    @ns_query.response(200, "Success", query_response)
    @ns_query.response(400, "Validation Error", error_model)
    @ns_query.response(500, "Internal Server Error", error_model)
    @ns_query.response(503, "Service Unavailable", error_model)
    def post(self):
        """
        Submit a query to the Sentiment Modulated RAG pipeline.

        This endpoint runs the full pipeline:
        1. **RAG Retrieval** — Retrieves relevant context from the ChromaDB knowledge base
        2. **Sentiment Classification** — Uses DistilBERT to detect the user's emotion
        3. **Hyperparameter Selection** — Maps the detected emotion to generation parameters
        4. **Response Generation** — Generates a response via Ollama with tuned hyperparameters
        5. **Memory Logging** — Stores the interaction in conversational memory
        """
        data = request.json
        user_query = data.get("query", "").strip()
        top_k = data.get("top_k", 3)

        if not user_query:
            return {"error": "Query cannot be empty", "details": "Provide a non-empty 'query' field."}, 400

        if not isinstance(top_k, int) or top_k < 1 or top_k > 10:
            return {"error": "Invalid top_k", "details": "top_k must be an integer between 1 and 10."}, 400

        try:
            start_time = time.time()
            response, task_type, emotion, confidence, hyperparams, context_found = \
                generate_unified_response(user_query, top_k=top_k)
            elapsed = time.time() - start_time

            return {
                "response": response,
                "emotion": emotion,
                "confidence": round(confidence, 4),
                "task_type": task_type,
                "hyperparameters": hyperparams,
                "context_retrieved": context_found,
                "processing_time_seconds": round(elapsed, 2),
            }

        except http_requests.exceptions.ConnectionError:
            return {
                "error": "Ollama server unreachable",
                "details": "Make sure Ollama is running: 'ollama serve'"
            }, 503
        except Exception as e:
            return {
                "error": "Pipeline failed",
                "details": traceback.format_exc()
            }, 500


@ns_kb.route("/stats")
class KBStats(Resource):
    @ns_kb.doc("kb_stats")
    @ns_kb.marshal_with(kb_stats_response)
    def get(self):
        """Get knowledge base statistics"""
        return {
            "total_documents": kb_collection.count(),
            "collection_name": "knowledge_base",
        }


@ns_kb.route("/ingest")
class KBIngest(Resource):
    @ns_kb.doc("ingest_pdf")
    @ns_kb.expect(ingest_input, validate=True)
    @ns_kb.response(200, "Success", ingest_response)
    @ns_kb.response(400, "Validation Error", error_model)
    @ns_kb.response(500, "Internal Server Error", error_model)
    def post(self):
        """
        Ingest a PDF document into the knowledge base.

        The PDF is split into small ~300-character chunks using
        RecursiveCharacterTextSplitter, each embedded using
        `all-MiniLM-L6-v2`, and stored in the ChromaDB `knowledge_base` collection.
        """
        data = request.json
        pdf_path = data.get("pdf_path", "").strip()

        if not pdf_path:
            return {"error": "pdf_path is required", "details": "Provide the path to a PDF file."}, 400

        if not os.path.exists(pdf_path):
            return {"error": "File not found", "details": f"'{pdf_path}' does not exist."}, 400

        try:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", ", ", " "],
            )
            count = 0

            for i, page in enumerate(pages):
                text = page.page_content
                if not text.strip():
                    continue
                chunks = splitter.split_text(text)
                for j, chunk in enumerate(chunks):
                    if not chunk.strip() or len(chunk.strip()) < 20:
                        continue
                    embedding = embedder.encode(chunk).tolist()
                    doc_id = f"doc_{pdf_path}_p{i}_c{j}_{uuid.uuid4().hex[:8]}"
                    kb_collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[{"source": pdf_path, "page": i, "chunk": j}],
                        ids=[doc_id]
                    )
                    count += 1

            del embedder
            gc.collect()

            return {
                "message": f"Successfully ingested {count} chunks from '{pdf_path}'.",
                "pages_ingested": count,
            }

        except Exception as e:
            return {"error": "Ingestion failed", "details": traceback.format_exc()}, 500


@ns_memory.route("")
class MemoryOperations(Resource):
    @ns_memory.doc("get_memory")
    @ns_memory.param("limit", "Max entries to return (default 20)", type=int, default=20)
    @ns_memory.response(200, "Success", memory_response)
    def get(self):
        """Retrieve recent conversational memory entries"""
        limit = request.args.get("limit", 20, type=int)
        limit = min(max(1, limit), 100)  # Clamp between 1 and 100

        try:
            count = memory_collection.count()
            if count == 0:
                return {"total_entries": 0, "entries": []}

            results = memory_collection.get(
                limit=limit,
                include=["documents", "metadatas"]
            )

            entries = []
            for i in range(len(results["ids"])):
                entries.append({
                    "id": results["ids"][i],
                    "document": results["documents"][i] if results["documents"] else "",
                    "task_type": results["metadatas"][i].get("task_type", "unknown") if results["metadatas"] else "unknown",
                })

            return {"total_entries": count, "entries": entries}

        except Exception as e:
            return {"error": "Failed to retrieve memory", "details": str(e)}, 500

    @ns_memory.doc("clear_memory")
    @ns_memory.response(200, "Memory cleared")
    @ns_memory.response(500, "Error", error_model)
    def delete(self):
        """Clear all conversational memory entries"""
        global memory_collection
        try:
            chroma_client.delete_collection(name="conversational_memory")
            memory_collection = chroma_client.get_or_create_collection(name="conversational_memory")
            return {"message": "Conversational memory cleared successfully."}
        except Exception as e:
            return {"error": "Failed to clear memory", "details": str(e)}, 500

# Serve the chat UI at the root
@app.route("/")
def index():
    return render_template("index.html")


# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    # Auto-ingest the default PDF if KB is empty (using chunked ingestion)
    pdf_file = "RAG_V2/nsbm_foc_data.pdf"
    if os.path.exists(pdf_file) and kb_collection.count() == 0:
        print(f"Auto-ingesting {pdf_file} with chunking...")
        try:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", ", ", " "],
            )
            count = 0
            for i, page in enumerate(pages):
                text = page.page_content
                if not text.strip():
                    continue
                chunks = splitter.split_text(text)
                for j, chunk in enumerate(chunks):
                    if not chunk.strip() or len(chunk.strip()) < 20:
                        continue
                    embedding = embedder.encode(chunk).tolist()
                    kb_collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[{"source": pdf_file, "page": i, "chunk": j}],
                        ids=[f"doc_p{i}_c{j}"]
                    )
                    count += 1
            del embedder
            gc.collect()
            print(f"Ingestion complete. {count} chunks stored.")
        except Exception as e:
            print(f"Warning: Auto-ingestion failed: {e}")

    print(f"\n{'='*55}")
    print(f"  Sentiment Modulated RAG API")
    print(f"  Chat UI:      http://localhost:{API_PORT}/")
    print(f"  Swagger Docs: http://localhost:{API_PORT}/docs")
    print(f"  API Base:     http://localhost:{API_PORT}/api/v1")
    print(f"{'='*55}\n")

    app.run(host="0.0.0.0", port=API_PORT, debug=True)
