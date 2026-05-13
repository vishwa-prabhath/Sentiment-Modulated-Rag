"""
Sentiment Modulated RAG: A Framework for Dynamic Response Generation
Uses Ollama for local LLM inference (Metal GPU acceleration).

used Llama 3.1 instead using Dolphin3.0-Llama3.2-3B 
"""
import os
import json
import uuid
import gc
import requests
import re
import collections

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Ollama model for generation (runs via Metal GPU, no PyTorch needed)
OLLAMA_MODEL = "llama3.1:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

CHUNK_SIZE = 300       
CHUNK_OVERLAP = 50     #

# 2.Database Setup (ChromaDB)

print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./unified_chroma_store")

# Two collections inside one unified database (as per interim report Section 9.2)
kb_collection = chroma_client.get_or_create_collection(name="knowledge_base")
memory_collection = chroma_client.get_or_create_collection(name="conversational_memory")


# 3. Knowledge Base Ingestion (Improved Chunking)

def ingest_pdf_to_kb(pdf_path, force_reingest=False):

    if kb_collection.count() > 0 and not force_reingest:
        print(f"Knowledge base already populated ({kb_collection.count()} chunks). Skipping ingestion.")
        return

    # Clear existing data if re-ingesting
    if force_reingest and kb_collection.count() > 0:
        print("Clearing old knowledge base for re-ingestion...")
        existing = kb_collection.get()
        if existing["ids"]:
            kb_collection.delete(ids=existing["ids"])

    print(f"Ingesting {pdf_path} into Knowledge Base (chunk_size={CHUNK_SIZE})...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Use RecursiveCharacterTextSplitter for intelligent chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )

    chunk_count = 0
    for i, page in enumerate(pages):
        text = page.page_content
        if not text.strip():
            continue

        # Split this page into smaller chunks
        chunks = splitter.split_text(text)
        for j, chunk in enumerate(chunks):
            if not chunk.strip() or len(chunk.strip()) < 20:
                continue  # Skip tiny fragments like nav menus
            embedding = embedder.encode(chunk).tolist()
            kb_collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": pdf_path, "page": i, "chunk": j}],
                ids=[f"doc_p{i}_c{j}"]
            )
            chunk_count += 1

    del embedder
    gc.collect()
    print(f"Ingestion complete. {chunk_count} chunks stored from {len(pages)} pages.")


# Ingest the FOC data from RAG_V2 repo
pdf_file = "RAG_V2/nsbm_foc_data.pdf"
if os.path.exists(pdf_file):
    ingest_pdf_to_kb(pdf_file)
else:
    print(f"Warning: {pdf_file} not found. Knowledge base will be empty.")

# 4. Core Pipeline Functions

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
    """
    Heuristic check: since
    DistilBERT was trained on emotional text (CARER dataset) and has no 'neutral'
    class, so it often misclassifies factual queries as 'anger'.
    """
    matches = FACTUAL_KEYWORDS.findall(text)
    # If 2+ factual keywords found, or query is short and starts with a question word
    if len(matches) >= 2:
        return True
    if len(text.split()) <= 12 and FACTUAL_KEYWORDS.search(text):
        return True
    return False


def infer_task_type(prompt):
    """
    Includes override heuristics to fix known limitations where DistilBERT's
    6-class emotion model (no 'neutral' class) misclassifies greetings and
    factual queries as 'anger' (Section 8.5 of interim report).
    """
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

    # Low confidence = ambiguous emotion → default to factual
    if confidence < 0.5:
        return "factual", emotion, confidence

    # Standard emotion → task type mapping
    if emotion in ["joy", "surprise"]:
        return "creative", emotion, confidence
    elif emotion in ["sadness", "fear", "love"]:
        return "emotional", emotion, confidence
    else:
        # "anger" with high confidence but NOT factual keywords → still factual
        # because DistilBERT's "anger" often just means "neutral/assertive"
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

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]


def generate_unified_response(user_query: str):
    """
    The main Sentiment Modulated RAG pipeline:
    """

    # Step 1: RAG - Retrieve factual context
    context = retrieve_context(user_query)

    # Step 2: HAG - Classify sentiment & select hyperparameters
    task_type, emotion, confidence = infer_task_type(user_query)
    target_hyperparams = dynamic_hyperparameter_selector(task_type)
    
    # Smooth these target parameters with the memory
    global hyperparameter_memory
    hyperparams = smooth_hyperparameters(target_hyperparams, hyperparameter_memory)
    hyperparameter_memory.append(hyperparams)

    # Step 3: Construct prompt with retrieved context
    context_snippet = context[:2000]
    prompt = (
        f"You are a helpful university assistant for NSBM Green University. "
        f"Answer the student's question based ONLY on the context provided below. "
        f"If the context does not contain the answer, say so honestly.\n\n"
        f"Context:\n{context_snippet}\n\n"
        f"Student Question: {user_query}\n\n"
        f"Answer:"
    )

    # Step 4: Generate with Ollama using dynamic hyperparameters
    response = generate_with_ollama(prompt, hyperparams)

    # Step 5: Log to conversational memory
    log_to_memory(user_query, response, task_type)

    return response, task_type, emotion, confidence, hyperparams


# ==========================================
# 5. Interactive Loop
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print(f"  Sentiment Modulated RAG System")
    print(f"  Model: {OLLAMA_MODEL} (via Ollama)")
    print(f"  KB Chunks: {kb_collection.count()}")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 55 + "\n")

    while True:
        user_input = input("Student Query: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        print("\nProcessing...")
        response, task_type, emotion, confidence, hyperparams = generate_unified_response(user_input)

        print(f"\n[Detected Emotion: {emotion} ({confidence:.2f})]")
        print(f"[Task Type: {task_type.upper()}]")
        print(f"[Dynamic Params: {hyperparams}]")
        print("-" * 55)
        print(f"AI Assistant: {response}")
        print("-" * 55 + "\n")
