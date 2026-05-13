import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

# Initialize ChromaDB client and collection
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_store"))
collection = client.get_or_create_collection(name="llm_knowledge")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def store_to_chroma(prompt, response, task_type):
    # Combine prompt and response for embedding
    text_to_embed = f"Prompt: {prompt}\nResponse: {response}"
    embedding = embedder.encode(text_to_embed).tolist()

    collection.add(
        documents=[text_to_embed],
        embeddings=[embedding],
        metadatas=[{"task_type": task_type}],
        ids=[str(uuid.uuid4())]
    )

    print("✅ Stored in ChromaDB")
