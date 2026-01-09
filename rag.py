# -*- coding: utf-8 -*-
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

DATA_PATH = Path("data") / "store_manual.txt"

# Embeddings (خفيف وممتاز كبداية)
from chromadb.utils import embedding_functions
import os

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)


# Persistent (يحفظ الفهرس على الهارد) - أفضل من Client() العادي
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="store_docs",
    embedding_function=embedding_function
)


def chunk_text(text: str, chunk_size_words: int = 220, overlap_words: int = 40):
    """Chunk by words with overlap for better retrieval."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size_words]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size_words - overlap_words)
    return chunks


def ingest_document(path: str = str(DATA_PATH)):
    """Index the document into Chroma. Safe to run multiple times."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Document is empty.")

    # تنظيف الفهرس القديم ثم إعادة إدخال (عشان التحديثات)
    try:
        collection.delete(where={})
    except Exception:
        pass

    chunks = chunk_text(text)

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )


def query_document(question: str, k: int = 3):
    """Return top-k relevant chunks."""
    results = collection.query(
        query_texts=[question],
        n_results=k
    )
    return results["documents"][0]
