# rag.py - RAG implementation using OpenAI embeddings and Chroma
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= Configuration =========
COLLECTION_NAME = "store_docs"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# OpenAI Embedding Function with latest model
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

# Chroma Client
try:
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    logger.info(f"Chroma client initialized with persist directory: {PERSIST_DIR}")
except Exception as e:
    logger.error(f"Failed to initialize Chroma client: {e}")
    raise

# Collection
try:
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )
    logger.info(f"Collection '{COLLECTION_NAME}' ready")
except Exception as e:
    logger.error(f"Failed to create/get collection: {e}")
    raise


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap for better retrieval.

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").strip()
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to break at sentence boundaries if possible
        if end < text_length:
            # Look for sentence endings within the last 100 chars of chunk
            chunk_candidate = text[start:end]
            last_period = chunk_candidate.rfind('.')
            last_newline = chunk_candidate.rfind('\n')

            break_point = max(last_period, last_newline)
            if break_point > len(chunk_candidate) - 100:  # Within last 100 chars
                end = start + break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(0, end - overlap)

    logger.info(f"Text chunked into {len(chunks)} chunks")
    return chunks


def ingest_document(
    text: str | None = None,
    doc_id: str = "store_manual",
    metadata: dict | None = None
) -> int:
    if metadata is None:
        metadata = {}

    # لو مفيش نص اتبعت، اقرا تلقائي من ملف الدليل
    if text is None:
        with open("data/store_manual.txt", "r", encoding="utf-8") as f:
            text = f.read()

    # مهم: استخدم اسم الدالة اللي عندك فعلاً
    # لو عندك chunk_text استخدمها، لو عندك _chunk_text استخدمها
    chunks = chunk_text(text)  # <- لو اسم الدالة عندك chunk_text

    if not chunks:
        return 0

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

    try:
        existing = collection.get(where={"doc_id": doc_id})
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


    """
    Ingest a text document into the Chroma collection.

    Args:
        text: The document text to ingest
        doc_id: Unique identifier for the document
        metadata: Additional metadata for the document
        chunk_size: Size of text chunks
        overlap: Overlap between chunks

    Returns:
        Number of chunks added
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for ingestion")
        return 0

    if metadata is None:
        metadata = {}

    # Add timestamp and other default metadata
    metadata.update({
        "ingested_at": datetime.now().isoformat(),
        "doc_length": len(text)
    })

    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        logger.warning("No chunks generated from text")
        return 0

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {**metadata, "doc_id": doc_id, "chunk_index": i, "total_chunks": len(chunks)}
        for i in range(len(chunks))
    ]

    try:
        # Remove existing chunks for this doc_id to avoid duplicates
        existing = collection.get(where={"doc_id": doc_id})
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
            logger.info(f"Removed {len(existing['ids'])} existing chunks for doc_id: {doc_id}")

        # Add new chunks
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

        logger.info(f"Successfully ingested {len(chunks)} chunks for document: {doc_id}")
        return len(chunks)

    except Exception as e:
        logger.error(f"Failed to ingest document {doc_id}: {e}")
        raise


def query_documents(
    query: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
    include_distances: bool = False
) -> Dict[str, Any]:
    """
    Query the document collection and return relevant chunks.

    Args:
        query: The search query
        top_k: Number of top results to return
        where: Metadata filters for the query
        include_distances: Whether to include similarity distances

    Returns:
        Dictionary containing query results
    """
    if not query or not query.strip():
        logger.warning("Empty query provided")
        return {"documents": [], "metadatas": [], "distances": []}

    try:
        include = ["documents", "metadatas"]
        if include_distances:
            include.append("distances")

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=include
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if include_distances else []

        logger.info(f"Query returned {len(docs)} results")

        return {
            "documents": docs,
            "metadatas": metadatas,
            "distances": distances,
            "query": query
        }

    except Exception as e:
        logger.error(f"Failed to query documents: {e}")
        raise


def get_context_from_query(query: str, top_k: int = 5) -> str:
    """
    Get formatted context string from a query.

    Args:
        query: The search query
        top_k: Number of top results to include in context

    Returns:
        Formatted context string
    """
    results = query_documents(query, top_k=top_k)

    docs = results.get("documents", [])
    if not docs:
        return ""

    # Format context with separators
    context_parts = []
    for i, doc in enumerate(docs):
        metadata = results.get("metadatas", [])[i] if i < len(results.get("metadatas", [])) else {}
        doc_id = metadata.get("doc_id", "unknown")
        chunk_index = metadata.get("chunk_index", i)

        context_parts.append(f"[Document: {doc_id}, Chunk: {chunk_index}]\n{doc}")

    return "\n\n---\n\n".join(context_parts)


# Legacy function for backward compatibility
def query_document(query: str, top_k: int = 4) -> str:
    """
    Legacy function - use get_context_from_query instead.
    """
    return get_context_from_query(query, top_k)


def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the collection.

    Returns:
        Dictionary with collection statistics
    """
    try:
        count = collection.count()
        return {
            "total_chunks": count,
            "collection_name": COLLECTION_NAME,
            "persist_directory": PERSIST_DIR
        }
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return {"error": str(e)}