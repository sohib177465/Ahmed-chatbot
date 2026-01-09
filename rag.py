# rag.py
import os
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

# ========= إعدادات عامة =========
COLLECTION_NAME = "store_docs"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")  # محليًا فقط (Railway غالبًا ephemeral)

# Embedding باستخدام OpenAI (خفيف جدًا)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)

# Chroma Client (PersistentClient محليًا - على Railway ممكن يبقى مؤقت)
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

# Collection واحدة للمستندات
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef
)


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """تقسيم النص لقطع صغيرة عشان البحث يبقى أدق."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def ingest_document(
    text: str,
    doc_id: str = "store_manual",
    metadata: Dict[str, Any] | None = None
) -> int:
    """
    يدخل مستند نصي (زي دليل المتجر) إلى Chroma.
    بيرجع عدد الـ chunks اللي اتضافت.
    """
    if metadata is None:
        metadata = {}

    chunks = _chunk_text(text)
    if not chunks:
        return 0

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

    # مهم: لو بتعمل ingest أكتر من مرة لنفس doc_id
    # امسح القديم الأول لتجنب تكرار
    try:
        existing = collection.get(where={"doc_id": doc_id})
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)


def query_document(query: str, top_k: int = 4) -> str:
    """
    يبحث في المستندات ويرجع أفضل سياق (context) كنص واحد.
    """
    query = (query or "").strip()
    if not query:
        return ""

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = (results.get("documents") or [[]])[0]
    if not docs:
        return ""

    # دمج أفضل النتائج في سياق واحد
    context = "\n\n---\n\n".join(docs)
    return context
