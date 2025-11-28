# modules/vector_store.py
"""
Chroma-backed vector store helper for Vitality AI.

This version lazy-loads heavy libs to avoid import-time crashes (torch, sentence-transformers, chromadb).
It will:
 - Prefer local SBERT (sentence-transformers) if available.
 - Fall back to OpenAI embeddings if sentence-transformers is not available and OPENAI_API_KEY is set.
 - Lazily import chromadb only when init_store() is called.

Public API:
 - init_store(persist_directory=None, collection_name="vitality_reports") -> Collection
 - ingest_text(source_name, text, chunk_size=800, chunk_overlap=100) -> int
 - query(query_text, top_k=4) -> List[Dict[str, Any]]
 - clear_collection()
 - status() -> Dict[str, Any]
"""
from __future__ import annotations
import os
import warnings
from typing import List, Dict, Any, Optional

# Configurable defaults
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(os.getcwd(), "chroma_db"))
SBERT_MODEL_NAME = os.environ.get("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Internal singletons (initialized lazily)
_embedding_backend: Optional[str] = None  # "sbert" or "openai"
_embedding_model = None
_client = None
_collection = None

# Fallback simple splitter (used if langchain.text_splitter not available)
def _simple_text_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    length = len(text)
    while i < length:
        end = min(i + chunk_size, length)
        chunks.append(text[i:end])
        i += chunk_size - chunk_overlap
    return chunks

# Lazy load sentence-transformers or OpenAI embeddings
def _detect_embedding_backend():
    """
    Detect and set the embedding backend:
      - prefer sentence-transformers (SBERT) if available
      - else use OpenAI if OPENAI_API_KEY and openai package available
      - otherwise raise on use
    """
    global _embedding_backend
    if _embedding_backend:
        return _embedding_backend

    # Try sentence-transformers first (this may require torch)
    try:
        import sentence_transformers  # noqa: F401
        _embedding_backend = "sbert"
        return _embedding_backend
    except Exception:
        pass

    # Try OpenAI next
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key".upper())
    if openai_key:
        try:
            import openai  # type: ignore
            _embedding_backend = "openai"
            return _embedding_backend
        except Exception:
            pass

    _embedding_backend = None
    return None

def _get_embedding_model():
    """
    Returns a callable that accepts List[str] and returns List[List[float]].
    Initializes backend models lazily.
    """
    global _embedding_model

    if _embedding_model:
        return _embedding_model

    backend = _detect_embedding_backend()
    if backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(f"Failed to import sentence-transformers: {e}")

        model = SentenceTransformer(SBERT_MODEL_NAME)
        # wrapper: model.encode(list_of_texts, convert_to_numpy=True) -> numpy array
        def encode_fn(texts: List[str], **kwargs) -> List[List[float]]:
            arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return arr.tolist()
        _embedding_model = encode_fn
        return _embedding_model

    if backend == "openai":
        try:
            import openai  # type: ignore
        except Exception as e:
            raise RuntimeError(f"OpenAI package not usable: {e}")

        # ensure API key set
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key".upper())):
            raise RuntimeError("OPENAI_API_KEY not set for OpenAI embedding backend.")

        def encode_fn(texts: List[str], **kwargs) -> List[List[float]]:
            # OpenAI accepts a list of inputs; returns data with embeddings in same order
            resp = openai.Embedding.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
            embs = [item["embedding"] for item in resp["data"]]
            return embs

        _embedding_model = encode_fn
        return _embedding_model

    raise RuntimeError(
        "No embedding backend available. Install 'sentence-transformers' (requires torch) "
        "or set OPENAI_API_KEY and install 'openai' package to use OpenAI embeddings."
    )

# ----------- Chroma client helpers (lazy-loaded) -----------
def _ensure_chroma():
    """
    Lazy import and initialize chromadb client when needed.
    Returns chromadb module and Settings class.
    """
    try:
        import chromadb
        from chromadb.config import Settings  # type: ignore
        return chromadb, Settings
    except Exception as e:
        raise RuntimeError(f"chromadb is required for vector DB features: {e}")

# ----------- Text splitter helper (lazy attempt to use langchain) -----------
def _get_text_splitter(chunk_size: int = 800, chunk_overlap: int = 100):
    try:
        # try langchain splitter if available
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception:
        # fallback to simple splitter object with .split_text
        class _FallbackSplitter:
            def __init__(self, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            def split_text(self, text: str) -> List[str]:
                return _simple_text_split(text, self.chunk_size, self.chunk_overlap)
        return _FallbackSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# ---------------- Public API ----------------
def init_store(persist_directory: Optional[str] = None, collection_name: str = "vitality_reports"):
    """
    Initialize or open a persistent Chroma client and collection (lazy).
    Returns the Collection object.
    """
    global _client, _collection
    persist_dir = persist_directory or CHROMA_PERSIST_DIR
    os.makedirs(persist_dir, exist_ok=True)

    chroma, Settings = _ensure_chroma()

    # Use the recommended settings for local duckdb+parquet persistence where available
    try:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        _client = chroma.Client(settings)
    except Exception:
        # fallback to a simpler Settings constructor or defaults
        try:
            settings = Settings(persist_directory=persist_dir)
            _client = chroma.Client(settings)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chroma client: {e}")

    # create or get collection
    try:
        _collection = _client.get_collection(name=collection_name)
    except Exception:
        _collection = _client.create_collection(name=collection_name)
    return _collection

def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Return embeddings for a list of texts (as lists of floats).
    """
    encoder = _get_embedding_model()
    return encoder(texts)

def ingest_text(source_name: str, text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> int:
    """
    Split input text into chunks, compute embeddings, and add to Chroma collection.
    Returns number_of_chunks_indexed (int)
    """
    global _collection
    if not text:
        return 0

    if _collection is None:
        init_store()

    splitter = _get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        chunks = splitter.split_text(text)
    except Exception:
        chunks = _simple_text_split(text, chunk_size, chunk_overlap)

    if not chunks:
        return 0

    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    try:
        embeddings = _embed_texts(chunks)
    except Exception as e:
        warnings.warn(f"Embedding generation failed: {e}")
        raise

    try:
        _collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    except Exception:
        # try delete+add (upsert-like)
        try:
            _collection.delete(ids=ids)
            _collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
        except Exception as ex:
            warnings.warn(f"Failed to add documents to Chroma collection: {ex}")
            raise

    try:
        _client.persist()
    except Exception:
        pass

    return len(chunks)

def query(query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Query the vector store and return list of results with keys {text, metadata, distance}.
    """
    global _collection
    if _collection is None:
        init_store()

    if not query_text:
        return []

    try:
        encoder = _get_embedding_model()
        q_emb = encoder([query_text])[0]
    except Exception as e:
        warnings.warn(f"Failed to embed query text: {e}")
        return []

    try:
        # Note: chroma client APIs may vary by version; this uses query_embeddings param
        results = _collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    except Exception as e:
        warnings.warn(f"Chroma query failed: {e}")
        return []

    try:
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [{"text": d, "metadata": m, "distance": dist} for d, m, dist in zip(docs, metadatas, distances)]
    except Exception:
        return []

def clear_collection():
    """
    Delete all documents in the collection. Use with caution.
    """
    global _collection, _client
    if _collection is None:
        init_store()
    try:
        _collection.delete()
        if _client:
            _client.persist()
    except Exception as e:
        warnings.warn(f"Failed to clear Chroma collection: {e}")

def status() -> Dict[str, Any]:
    return {
        "chroma_persist_dir": CHROMA_PERSIST_DIR,
        "sbert_model": SBERT_MODEL_NAME,
        "embedding_backend": _embedding_backend,
        "vector_store_available": _collection is not None
    }