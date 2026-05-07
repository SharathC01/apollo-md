"""
retrieve.py — Semantic retrieval (RAG) over paper chunks.

Uses sentence-transformers "all-MiniLM-L6-v2" to embed chunk texts and
compute cosine similarity against a query. Supports optional global caching
so chunks and embeddings are not reloaded on every query.

Functions:
  embed_chunks(chunks) -> tuple[list, np.ndarray]
  retrieve(query, chunks, embeddings, top_k=5) -> list[dict]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

# Optional pre-loaded global state
_cached_chunks: list[dict] | None = None
_cached_embeddings: np.ndarray | None = None


def _get_model():
    """Lazy-load the SentenceTransformer model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_chunks(chunks: list[dict]) -> tuple[list, np.ndarray]:
    """
    Encode all chunk texts using the SentenceTransformer model.

    Returns (chunks, embeddings_array) where embeddings_array has shape
    (len(chunks), embedding_dim).
    """
    model = _get_model()
    texts = [c.get("text") or "" for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,
    )
    return chunks, embeddings


def retrieve(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """
    Find the top_k most semantically similar chunks to the query.

    Computes cosine similarity between the query embedding and all chunk
    embeddings. Adds a 'score' field (float in [0, 1]) to each returned chunk.
    Returns list sorted by score descending.
    """
    model = _get_model()
    q_emb = model.encode([query], convert_to_numpy=True)  # shape (1, dim)

    # Cosine similarity: dot(A, B) / (|A| * |B|)
    chunk_norms = np.linalg.norm(embeddings, axis=1)  # (N,)
    q_norm = float(np.linalg.norm(q_emb))
    denom = chunk_norms * q_norm + 1e-10
    scores = (embeddings @ q_emb.T).flatten() / denom  # (N,)

    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = dict(chunks[idx])
        chunk["score"] = float(scores[idx])
        results.append(chunk)

    return results


def preload(chunks: list[dict]) -> None:
    """
    Pre-embed chunks and store in global cache so retrieve() can be called
    without passing chunks/embeddings explicitly via the cached accessors.
    """
    global _cached_chunks, _cached_embeddings
    _cached_chunks, _cached_embeddings = embed_chunks(chunks)


def retrieve_cached(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve using globally cached chunks/embeddings (must call preload first).
    Raises RuntimeError if preload has not been called.
    """
    if _cached_chunks is None or _cached_embeddings is None:
        raise RuntimeError("Call preload(chunks) before retrieve_cached().")
    return retrieve(query, _cached_chunks, _cached_embeddings, top_k=top_k)


if __name__ == "__main__":
    # Quick smoke test: embed a handful of fake chunks and run a query
    sample_chunks = [
        {"text": "SOFA score predicts in-hospital mortality in ICU patients.", "source_file": "test.pdf", "page": 1},
        {"text": "Lactate levels are elevated in septic shock.", "source_file": "test.pdf", "page": 2},
        {"text": "Antibiotic timing is critical in sepsis management.", "source_file": "test.pdf", "page": 3},
        {"text": "qSOFA is a quick bedside tool for sepsis screening.", "source_file": "test.pdf", "page": 4},
        {"text": "Lymphocyte count correlates with 28-day mortality.", "source_file": "test.pdf", "page": 5},
    ]
    print("Embedding sample chunks...")
    chunks, emb = embed_chunks(sample_chunks)
    print(f"Embeddings shape: {emb.shape}")

    query = "SOFA score mortality prediction"
    results = retrieve(query, chunks, emb, top_k=3)
    print(f"\nTop 3 results for '{query}':")
    for r in results:
        print(f"  score={r['score']:.3f}  page={r['page']}  text={r['text'][:80]}")
