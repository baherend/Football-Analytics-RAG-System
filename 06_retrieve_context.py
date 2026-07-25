"""
06_retrieve_context.py — Stage 6: Context Retrieval

Combines BM25 lexical retrieval and dense embedding retrieval using
Reciprocal Rank Fusion (RRF). Builds formatted context for the LLM.

Pipeline:
    User Query -> BM25 Search -> Dense Search -> RRF Merge -> Top-K -> Context

Input: user query
Output: formatted context string + source list
"""

from __future__ import annotations

import json
import pickle
import re
from importlib import import_module
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDICES_DIR = Path("output/indices")
CHUNKS_PATH = Path("output/chunks.json")
CHROMA_DIR = Path("output/chroma_db")
COLLECTION_NAME = "wc2022_documents"
MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60  # RRF constant from the original paper


# ---------------------------------------------------------------------------
# Index Loading (cached)
# ---------------------------------------------------------------------------

_bm25_cache = None
_chunks_cache = None
_model_cache = None


def _load_bm25():
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache
    bm25_path = INDICES_DIR / "bm25.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            _bm25_cache = pickle.load(f)
    return _bm25_cache


def _load_chunks():
    global _chunks_cache
    if _chunks_cache is not None:
        return _chunks_cache
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            _chunks_cache = json.load(f)
    else:
        _chunks_cache = import_module("03_chunking").chunks
    return _chunks_cache


def _get_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    from sentence_transformers import SentenceTransformer
    _model_cache = SentenceTransformer(MODEL_NAME)
    return _model_cache


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


# ---------------------------------------------------------------------------
# BM25 Retrieval
# ---------------------------------------------------------------------------

def bm25_search(query: str, k: int = 20) -> list[dict]:
    """Lexical retrieval using BM25."""
    bm25 = _load_bm25()
    chunks = _load_chunks()
    if bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(query))
    top_indices = scores.argsort()[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] > 0:
            chunk = chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    "document_id": chunk["document_id"],
                    "level": chunk.get("level", "unknown"),
                    "match_id": chunk.get("match_id"),
                    "player_name": chunk.get("player_name"),
                    "team_name": chunk.get("team_name"),
                },
                "score": float(scores[idx]),
                "rank": rank + 1,
                "source": "bm25",
            })
    return results


# ---------------------------------------------------------------------------
# Dense Retrieval (ChromaDB)
# ---------------------------------------------------------------------------

def dense_search(query: str, k: int = 20) -> list[dict]:
    """Dense retrieval using ChromaDB."""
    import chromadb

    model = _get_model()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = max(0, 1 - distance)
        formatted.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": score,
            "rank": i + 1,
            "source": "dense",
        })
    return formatted


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(result_sets: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Merge multiple result sets using RRF."""
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for result_set in result_sets:
        for result in result_set:
            chunk_id = result["chunk_id"]
            rank = result["rank"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

    merged = []
    for chunk_id, rrf_score in rrf_scores.items():
        result = chunk_data[chunk_id].copy()
        result["rrf_score"] = rrf_score
        merged.append(result)

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Context Building
# ---------------------------------------------------------------------------

def build_context(question: str, k: int = 4, max_sources: int = 3) -> tuple[str, list[dict]]:
    """
    Retrieve and format context for a question.

    Returns (context_string, list_of_source_dicts).
    """
    # Retrieve from both indices
    bm25_results = bm25_search(question, k=k * 4)
    dense_results = dense_search(question, k=k * 4)

    # Fuse
    merged = reciprocal_rank_fusion([bm25_results, dense_results])

    # Deduplicate and select top sources
    selected = []
    seen_documents = set()

    for row in merged:
        if row["score"] <= 0:
            continue
        doc_id = row["metadata"].get("document_id", row["chunk_id"])
        if doc_id in seen_documents:
            continue
        selected.append(row)
        seen_documents.add(doc_id)
        if len(selected) >= max_sources:
            break

    # Format context
    context_parts = []
    for i, source in enumerate(selected, start=1):
        meta = source.get("metadata", {})
        level = meta.get("level", "unknown")
        header = f"[Source {i} — Level {level}"
        if meta.get("player_name"):
            header += f", {meta['player_name']}"
        if meta.get("team_name"):
            header += f", {meta['team_name']}"
        if meta.get("match_id"):
            header += f", Match {meta['match_id']}"
        score_key = "rrf_score" if "rrf_score" in source else "score"
        header += f", Score: {source.get(score_key, 0):.4f}]"
        context_parts.append(f"{header}\n{source['text']}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
    return context, selected


if __name__ == "__main__":
    question = "How many goals did Messi score?"
    context, sources = build_context(question)
    print(f"Question: {question}")
    print(f"Sources: {len(sources)}")
    print(f"\nContext:\n{context}")
