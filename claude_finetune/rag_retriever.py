"""
rag_retriever.py

Minimal, dependency-free knowledge base loader + retriever for the AECI
emergency multi-agent system. Deliberately does NOT use ChromaDB /
sentence-transformers / scikit-learn: the knowledge base is small and
already partitioned by emergency_type, so a direct category lookup is
both simpler and more reliable than an embedding similarity search for
the common case (we already know the category from the Emergency Agent).

A pure-Python cosine-similarity-over-word-counts search is included as a
fallback for the genuinely ambiguous case (emergency_type == "unknown"
or "unknown_other"), so the system still does something recognizably
"RAG-like" when the category itself is uncertain — without pulling in a
new ML dependency.
"""

import math
import os
import re
from collections import Counter

KB_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")


def load_kb(kb_dir: str = KB_DIR_DEFAULT) -> dict:
    """Load every *.txt file in kb_dir into {category_name: text}."""
    kb = {}
    if not os.path.isdir(kb_dir):
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")
    for fname in sorted(os.listdir(kb_dir)):
        if fname.endswith(".txt"):
            category = fname[:-4]
            with open(os.path.join(kb_dir, fname), encoding="utf-8") as f:
                kb[category] = f.read()
    return kb


def _tokenize(text: str):
    return re.findall(r"[a-zA-Z\u0900-\u097F]+", text.lower())


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search_kb(query: str, kb: dict, top_k: int = 1):
    """Fallback fuzzy search across all KB categories by word-overlap
    cosine similarity. Returns list of (category, score) sorted desc."""
    q_vec = Counter(_tokenize(query))
    scores = []
    for category, text in kb.items():
        doc_vec = Counter(_tokenize(text))
        scores.append((category, _cosine(q_vec, doc_vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def get_context(emergency_type: str, transcript: str, kb: dict) -> tuple:
    """
    Primary path: direct lookup by emergency_type (reliable, since the
    Emergency Agent already classified the transcript).
    Fallback path: if emergency_type is missing/unknown/unrecognized,
    fuzzy-search the whole KB using the transcript text instead.

    Returns (source_category, context_text, was_fallback: bool).
    """
    if emergency_type in kb and emergency_type not in ("unknown", "unknown_other"):
        return emergency_type, kb[emergency_type], False

    # ambiguous / unknown category -> fuzzy search
    results = search_kb(transcript, kb, top_k=1)
    if results and results[0][1] > 0:
        best_cat, score = results[0]
        return best_cat, kb[best_cat], True

    # nothing matched at all
    fallback_text = kb.get("unknown_other", "")
    return "unknown_other", fallback_text, True


if __name__ == "__main__":
    kb = load_kb()
    print(f"Loaded {len(kb)} knowledge base categories: {sorted(kb.keys())}")
    cat, ctx, fb = get_context("fire", "ghar mein aag lag gayi hai", kb)
    print(f"\nDirect lookup test -> category={cat}, fallback={fb}")
    print(ctx[:200], "...")

    cat, ctx, fb = get_context("unknown", "mera bhai subah se nahi mila kahin nahi", kb)
    print(f"\nFallback search test -> category={cat}, fallback={fb}")
    print(ctx[:200], "...")
