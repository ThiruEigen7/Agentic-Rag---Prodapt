"""
tools/search_docs.py
--------------------
Semantic search over unstructured MD documents stored in FAISS.

Requires:
    indexes/faiss.index   — FAISS vector index (untouched)
    indexes/chunks.pkl    — chunk metadata WITH 'company' field
                            (run scripts/patch_chunks.py once if field missing)

Three search modes to prevent retrieval bias:

    1. company="Accenture"  → only Accenture chunks searched
       Use when: question is clearly about one company

    2. balanced=True        → top_k per company, merged
       Use when: question compares multiple companies

    3. neither              → standard full-index search
       Use when: question is generic / not company-specific

Standalone test:
    python tools/search_docs.py "CTS revenue FY24"
    python tools/search_docs.py "CTS revenue FY24" --company Cognizant
    python tools/search_docs.py "compare revenue" --balanced
"""

import os
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
INDEX_FILE  = ROOT / "indexes" / "faiss.index"
CHUNKS_FILE = ROOT / "indexes" / "chunks.pkl"

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM   = 768

# company name detection — used as fallback if chunk has no 'company' field
COMPANY_MAP = {
    "accenture"  : "Accenture",
    "infosys"    : "Infosys",
    "cognizant"  : "Cognizant",
    "cts"        : "Cognizant",
}

# load model once at module level
_model: SentenceTransformer = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    return _model


# ── helpers ───────────────────────────────────────────────────────────────────
def _detect_company_from_source(source: str) -> str:
    """Fallback: detect company from filename if 'company' key missing in pkl."""
    name = source.lower()
    for kw, comp in COMPANY_MAP.items():
        if kw in name:
            return comp
    return Path(source).stem.title()


def _load() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load FAISS index and chunk metadata. Auto-adds company field if missing."""
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}\n"
            "Run: python scripts/ingest_unstructured.py"
        )
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}\n"
            "Run: python scripts/ingest_unstructured.py"
        )

    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "rb") as f:
        chunks: list[dict] = pickle.load(f)

    # auto-patch: if 'company' field missing, derive it from 'source'
    # this means search_docs works even if patch_chunks.py hasn't been run yet
    patched = 0
    for chunk in chunks:
        if "company" not in chunk:
            chunk["company"] = _detect_company_from_source(chunk.get("source", ""))
            patched += 1
    if patched > 0:
        print(f"[search_docs] Auto-added company field to {patched} chunks.")
        print("[search_docs] Run 'python scripts/patch_chunks.py' to persist this.")

    return index, chunks


def _embed_query(query: str) -> np.ndarray:
    """Embed a query string. Returns shape (1, EMBED_DIM), L2-normalised."""
    model = _get_model()
    vec   = model.encode(
        [f"search_query: {query}"],
        normalize_embeddings = True,
    )
    return np.array(vec, dtype="float32")


def _sub_search(
    index         : faiss.IndexFlatIP,
    subset_chunks : list[dict],
    qvec          : np.ndarray,
    k             : int,
) -> list[dict]:
    """
    Build a temporary sub-index from a subset of chunks, search it.
    Uses index.reconstruct(chunk_id) to pull stored vectors — no re-embedding.
    chunk_id == position in the main index (they were added in order).
    """
    if not subset_chunks:
        return []

    # pull vectors from main index by chunk_id
    sub_vecs = np.zeros((len(subset_chunks), EMBED_DIM), dtype="float32")
    for i, chunk in enumerate(subset_chunks):
        sub_vecs[i] = index.reconstruct(chunk["chunk_id"])

    # build temporary sub-index and search
    sub_idx = faiss.IndexFlatIP(EMBED_DIM)
    sub_idx.add(sub_vecs)

    k       = min(k, len(subset_chunks))
    scores, positions = sub_idx.search(qvec, k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue
        results.append({**subset_chunks[pos], "score": float(score)})
    return results


# ── main tool function ────────────────────────────────────────────────────────
def search_docs(
    query    : str,
    top_k    : int  = 3,
    company  : str  = None,
    balanced : bool = False,
) -> dict:
    """
    Search the FAISS index for relevant document chunks.

    Args:
        query    : natural language search string
        top_k    : results to return per company (balanced) or total (others)
        company  : restrict to one company only e.g. "Accenture", "Cognizant"
        balanced : return top_k per company — use for comparison questions

    Returns:
        dict with key 'chunks': list of dicts each containing
            text     : chunk text
            source   : source filename
            section  : section heading
            company  : company name
            score    : cosine similarity (0-1)
            chunk_id : position in index

    Raises:
        FileNotFoundError : if index or chunks file missing
    """
    index, all_chunks = _load()
    qvec = _embed_query(query)

    if company:
        # ── mode 1: single company ────────────────────────────────────────────
        subset = [c for c in all_chunks if c["company"] == company]
        if not subset:
            available = sorted({c["company"] for c in all_chunks})
            return {
                "chunks" : [],
                "error"  : f"No chunks found for company='{company}'. "
                           f"Available: {available}",
            }
        results = _sub_search(index, subset, qvec, top_k)

    elif balanced:
        # ── mode 2: balanced — top_k per company ─────────────────────────────
        by_company = defaultdict(list)
        for c in all_chunks:
            by_company[c["company"]].append(c)

        results = []
        for comp_chunks in by_company.values():
            results.extend(_sub_search(index, comp_chunks, qvec, top_k))
        results.sort(key=lambda x: x["score"], reverse=True)

    else:
        # ── mode 3: standard full-index search ────────────────────────────────
        scores, idxs = index.search(qvec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx != -1:
                results.append({**all_chunks[idx], "score": float(score)})

    return {"chunks": results}


# ── agent helper: detect company from question ─────────────────────────────────
def detect_company_in_query(query: str) -> str | None:
    """
    Detect if a specific company is mentioned in the query.
    Returns canonical company name or None.

    Used by the agent to decide whether to use company= filter.

    Examples:
        "CTS revenue FY24"        → "Cognizant"
        "Accenture operating margin" → "Accenture"
        "compare all companies"   → None
    """
    q = query.lower()
    for keyword, company in COMPANY_MAP.items():
        if keyword in q:
            return company
    return None


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python tools/search_docs.py "your query"')
        print('  python tools/search_docs.py "CTS revenue" --company Cognizant')
        print('  python tools/search_docs.py "compare revenue" --balanced')
        sys.exit(1)

    args    = sys.argv[1:]
    query   = args[0]
    company = None
    balanced = "--balanced" in args

    if "--company" in args:
        idx     = args.index("--company")
        company = args[idx + 1]

    print(f"\nQuery   : '{query}'")
    print(f"Mode    : {'company='+company if company else 'balanced' if balanced else 'standard'}\n")

    result = search_docs(query, top_k=3, company=company, balanced=balanced)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    chunks = result["chunks"]
    if not chunks:
        print("No results found.")
        sys.exit(0)

    for i, chunk in enumerate(chunks, 1):
        print(f"── Result {i} {'─'*40}")
        print(f"Company : {chunk['company']}")
        print(f"Source  : {chunk['source']}")
        print(f"Section : {chunk['section'][:60]}")
        print(f"Score   : {chunk['score']:.4f}")
        print(f"Text    : {chunk['text'][:200]}...")
        print()