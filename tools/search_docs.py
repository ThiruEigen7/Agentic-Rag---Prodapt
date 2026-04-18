"""
Tool 1: search_docs
-------------------
Semantic search over unstructured annual report PDFs.

Standalone usage:
    python tools/search_docs.py "what did Infosys say about margin improvement"
    python tools/search_docs.py "TCS strategic priorities FY24"
"""

import os
import sys
import json
import pickle
from pathlib import Path
from dataclasses import dataclass

# ── third-party (install via requirements.txt) ────────────────────────────────
try:
    import faiss
    import numpy as np
    import pdfplumber
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
PDF_DIR       = Path(__file__).parent.parent / "data" / "pdfs"
INDEX_DIR     = Path(__file__).parent.parent / "indexes"
INDEX_FILE    = INDEX_DIR / "faiss.index"
CHUNKS_FILE   = INDEX_DIR / "chunks.pkl"
EMBED_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE    = 400   # tokens (approximate via word count)
CHUNK_OVERLAP = 50
TOP_K         = 3


# ── data types ────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    source: str   # filename, e.g. "infosys_ar_fy24.pdf"
    page: int


@dataclass
class SearchResult:
    text: str
    source: str
    page: int
    score: float


# ── embedding helper ──────────────────────────────────────────────────────────
def embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings using OpenAI text-embedding-3-small.
    Returns a float32 numpy array of shape (len(texts), 1536).
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


# ── chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, source: str, page: int) -> list[Chunk]:
    """
    Split page text into overlapping word-based chunks.
    Returns a list of Chunk objects.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if len(chunk_text) > 50:           # skip near-empty chunks
            chunks.append(Chunk(text=chunk_text, source=source, page=page))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── index builder ─────────────────────────────────────────────────────────────
def build_index():
    """
    Parse all PDFs in PDF_DIR, chunk them, embed them, and save a FAISS index.
    Run this once before using search_docs.

    Usage:
        python tools/search_docs.py --build
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        print("Add annual report PDFs named like: infosys_ar_fy24.pdf")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_files:
        print(f"  Parsing {pdf_path.name}...")
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                page_chunks = chunk_text(text, pdf_path.name, page_num)
                all_chunks.extend(page_chunks)

    print(f"Total chunks: {len(all_chunks)}")
    print("Embedding chunks (this may take a minute)...")

    texts = [c.text for c in all_chunks]

    # embed in batches of 100 to avoid API limits
    all_vectors = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embed(batch)
        all_vectors.append(vecs)
        print(f"  Embedded {min(i + batch_size, len(texts))} / {len(texts)}")

    vectors = np.vstack(all_vectors)

    # build FAISS flat index (exact cosine similarity via normalisation)
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after normalisation
    index.add(vectors)

    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Index saved to {INDEX_FILE} ({index.ntotal} vectors)")
    print(f"Chunks saved to {CHUNKS_FILE}")


# ── main tool function ────────────────────────────────────────────────────────
def search_docs(query: str) -> dict:
    """
    Semantic search over the indexed annual report PDFs.

    Args:
        query: Natural language question or phrase to search for.

    Returns:
        dict with key "chunks" — list of top-3 results, each containing:
            text   : extracted text chunk
            source : source filename
            page   : page number
            score  : cosine similarity (0–1, higher = more relevant)

    Raises:
        FileNotFoundError: if the FAISS index hasn't been built yet.
    """
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_FILE}. "
            "Run: python tools/search_docs.py --build"
        )

    # load index and chunks
    index = faiss.read_index(str(INDEX_FILE))
    with open(CHUNKS_FILE, "rb") as f:
        all_chunks: list[Chunk] = pickle.load(f)

    # embed the query
    query_vec = embed([query])
    faiss.normalize_L2(query_vec)

    # search
    scores, indices = index.search(query_vec, TOP_K)
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:                      # FAISS returns -1 for empty slots
            continue
        chunk = all_chunks[idx]
        results.append(
            SearchResult(
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                score=round(float(score), 4),
            )
        )

    return {
        "chunks": [
            {
                "text": r.text,
                "source": r.source,
                "page": r.page,
                "score": r.score,
            }
            for r in results
        ]
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tools/search_docs.py --build")
        print('  python tools/search_docs.py "your query here"')
        sys.exit(1)

    if sys.argv[1] == "--build":
        build_index()
    else:
        query = " ".join(sys.argv[1:])
        print(f"\nSearching for: '{query}'\n")
        try:
            result = search_docs(query)
            for i, chunk in enumerate(result["chunks"], 1):
                print(f"── Result {i} ──────────────────────────────────")
                print(f"Source : {chunk['source']}  (page {chunk['page']})")
                print(f"Score  : {chunk['score']}")
                print(f"Text   : {chunk['text'][:300]}...")
                print()
        except FileNotFoundError as e:
            print(f"Error: {e}")