"""
scripts/ingest_unstructured.py
------------------------------
Reads all .md files from data/unstructured/
Chunks them → embeds with Google Gemini → stores in FAISS index

Run:
    python scripts/ingest_unstructured.py

Output:
    indexes/faiss.index   — the FAISS vector index
    indexes/chunks.pkl    — chunk metadata (text, source, section, chunk_id)

What this file does step by step:
    1. Read every .md file in data/unstructured/
    2. Split each file into chunks (by heading sections first, then by word count)
    3. Embed each chunk using Google Gemini embedding model
    4. Store vectors in a FAISS IndexFlatIP (cosine similarity via L2 normalisation)
    5. Store chunk metadata (text + source info) in a pickle file alongside the index
"""

import os
import re
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
UNSTRUCTURED  = ROOT / "data" / "unstructure"
INDEX_DIR     = ROOT / "indexes"
INDEX_FILE    = INDEX_DIR / "faiss.index"
CHUNKS_FILE   = INDEX_DIR / "chunks.pkl"

# ── chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 300    # max words per chunk
CHUNK_OVERLAP = 50     # words of overlap between consecutive chunks
EMBED_MODEL   = "nomic-ai/nomic-embed-text-v1.5"  # Local model (runs on your machine)
EMBED_DIM     = 768    # nomic-embed-text-v1.5 returns 768 dimensions
BATCH_SIZE    = 50     # how many chunks to embed per batch


# ── data class ────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id : int
    text     : str
    source   : str    # filename, e.g. "infosys_ar_fy24.md"
    section  : str    # nearest heading above this chunk, e.g. "## Operating Margin"


# ── MD chunking ───────────────────────────────────────────────────────────────
def parse_md_into_sections(md_text: str, filename: str) -> list[tuple[str, str]]:
    """
    Split a markdown file into (section_heading, section_text) pairs.
    Splits on any line starting with # (any level heading).
    Returns a list of (heading, body_text) tuples.
    """
    # Pattern: any line that starts with one or more # characters
    heading_pattern = re.compile(r"^(#{1,6} .+)$", re.MULTILINE)
    positions = [(m.start(), m.group()) for m in heading_pattern.finditer(md_text)]

    sections = []
    for i, (pos, heading) in enumerate(positions):
        # body goes from after this heading to before the next heading (or EOF)
        start = pos + len(heading)
        end   = positions[i + 1][0] if i + 1 < len(positions) else len(md_text)
        body  = md_text[start:end].strip()
        if body:
            sections.append((heading.strip(), body))

    # if no headings found, treat whole file as one section
    if not sections:
        sections.append((filename, md_text.strip()))

    return sections


def split_into_word_chunks(text: str, size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Each chunk has at most `size` words. Consecutive chunks share `overlap` words.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end        = start + size
        chunk_text = " ".join(words[start:end]).strip()
        if len(chunk_text) > 40:      # skip near-empty chunks
            chunks.append(chunk_text)
        start += size - overlap
    return chunks


def load_and_chunk_all_md_files() -> list[Chunk]:
    """
    Read every .md file in UNSTRUCTURED directory.
    Parse into sections, then split each section into overlapping word chunks.
    Returns a flat list of Chunk objects.
    """
    md_files = sorted(UNSTRUCTURED.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {UNSTRUCTURED}")
        print("Add your markdown files there and re-run.")
        sys.exit(1)

    print(f"Found {len(md_files)} .md file(s): {[f.name for f in md_files]}")

    all_chunks: list[Chunk] = []
    chunk_id = 0

    for md_path in md_files:
        text     = md_path.read_text(encoding="utf-8")
        sections = parse_md_into_sections(text, md_path.name)

        for heading, body in sections:
            word_chunks = split_into_word_chunks(body, CHUNK_SIZE, CHUNK_OVERLAP)
            for wc in word_chunks:
                all_chunks.append(
                    Chunk(
                        chunk_id = chunk_id,
                        text     = wc,
                        source   = md_path.name,
                        section  = heading,
                    )
                )
                chunk_id += 1

        print(f"  {md_path.name}: {len(sections)} section(s), "
              f"{sum(len(split_into_word_chunks(b, CHUNK_SIZE, CHUNK_OVERLAP)) for _, b in sections)} chunk(s)")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    return all_chunks


# ── embedding ─────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings using sentence-transformers (local model).
    Returns float32 numpy array of shape (len(texts), EMBED_DIM).
    No API calls, no rate limits!
    """
    vectors = model.encode(texts, show_progress_bar=True, batch_size=BATCH_SIZE)
    return np.array(vectors, dtype="float32")


# ── FAISS index builder ───────────────────────────────────────────────────────
def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (Inner Product) index.
    Vectors must be L2-normalised first so that inner product == cosine similarity.
    Returns the built index.
    """
    faiss.normalize_L2(vectors)                  # normalise in-place
    index = faiss.IndexFlatIP(EMBED_DIM)         # flat exact-search index
    index.add(vectors)
    return index


# ── save ──────────────────────────────────────────────────────────────────────
def save(index: faiss.IndexFlatIP, chunks: list[Chunk]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"FAISS index saved → {INDEX_FILE}  ({index.ntotal} vectors)")

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump([asdict(c) for c in chunks], f)
    print(f"Chunk metadata saved → {CHUNKS_FILE}  ({len(chunks)} chunks)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Unstructured Ingestion: MD → FAISS ===\n")

    # 1. load and chunk
    chunks = load_and_chunk_all_md_files()

    # 2. embed
    print("\nEmbedding chunks...")
    texts   = [c.text for c in chunks]
    vectors = embed_texts(texts)

    # 3. build index
    print("\nBuilding FAISS index...")
    index = build_faiss_index(vectors)

    # 4. save
    print("\nSaving...")
    save(index, chunks)

    print("\n=== Done. Unstructured data ready. ===")
    print(f"  Index : {INDEX_FILE}")
    print(f"  Chunks: {CHUNKS_FILE}")
    print("\nVerify retrieval with:")
    print('  python scripts/ingest_unstructured.py --test "operating margin improvement"')


# ── quick retrieval test ──────────────────────────────────────────────────────
def test_retrieval(query: str, top_k: int = 3) -> None:
    """
    Quick smoke test: load the saved index and search for a query.
    Prints top_k results so you can manually verify relevance.
    """
    if not INDEX_FILE.exists():
        print("Index not found. Run without --test first.")
        return

    index  = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)   # list of dicts

    # embed query using local model
    qvec = np.array([model.encode(query)], dtype="float32")
    faiss.normalize_L2(qvec)

    # search
    scores, idxs = index.search(qvec, top_k)

    print(f"\nQuery: '{query}'\n")
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
        if idx == -1:
            continue
        c = chunks[idx]
        print(f"Rank {rank} | Score: {score:.4f} | {c['source']} § {c['section']}")
        print(f"  {c['text'][:200]}...")
        print()


if __name__ == "__main__":
    if "--test" in sys.argv:
        query = " ".join(sys.argv[sys.argv.index("--test") + 1:]) or "operating margin"
        test_retrieval(query)
    else:
        main()