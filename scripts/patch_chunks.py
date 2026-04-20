"""
scripts/patch_chunks.py
-----------------------
ONE-TIME script. Run this ONCE on your existing chunks.pkl.
Adds a 'company' field to every chunk by reading the 'source' filename.

Your existing pkl looks like:
    {'chunk_id': 53, 'source': 'accenture-fy25.md', 'section': '...', 'text': '...'}

After patch:
    {'chunk_id': 53, 'source': 'accenture-fy25.md', 'section': '...', 'text': '...', 'company': 'Accenture'}

NO re-embedding. NO changes to faiss.index. Just pkl gets updated.

Run:
    python scripts/patch_chunks.py
    python scripts/patch_chunks.py --dry-run   # preview without saving
"""

import pickle
import shutil
import sys
from pathlib import Path
from collections import defaultdict

ROOT        = Path(__file__).parent.parent
CHUNKS_FILE = ROOT / "indexes" / "chunks.pkl"
BACKUP_FILE = ROOT / "indexes" / "chunks_backup.pkl"

# ── company detection from filename ──────────────────────────────────────────
# add any new company here if your corpus grows
COMPANY_MAP = {
    "accenture"  : "Accenture",
    "infosys"    : "Infosys",
    "cognizant"  : "Cognizant",
    "cts"        : "Cognizant",   # CTS = Cognizant Technology Solutions
    "tcs"        : "TCS",
    "wipro"      : "Wipro",
}

def detect_company(source_filename: str) -> str:
    """
    Extract company name from the source filename.
    'accenture-fy25.md'  → 'Accenture'
    'infosys_ar_fy24.md' → 'Infosys'
    'cognizant-fy24.md'  → 'Cognizant'
    Falls back to the filename stem if no match found.
    """
    name = source_filename.lower()
    for keyword, company in COMPANY_MAP.items():
        if keyword in name:
            return company
    # fallback: use filename without extension, title-cased
    return Path(source_filename).stem.title()


def main(dry_run: bool = False):
    print("=== Patch chunks.pkl — add company field ===\n")

    # ── check file exists ─────────────────────────────────────────────────────
    if not CHUNKS_FILE.exists():
        print(f"ERROR: {CHUNKS_FILE} not found.")
        print("Make sure your chunks.pkl is at indexes/chunks.pkl")
        sys.exit(1)

    # ── load existing pkl ─────────────────────────────────────────────────────
    with open(CHUNKS_FILE, "rb") as f:
        chunks: list[dict] = pickle.load(f)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE.name}")
    print(f"Existing keys: {list(chunks[0].keys())}\n")

    # ── check if already patched ──────────────────────────────────────────────
    if "company" in chunks[0]:
        print("'company' field already present. Nothing to do.")
        dist = defaultdict(int)
        for c in chunks:
            dist[c["company"]] += 1
        print("Current distribution:")
        for comp, count in sorted(dist.items()):
            print(f"  {comp}: {count} chunks")
        return

    # ── add company field to every chunk ──────────────────────────────────────
    unknown_sources = set()
    dist            = defaultdict(int)

    for chunk in chunks:
        source  = chunk.get("source", "")
        company = detect_company(source)
        chunk["company"] = company
        dist[company] += 1

        # track sources that fell through to fallback
        name = source.lower()
        if not any(kw in name for kw in COMPANY_MAP):
            unknown_sources.add(source)

    # ── preview ───────────────────────────────────────────────────────────────
    print("Company detection results:")
    for comp, count in sorted(dist.items()):
        print(f"  {comp}: {count} chunks")

    if unknown_sources:
        print(f"\nWARNING: These sources had no keyword match (used filename fallback):")
        for s in sorted(unknown_sources):
            print(f"  {s}")
        print("  → Add them to COMPANY_MAP in this script if the name is wrong.")

    print(f"\nSample after patch:")
    for chunk in chunks[:3]:
        print(f"  chunk_id={chunk['chunk_id']} | source={chunk['source']} | company={chunk['company']}")

    if dry_run:
        print("\n[DRY RUN] No files written. Run without --dry-run to apply.")
        return

    # ── backup original ───────────────────────────────────────────────────────
    shutil.copy2(CHUNKS_FILE, BACKUP_FILE)
    print(f"\nBackup saved → {BACKUP_FILE.name}")

    # ── save patched pkl ──────────────────────────────────────────────────────
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Patched pkl saved → {CHUNKS_FILE.name}")
    print("\n=== Done. faiss.index unchanged. No re-embedding needed. ===")
    print("\nVerify with:")
    print("  python scripts/patch_chunks.py --dry-run")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)