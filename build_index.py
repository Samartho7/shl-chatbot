"""
SHL Catalog Embedding Pipeline  —  v2 (official catalog)
---------------------------------------------------------
Source: shl_official_catalog.json  (downloaded from SHL's own endpoint)
Each record contains: name, link, description, job_levels, languages,
duration, remote, adaptive, keys (type labels).

Run once (or after re-downloading the catalog):
    python build_index.py

Output:
    shl_index.faiss        — FAISS IndexFlatIP (384-dim, 377 vectors)
    shl_index_meta.json    — parallel metadata list
"""

import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG_PATH = "shl_official_catalog.json"   # official SHL catalog with descriptions
INDEX_PATH   = "shl_index.faiss"
META_PATH    = "shl_index_meta.json"
MODEL_NAME   = "all-MiniLM-L6-v2"           # 384-dim, fully offline after first download

# Type-label → single-letter code map
LABEL_TO_CODE = {
    "Ability & Aptitude":             "A",
    "Biodata & Situational Judgment": "B",
    "Competencies":                   "C",
    "Development & 360":              "D",
    "Assessment Exercises":           "E",
    "Knowledge & Skills":             "K",
    "Personality & Behavior":         "P",
    "Simulations":                    "S",
}


# ── Text chunk builder ────────────────────────────────────────────────────────
def build_text_chunk(item: dict) -> str:
    """
    Build a rich natural-language embedding string from the official catalog record.
    Includes the actual description (written by SHL), job levels, duration,
    languages, and type labels — far superior to heuristic keyword enrichment.
    """
    name        = item.get("name", "").strip()
    description = item.get("description", "").strip()
    job_levels  = item.get("job_levels", [])
    languages   = item.get("languages", [])
    duration    = item.get("duration", "").strip()
    keys        = item.get("keys", [])          # full type labels
    remote      = item.get("remote", "no")
    adaptive    = item.get("adaptive", "no")
    url         = item.get("link", item.get("url", "")).strip()

    type_codes = [LABEL_TO_CODE.get(k, k[0]) for k in keys]

    lines = [
        f"Assessment: {name}",
        f"Description: {description}"             if description  else "",
        f"Test type: {', '.join(keys)}"           if keys         else "",
        f"Type codes: {', '.join(type_codes)}"    if type_codes   else "",
        f"Job levels: {', '.join(job_levels)}"    if job_levels   else "",
        f"Duration: {duration}"                   if duration     else "",
        f"Languages: {', '.join(languages)}"      if languages    else "",
        f"Remote testing: {remote}",
        f"Adaptive/IRT: {adaptive}",
        f"URL: {url}",
    ]
    return "\n".join(line for line in lines if line)


# ── Metadata record builder ────────────────────────────────────────────────────
def build_meta_record(item: dict) -> dict:
    """Metadata stored parallel to FAISS index — returned verbatim by the chatbot."""
    keys       = item.get("keys", [])
    type_codes = [LABEL_TO_CODE.get(k, k[0]) for k in keys]
    url        = item.get("link", item.get("url", ""))

    return {
        "name":            item.get("name", ""),
        "url":             url,
        "remote_testing":  str(item.get("remote",    "no")).strip().lower() == "yes",
        "adaptive_irt":    str(item.get("adaptive",  "no")).strip().lower() == "yes",
        "test_type_codes": type_codes,
        "test_types":      keys,
        "description":     item.get("description", ""),
        "job_levels":      item.get("job_levels",  []),
        "languages":       item.get("languages",   []),
        "duration":        item.get("duration",    ""),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load catalog
    print(f"Loading catalog from '{CATALOG_PATH}' ...")
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"  {len(catalog)} assessments loaded.")

    # 2. Build text chunks
    print("\nBuilding text chunks ...")
    chunks = [build_text_chunk(item) for item in catalog]
    empty  = sum(1 for c in chunks if not c.strip())
    if empty:
        print(f"  [WARN] {empty} empty chunks.")

    print("\nSample chunk (item 0):")
    print("-" * 60)
    print(chunks[0])
    print("-" * 60)

    # 3. Embed
    print(f"\nLoading embedding model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding all chunks (10–30 s on CPU) ...")
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine sim via dot product
    )
    print(f"Embedding shape: {embeddings.shape}")   # expected (377, 384)

    # 4. Build FAISS index
    print("\nBuilding FAISS index ...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"  {index.ntotal} vectors indexed.")

    # 5. Save
    faiss.write_index(index, INDEX_PATH)
    print(f"  FAISS index saved -> {INDEX_PATH}")

    meta = [build_meta_record(item) for item in catalog]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved -> {META_PATH}")

    # 6. Smoke test
    print("\n-- Quick retrieval smoke-test --")
    test_queries = [
        "Java developer mid level coding and personality",
        "personality assessment for senior sales rep",
        "cognitive ability numerical reasoning graduate",
        "Python data scientist machine learning",
        "adaptive deductive reasoning test",
    ]
    for query in test_queries:
        q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, idxs = index.search(q_vec, k=3)
        print(f"\nQuery: '{query}'")
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
            m = meta[idx]
            print(f"  {rank}. [{score:.3f}]  {m['name']}  |  {', '.join(m['test_types'])}")

    print("\nIndex build complete.")


if __name__ == "__main__":
    main()