"""
SHL Catalog Embedding Pipeline  —  v4 (fastembed ONNX)
-------------------------------------------------------
Embedder: fastembed TextEmbedding BAAI/bge-small-en-v1.5
  • 33MB ONNX model, 384-dim
  • No PyTorch, no API calls, no geographic restrictions
  • ~100MB total memory on Render free tier

Run once (or after re-downloading the catalog):
    python build_index.py

Output:
    shl_index.faiss        — FAISS IndexFlatIP (384-dim, 377 vectors)
    shl_index_meta.json    — parallel metadata list
"""

import json
import sys
import numpy as np
import faiss
from pathlib import Path
from fastembed import TextEmbedding

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG_PATH = "shl_official_catalog.json"
INDEX_PATH   = "shl_index.faiss"
META_PATH    = "shl_index_meta.json"
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"   # 384-dim ONNX

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


def build_text_chunk(item: dict) -> str:
    name        = item.get("name", "").strip()
    description = item.get("description", "").strip()
    job_levels  = item.get("job_levels", [])
    languages   = item.get("languages", [])
    duration    = item.get("duration", "").strip()
    keys        = item.get("keys", [])
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


def build_meta_record(item: dict) -> dict:
    keys       = item.get("keys", [])
    type_codes = [LABEL_TO_CODE.get(k, k[0]) for k in keys]
    url        = item.get("link", item.get("url", ""))

    return {
        "name":            item.get("name", ""),
        "url":             url,
        "remote_testing":  str(item.get("remote",   "no")).strip().lower() == "yes",
        "adaptive_irt":    str(item.get("adaptive", "no")).strip().lower() == "yes",
        "test_type_codes": type_codes,
        "test_types":      keys,
        "description":     item.get("description", ""),
        "job_levels":      item.get("job_levels",  []),
        "languages":       item.get("languages",   []),
        "duration":        item.get("duration",    ""),
    }


def main():
    # 1. Load catalog
    print(f"Loading catalog from '{CATALOG_PATH}' ...")
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"  {len(catalog)} assessments loaded.")

    # 2. Build text chunks
    print("\nBuilding text chunks ...")
    chunks = [build_text_chunk(item) for item in catalog]

    # 3. Embed via fastembed (local ONNX, no API calls)
    print(f"\nLoading fastembed model '{EMBED_MODEL}' ...")
    embedder = TextEmbedding(EMBED_MODEL)

    print("Embedding all chunks (local ONNX, no rate limits)...")
    raw = list(embedder.embed(chunks))
    embeddings = np.array(raw, dtype=np.float32)

    # L2-normalise for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms
    print(f"Embedding matrix: {embeddings.shape}")   # expected (377, 384)

    # 4. Build FAISS index
    print("\nBuilding FAISS IndexFlatIP ...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  {index.ntotal} vectors indexed (dim={dim}).")

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
        "adaptive deductive reasoning",
    ]
    for query in test_queries:
        q_vec = np.array(list(embedder.embed([query]))[0], dtype=np.float32)
        q_vec = (q_vec / np.linalg.norm(q_vec)).reshape(1, -1)
        scores, idxs = index.search(q_vec, k=3)
        print(f"\nQuery: '{query}'")
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
            m = meta[idx]
            print(f"  {rank}. [{score:.3f}]  {m['name']}  |  {', '.join(m['test_types'])}")

    print("\nIndex build complete.")


if __name__ == "__main__":
    main()