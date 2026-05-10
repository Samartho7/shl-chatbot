"""
SHL Catalog Embedding Pipeline  —  v3 (Gemini text-embedding-004)
-----------------------------------------------------------------
Source: shl_official_catalog.json  (official SHL endpoint)
Embedder: Google Gemini text-embedding-004  (768-dim, no local model)
  • Eliminates sentence-transformers / PyTorch dependency
  • Identical semantic quality or better than all-MiniLM-L6-v2
  • Requires GEMINI_API_KEY in environment

Run once (or after re-downloading the catalog):
    python build_index.py

Output:
    shl_index.faiss        — FAISS IndexFlatIP (768-dim, 377 vectors)
    shl_index_meta.json    — parallel metadata list
"""

import json
import os
import sys
import time
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

from google import genai

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG_PATH = "shl_official_catalog.json"
INDEX_PATH   = "shl_index.faiss"
META_PATH    = "shl_index_meta.json"
EMBED_MODEL  = "models/gemini-embedding-001"  # 768-dim stable embedding model
BATCH_SIZE   = 5     # keep well under 100 RPM free-tier limit
SLEEP_BATCH  = 4.0   # seconds between batches → ~15 batches/min = safe

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


# ── Gemini batch embedding ────────────────────────────────────────────────────
def embed_texts(client: genai.Client, texts: list[str]) -> np.ndarray:
    """Embed all texts in batches via Gemini text-embedding-004 (768-dim)."""
    all_embeddings = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in range(total_batches):
        batch = texts[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        print(f"  Embedding batch {b_idx+1}/{total_batches} ({len(batch)} items)...", end=" ")

        # Retry logic for transient API errors / 429 rate-limit
        for attempt in range(5):
            try:
                response = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                )
                vecs = [e.values for e in response.embeddings]
                all_embeddings.extend(vecs)
                print(f"OK (dim={len(vecs[0])})")
                break
            except Exception as exc:
                err_str = str(exc)
                # Parse retry delay from 429 message
                wait = 20
                import re as _re
                m = _re.search(r"retry in (\d+)", err_str)
                if m:
                    wait = int(m.group(1)) + 2
                print(f"\n  [WARN] attempt {attempt+1} failed (waiting {wait}s): {exc}")
                if attempt == 4:
                    raise
                time.sleep(wait)

        if b_idx < total_batches - 1:
            time.sleep(SLEEP_BATCH)   # rate-limit buffer

    arr = np.array(all_embeddings, dtype=np.float32)
    # L2-normalise for cosine similarity via dot product
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY not set. Add it to .env or your environment.")

    client = genai.Client(api_key=api_key)

    # 1. Load catalog
    print(f"Loading catalog from '{CATALOG_PATH}' ...")
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"  {len(catalog)} assessments loaded.")

    # 2. Build text chunks
    print("\nBuilding text chunks ...")
    chunks = [build_text_chunk(item) for item in catalog]
    print(f"  {len(chunks)} chunks built.")

    # 3. Embed via Gemini API
    print(f"\nEmbedding with Gemini '{EMBED_MODEL}' ...")
    embeddings = embed_texts(client, chunks)
    print(f"Embedding matrix: {embeddings.shape}")   # expected (377, 768)

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
        "adaptive deductive reasoning test",
    ]
    for query in test_queries:
        q_resp = client.models.embed_content(model=EMBED_MODEL, contents=query)
        q_vec  = np.array(q_resp.embeddings[0].values, dtype=np.float32)
        q_vec  = (q_vec / np.linalg.norm(q_vec)).reshape(1, -1)
        scores, idxs = index.search(q_vec, k=3)
        print(f"\nQuery: '{query}'")
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
            m = meta[idx]
            print(f"  {rank}. [{score:.3f}]  {m['name']}  |  {', '.join(m['test_types'])}")

    print("\nIndex build complete.")


if __name__ == "__main__":
    main()