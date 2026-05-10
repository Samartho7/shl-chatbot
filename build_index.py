"""
SHL Catalog Embedding Pipeline  —  Gemini embedding-001
--------------------------------------------------------
Embedder: models/gemini-embedding-001  (3072-dim, Gemini API)

Run once:
    python build_index.py

Output:
    shl_index.faiss        — FAISS IndexFlatIP (3072-dim)
    shl_index_meta.json    — parallel metadata list
"""

import json
import os
import sys
import time
import re
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

from google import genai
from google.genai import types as _gt

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG_PATH = "shl_official_catalog.json"
INDEX_PATH   = "shl_index.faiss"
META_PATH    = "shl_index_meta.json"
EMBED_MODEL  = "models/gemini-embedding-001"
BATCH_SIZE   = 5     # stay under 100 RPM free-tier limit
SLEEP_BATCH  = 4.0   # seconds between batches

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
    type_codes  = [LABEL_TO_CODE.get(k, k[0]) for k in keys]

    lines = [
        f"Assessment: {name}",
        f"Description: {description}"          if description else "",
        f"Test type: {', '.join(keys)}"        if keys        else "",
        f"Type codes: {', '.join(type_codes)}" if type_codes  else "",
        f"Job levels: {', '.join(job_levels)}" if job_levels  else "",
        f"Duration: {duration}"                if duration    else "",
        f"Languages: {', '.join(languages)}"   if languages   else "",
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


def embed_texts(client, texts: list[str]) -> np.ndarray:
    all_embeddings = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in range(total_batches):
        batch = texts[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        print(f"  Batch {b_idx+1}/{total_batches} ({len(batch)} items)...", end=" ", flush=True)

        for attempt in range(5):
            try:
                response = client.models.embed_content(model=EMBED_MODEL, contents=batch)
                vecs = [e.values for e in response.embeddings]
                all_embeddings.extend(vecs)
                print(f"OK (dim={len(vecs[0])})")
                break
            except Exception as exc:
                wait = 20
                m = re.search(r"retry in (\d+)", str(exc))
                if m:
                    wait = int(m.group(1)) + 2
                print(f"\n  [WARN] attempt {attempt+1} failed (waiting {wait}s): {exc}")
                if attempt == 4:
                    raise
                time.sleep(wait)

        if b_idx < total_batches - 1:
            time.sleep(SLEEP_BATCH)

    arr = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def main():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY not set.")

    client = genai.Client(
        api_key=api_key,
        http_options=_gt.HttpOptions(api_version="v1"),
    )

    print(f"Loading catalog from '{CATALOG_PATH}' ...")
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"  {len(catalog)} assessments loaded.")

    chunks = [build_text_chunk(item) for item in catalog]
    print(f"\nEmbedding {len(chunks)} chunks via Gemini '{EMBED_MODEL}' ...")
    embeddings = embed_texts(client, chunks)
    print(f"Embedding matrix: {embeddings.shape}")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"\nFAISS index: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, INDEX_PATH)
    meta = [build_meta_record(item) for item in catalog]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved: {INDEX_PATH}, {META_PATH}")

    # Smoke test
    print("\n-- Smoke test --")
    for query in ["Java developer coding", "personality sales rep", "numerical reasoning graduate"]:
        r = client.models.embed_content(model=EMBED_MODEL, contents=query)
        q = np.array(r.embeddings[0].values, dtype=np.float32)
        q = (q / np.linalg.norm(q)).reshape(1, -1)
        scores, idxs = index.search(q, 3)
        print(f"\n'{query}'")
        for score, idx in zip(scores[0], idxs[0]):
            print(f"  [{score:.3f}] {meta[idx]['name']}")

    print("\nDone.")


if __name__ == "__main__":
    main()