# Indexing Pipeline

## Purpose

Convert chunked documents into vector embeddings and build a
retrieval index for semantic search.

---

## Steps

1. Load approved documents (status=reviewed)
2. Load chunk files
3. Embed chunk text
4. Build FAISS index
5. Write:
   - index.faiss
   - records.jsonl
   - manifest.json

---

## Output Files

04_index/
│
├── index.faiss
├── records.jsonl
└── manifest.json

---

## records.jsonl Structure

Each record contains:

{
  "id": "DOC:xxx.c12",
  "text": "...",
  "doc_id": "...",
  "chunk_id": 12,
  "citation": "DOC:xxx.c12",
  "anchors": [...],
  "meta": {...}
}

---

## Clustering (Optional)

cluster.py performs:
- UMAP dimensionality reduction
- HDBSCAN clustering
- Writes cluster_id to metadata

---

## Rebuild Index

```bash
python -m agent_service.indexing.index
