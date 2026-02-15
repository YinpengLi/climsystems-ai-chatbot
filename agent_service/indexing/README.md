# Indexing Pipeline

## Purpose

Convert chunked documents into vector embeddings and build a
retrieval index for semantic search.

---

## Steps

1. Load approved documents (status=reviewed)
2. Run cluster_docs_embed.py to cluster documents
3. Human cluster review
4. Run cluster_name.py to name clusters
5. Human cluster name review
6. Build FAISS index with index.py
7. Write:
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
