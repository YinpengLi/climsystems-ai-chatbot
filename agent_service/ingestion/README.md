
---

# 2️⃣ `ingestion/README.md`

```markdown
# Ingestion Pipeline

## Purpose

This module manages document intake, ID assignment, metadata governance,
and promotion to canonical evidence sources.

This stage ensures:

- Unique doc_id assignment
- No duplicate documents
- Clean metadata structure
- Controlled promotion to immutable storage

---

## Workflow

1. Place documents in inbox folder
2. Run doc_id_generator.py
3. Review documents.csv
4. Confirm doc_id + set status=reviewed
5. Run promote_to_sources.py
6. Run build_profiles.py
7. Run chunk_documents.py
8. Run language_detection.py

---

## Scripts

### doc_id_generator.py

- Generates stable human-readable IDs
- Detects duplicates via SHA256
- Proposes metadata

### promote_to_sources.py

- Moves approved documents to 01_sources
- Never overwrites
- Logs all promotions

### build_profiles.py

- Extracts:
  - metadata
  - document structure
  - detected title
  - keywords

### chunk_documents.py

- Splits text into retrieval-sized chunks
- Generates citation IDs (DOC:xxx.c12)
- Writes JSONL chunk files

---

## Metadata Governance

documents.csv is the canonical metadata register.

Important columns:
- doc_id
- status
- doc_type
- perils
- jurisdiction
- title
- cluster_label

---

## Rules

- doc_id is permanent once approved.
- 01_sources is immutable.
- No serving logic allowed in ingestion layer.

---
