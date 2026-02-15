
---

# 4️⃣ `api/README.md`

```markdown
# API Service

## Purpose

Serve the climate evidence AI via HTTP.

This layer:
- Accepts questions
- Applies filters
- Retrieves evidence
- Generates answer
- Enforces citation validation
- Logs session data

---

## Endpoints

### GET /health

Returns:
{
  "status": "ok"
}

---

### POST /ask

Request:

{
  "question": "...",
  "doc_type": "",
  "jurisdiction": "",
  "peril": "",
  "cluster_contains": "",
  "top_k": 8
}

Response:

{
  "answer": "...",
  "citations": [...],
  "status": "ok"
}

---

### POST /reload

Reloads index + records into memory.

---

## Architecture

- Engine loaded at startup
- Global cache for:
  - FAISS index
  - records dataframe
  - embedding model

---

## Citation Enforcement

- All answers validated against allowed citation IDs
- Hallucinated citations trigger repair loop

---

## Logging

All sessions written to:

evidence_library/06_logs/session_log.jsonl

---

## Start Server

```bash
uvicorn agent_service.api.app:app --reload
