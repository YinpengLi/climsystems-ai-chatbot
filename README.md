# ClimSystems Climate Risk Evidence AI

## Overview

This system provides an evidence-grounded AI assistant for climate risk,
regulatory disclosure, methodology support, and technical knowledge retrieval with provided sources.

The architecture separates:

- Ingestion (document governance + metadata)
- Indexing (embedding + clustering)
- Engine (retrieval + answer generation)
- API (serving layer)
- Logging & Metrics (governance + monitoring)

All answers are citation-bound to internal evidence documents.

---

## Folder layout

evidence_library/
│
├── 00_inbox/            # Raw documents
├── 01_sources/          # Approved canonical sources (immutable)
├── 02_normalised/       # Extracted text + structure
├── 03_chunks/           # Chunked text for embedding
├── 04_index/            # FAISS index + records.jsonl
├── 05_metadata/         # documents.csv
├── 06_logs/             # session logs

agent_service/
│
├── ingestion/
├── indexing/
├── core/
├── api/
└── ui/

---

## Workflow

1. Drop documents into inbox
2. Ingestion (ingestion folder)
3. Indexing (indexing folder)
4. Serve API (api folder)
5. Start the UI (ui folder)
6. Use AI chatbot

## Web UI (Streamlit)

The system includes an optional Streamlit-based user interface
for interactive querying of the Climate Risk Evidence AI.

The UI connects to the FastAPI backend.

The UI and API runs at http://127.0.0.1:8000

---

### 1. Start the API

In terminal:

```bash
uvicorn agent_service.api.app:app --reload
```

### 2. Start the UI
```bash
streamlit run streamlit_app.py
```

---

### View the requirements
```bash
pip install -r requirements.txt
```
