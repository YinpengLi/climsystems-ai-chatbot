from datetime import datetime, timezone
import json
from pathlib import Path

EVIDENCE_ROOT = Path(r"C:\Users\Yinpeng Li\CLIMsystems Dropbox\Yinpeng Li\climsystems_ai\evidence_library")
INDEX_DIR  = EVIDENCE_ROOT/ "04_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
RECORDS_FILE = INDEX_DIR / "records.jsonl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
LOG_DIR = EVIDENCE_ROOT / "06_logs"
SESSION_LOG = LOG_DIR / "session_log.jsonl"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_session (request_id, question, filters, result, status, latency_ms):
    record = {
        "request_id": request_id, 
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "filters": filters,
        "retrieved_chunks": len(result["evidence"]),
        "status": status,
        "citations_used": result["citations_used"],
        "citations_used_n": len(result["citations_used"] or ""),
        "allowed_citations_n": len(result["allowed_citations_n"] or ""),
        "invalid_citations_n": len(result["invalid_citations_n"] or ""),
        "fixup_attempted": result["fixup_attempted"],
        "fixup_succeeded": result["fixup_succeeded"],
        "evidence": result["evidence"],
        "warnings": result["warnings"],
        "answer_chars": result["answer_chars"],
        "latency": latency_ms,
        "embed_model": result["embed_model"],
        "llm_model": result["llm_model"]
    }

    with SESSION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")