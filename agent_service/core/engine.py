import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import ollama
import faiss #pip install faiss
import numpy as np
import re
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

EVIDENCE_ROOT = Path(__file__).resolve.parent().parent() / "evidence_library"
INDEX_DIR  = EVIDENCE_ROOT/ "04_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
RECORDS_FILE = INDEX_DIR / "records.jsonl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
LOG_DIR = EVIDENCE_ROOT / "06_logs"
SESSION_LOG = LOG_DIR / "session_log.jsonl"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("engine")
logger.setLevel(logging.INFO)  # change to DEBUG when needed

#LLM_PROVIDER = 
TOP_K = 8
OVERRETRIEVE = 8
MAX_CONTENT_CHARS= 1200
_FAISS_INDEX = None
_RECORDS = None
_EMBED_MODEL = None
LLM_MODEL = 'qwen2.5:7b-instruct'

NON_DOC_PAREN = re.compile(r"\((?!DOC:)[^)]+\)")  # parentheses not starting with DOC:

def scrub_non_doc_citations(s: str) -> str:
    return NON_DOC_PAREN.sub("", s or "")
CIT_RE = re.compile(r"\bDOC:[A-Za-z0-9_\-]+\.c\d+\b")
def extract_citations_from_text(text):
    return set(CIT_RE.findall(text or ""))

def validate_no_hallucinated_citations(answer_text, allowed):
    found = extract_citations_from_text(answer_text)
    invalid = sorted([c for c in found if c not in allowed])
    return invalid

def init_engine(index_path: Path, records_path: Path, embed_model_name: str):
    global _FAISS_INDEX, _RECORDS, _EMBED_MODEL
    if not INDEX_FILE.is_file():
        print(f"{INDEX_FILE} does not exist")
    if not RECORDS_FILE.is_file():
        print(f"{RECORDS_FILE} does not exist")

    _FAISS_INDEX = faiss.read_index(str(index_path))
    #RECORDS = []
    record_df = pd.read_json(path_or_buf = RECORDS_FILE, lines = True)
    _RECORDS = record_df.to_dict(orient="records")
    _RECORDS = normalize_records(_RECORDS)
    #or i, row in record_df.iterrows():
    #   _RECORDS.append(row)
    if len(_RECORDS) != _FAISS_INDEX.ntotal:
        print("Records length does not match index length")

    _EMBED_MODEL = SentenceTransformer(embed_model_name)
    return

def normalize_records(records):
    for r in records:
        meta = r.get("meta") or {}
        r["_doc_type"] = (meta.get("auto_doc_type") or meta.get("doc_type") or "").strip().lower()
        r["_juris"]    = (meta.get("auto_jurisdiction") or meta.get("jurisdiction") or "").strip().lower()
        r["title"] = meta.get("title")
        r["anchors"] = r.get("anchors")
        per = meta.get("auto_perils") or meta.get("perils") or meta.get("peril") or ""
        # if perils sometimes list
        if isinstance(per, list):
            r["_perils_set"] = {str(x).strip().lower() for x in per}
        else:
            # support comma/semicolon separated
            r["_perils_set"] = {p.strip().lower() for p in str(per).replace(";", ",").split(",") if p.strip()}

        r["_cluster"] = (meta.get("cluster_label") or "").strip().lower()
        #print(r)
    return records

def passes_filters(rec, filters):
    # filters are lists from Streamlit multiselect; empty list means "no filter"
    if filters.get("doc_type"):
        # OR semantics: match any selected doc_type
        if rec["_doc_type"] not in {d.strip().lower() for d in filters["doc_type"]}:
            return False

    if filters.get("jurisdiction"):
        if rec["_juris"] not in {j.strip().lower() for j in filters["jurisdiction"]}:
            return False

    if filters.get("peril"):
        wanted = {p.strip().lower() for p in filters["peril"]}
        if rec["_perils_set"].isdisjoint(wanted):
            return False

    cc = (filters.get("cluster_contains") or "").strip().lower()
    if cc:
        for kw in cc.split():
            if kw not in rec["_cluster"]:
                return False

    return True

def call_llm(provider, system, prompt, temperature):
    response = ollama.generate(
        model = provider,
        system = system,
        prompt = prompt,
        options = {"temperature": temperature}
    )
    return response["response"] #if isinstance(response, dict) else str(response)

def build_context():
    return
def call_llm_with_fixup(original_answer, allowed_citations, sources):
    system = (
        "You are a climate risk evidence assistant. "
        "You must ONLY cite from the allowed citation list. "
        "Do not invent citations."
    )
    allowed = ", ".join(sorted(allowed_citations))
    user = (
        "Rewrite the answer so that every citation is from the allowed list.\n\n"
        f"ALLOWED CITATIONS:\n{allowed}\n\n"
        f"SOURCES:\n{sources}\n\n"
        f"ORIGINAL ANSWER:\n{original_answer}\n"
    )
    return call_llm(LLM_MODEL, system, user, 0.2)


def run_answer(question, filters, top_k=8, overretrieve=60):
    fixup_attempted = False
    fixup_succeeded = False
    if _FAISS_INDEX is None or _RECORDS is None or _EMBED_MODEL is None:
        raise RuntimeError("Engine not initialised. Lifespan/startup did not run.")

    evidence_items = []
    q_vec = _EMBED_MODEL.encode(question, normalize_embeddings = True)
    q_vec = np.asarray(q_vec, dtype="float32")
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)
    distances, indices = _FAISS_INDEX.search(q_vec, OVERRETRIEVE)

    candidates = []
    for rank in range(min(OVERRETRIEVE, indices.shape[1])):
        idx = int(indices[0][rank])
        if idx < 0:
            continue
        rec = _RECORDS[idx]
        #print(rec, flush = True)
        #logger.info("rank=%s idx=%s score=%s rec_type=%s rec_keys=%s",
        #    rank, idx, float(distances[0][rank]), type(rec), list(rec.keys()) if isinstance(rec, dict) else None)
        #logger.info("rank=%s idx=%s score=%s rec_type=%s rec_keys=%s",
        #    rank, idx, float(distances[0][rank]), type(rec), list(rec.keys()) if isinstance(rec, dict) else None)
        score = float(distances[0][rank])

        if not passes_filters(rec, filters):
            continue

        candidates.append({"score": score, "rec": rec})

        if len(candidates) == int(TOP_K):
            break

    if not candidates:
        print("No evidence found under current filters.")
        print("Suggestion: remove filters or broaden query.")
        status = "no_evidence"
        return {
            "status": status,
            "answer": "",
            "allowed_citations_n": [],
            "citations_used": [],
            "invalid_citations_n": [],
            "fixup_attempted": fixup_attempted,
            "fixup_succeeded": fixup_succeeded,
            "evidence": evidence_items,
            "warnings": ["No evidence found"],
            "answer_chars": 0,
            "embed_model": EMBED_MODEL_NAME,
            "llm_model": LLM_MODEL
        }

    
    evidence_blocks = []
    allowed_citations  = set()
    total_chars = 0

    for item in candidates:
        rec = item["rec"]
        citation = rec.get("citation", "")
        text = scrub_non_doc_citations(rec.get("text", ""))[:MAX_CONTENT_CHARS]


        if citation in allowed_citations:
            continue
        allowed_citations.add(citation)

        block = f"SOURCE Citation: {citation} \nMETA doc_type={rec["_doc_type"]}; perils={rec["_perils_set"]}; jurisdiction={rec["_juris"]}; title={rec["title"]}; anchors={rec["anchors"]}\n{text}\n"

        if total_chars + len(block) > MAX_CONTENT_CHARS:
            break

        evidence_blocks.append(block)
        total_chars+=len(block)
    evidence_context = "\n---\n".join(evidence_blocks)
    #print(f"EVIDENCE CONTEXT: {evidence_context}")

    #evidence_context, allowed_citations, evidence_items = build_context(candidates)
    for item in candidates:
        #print(item)
        evidence_items.append(f"-{item["rec"]["citation"]}|{item["rec"]["meta"]["title"]}|{item["rec"]["anchors"]}")

    SYSTEM_PROMPT = """
    You are a climate risk evidence assistant.
    You must answer ONLY using the provided sources.
    If the sources are insufficient, say so clearly and ask for what is missing.
    Every factual statement must include at least one citation in parentheses,
    using the exact SOURCE IDs like (DOC:xxxx.cxx). Write these citations in your response. 
    Do NOT invent citations.
    """
    allowed_list = "\n".join(f"- {c}" for c in sorted(allowed_citations))
    USER_PROMPT = f"""QUESTION:\n{question}\n\nALLOWED CITATIONS:\n{allowed_list}\n\nSOURCES:\n{evidence_context}\n\nINSTRUCTIONS:\n- Every factual statement must include at least one citation in parentheses , using the exact SOURCE IDs like (DOC:xxxx.cxx) from the ALLOWED CITATIONS list I have given.\n- Only use citations from the ALLOWED CITATIONS list. Copy exactly.\n"""

    draft_answer = call_llm(
        provider = LLM_MODEL,
        #model = 'gemma3:1b',
        system = SYSTEM_PROMPT,
        prompt = USER_PROMPT,
        temperature = 0.0
    )

    invalid = validate_no_hallucinated_citations(answer_text=draft_answer, allowed = allowed_citations)
    citations_used = extract_citations_from_text(draft_answer)
    if invalid or citations_used == set():
        print(f"[WARN] Model produced citations not in sources: {invalid}")
        draft_answer = call_llm_with_fixup(draft_answer, allowed_citations, evidence_context)
        invalid2 = validate_no_hallucinated_citations(draft_answer, allowed_citations)
        fixup_attempted = True
        citations_used = extract_citations_from_text(draft_answer)
        if invalid2 or citations_used == set():
            status = "failed_invalid_citations"
            return {
                "status": status,
                "answer": "",
                "allowed_citations_n": list(allowed_citations),
                "citations_used": list(citations_used),
                "invalid_citations_n": invalid.extend(invalid2),
                "fixup_attempted": fixup_attempted,
                "fixup_succeeded": fixup_succeeded,
                "evidence": evidence_items,
                "warnings": ["Answer rejected due to invalid citations.", invalid2],
                "answer_chars": 0,
                "embed_model": EMBED_MODEL_NAME,
                "llm_model": LLM_MODEL
            }
        
        else:
           fixup_succeeded = True
           status = "fixed_citations"
        
    else:
        status = "ok"

    return {
        "status": status,
        "answer": draft_answer,
        "allowed_citations_n": list(allowed_citations),
        "citations_used": list(citations_used),
        "invalid_citations_n": invalid,
        "fixup_attempted": fixup_attempted,
        "fixup_succeeded": fixup_succeeded,
        "evidence": evidence_items,
        "warnings": [],
        "answer_chars": len(draft_answer),
        "embed_model": EMBED_MODEL_NAME,
        "llm_model": LLM_MODEL
    }