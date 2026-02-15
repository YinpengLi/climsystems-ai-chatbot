from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager
from core.engine import init_engine
from core.engine import run_answer
from core.log_maker import log_session
import uuid 
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
logger = logging.getLogger("uvicorn.error")
EVIDENCE_ROOT = Path(__file__).resolve.parent().parent() / "evidence_library"
INDEX_DIR  = EVIDENCE_ROOT/ "04_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
RECORDS_FILE = INDEX_DIR / "records.jsonl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
LOG_DIR = EVIDENCE_ROOT / "06_logs"
SESSION_LOG = LOG_DIR / "session_log.jsonl"
LOG_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app:FastAPI):
    try:
        init_engine(
            index_path = INDEX_FILE,
            records_path  = RECORDS_FILE,
            embed_model_name = EMBED_MODEL_NAME,
        )
        print("[app] Engine initialised")
        yield
    finally:
        # --- SHUTDOWN ---
        print("[lifespan] Shutdown")

app = FastAPI(
    title = "ClimSystems Climate Risk Evidence API",
    version = "0.1.0",
    lifespan = lifespan,
)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Use incoming request-id if provided (good for proxies)
        request_id = request.headers.get("X-Request-ID")

        # 2. Otherwise generate one
        if not request_id:
            request_id = str(uuid.uuid4())

        # 3. Attach to request state (FastAPI-native)
        request.state.request_id = request_id

        # 4. Process request
        response = await call_next(request)

        # 5. Echo back to client (best practice)
        response.headers["X-Request-ID"] = request_id
        return response
    
app.add_middleware(RequestIDMiddleware)

class AskRequest(BaseModel):
    question: str
    doc_type: Optional[list] = []
    jurisdiction: Optional[list] = []
    peril: Optional[list] = []
    cluster_contains: Optional[str] = ""

    top_k: int = 8

class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    status: str
    

@app.get("/health")
def health():
    return {"status" : "ok"}


@app.post("/ask", response_model = AskResponse)
def ask(req: AskRequest, request: Request):
    
    try:
        start_ts = time.perf_counter()
        request_id = getattr(request.state, "request_id", "")
        filters: Dict[str, list] = {
            "doc_type": req.doc_type or [],
            "jurisdiction": req.jurisdiction or [],
            "peril": req.peril or [],
            "cluster_contains": req.cluster_contains or "",
        }

        result = run_answer(
            question  = req.question,
            filters = filters,
            top_k = req.top_k
        )
        latency_ms = int((time.perf_counter() - start_ts) * 1000)
        print(result["answer"])
        print(result["citations_used"])

        log_session(
            request_id, 
            req.question, 
            [
                req.doc_type,
                req.jurisdiction, 
                req.peril, 
                req.cluster_contains
            ], 
            result,
            result["status"],
            latency_ms

        )
        return AskResponse(
            answer = result["answer"],
            citations = result["citations_used"],
            status = result["status"]
        )
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_ts) * 1000)
        log_session(
            request_id, 
            req.question, 
            [
                req.doc_type,
                req.jurisdiction, 
                req.peril, 
                req.cluster_contains
            ], 
            "",
            "error",
            latency_ms

        )
        raise HTTPException(status_code=500, detail = str(e))
    
@app.post("/reload")
def reload_engine():
    try:
        init_engine(
            index_path = INDEX_FILE,
            records_path = RECORDS_FILE,
            embed_model_name = EMBED_MODEL_NAME,
        )
        return {"status": "reloaded"}
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    