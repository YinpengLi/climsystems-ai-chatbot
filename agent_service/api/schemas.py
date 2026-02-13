from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class FilterItems(BaseModel):
    doc_type: str
    jurisdiction: str
    peril: str
    cluster_contains: str

class AskRequest(BaseModel):
    question: str
    top_k: int
    filters: FilterItems

class EvidenceItem(BaseModel):
    citation: str
    doc_id: str
    title: str
    anchors: list
    score: float

class AskResponse(BaseModel):
    status: str
    answer: str
    citations_used: list[str]
    evidence: list[EvidenceItem]
    warnings: list[str]

