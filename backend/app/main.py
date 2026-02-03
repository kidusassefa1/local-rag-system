from typing import List, Optional
import requests
import uuid
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import OLLAMA_BASE_URL
from .db import conn
from .ollama import embed
from .schema import ensure_schema


app = FastAPI(title="Local RAG System Backend", version="0.1")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    overlap = min(overlap, chunk_size - 1)
    out = []
    i, n = 0, len(text)

    while i < n:
        j = min(n, i + chunk_size)
        piece = text[i:j].strip()

        if piece:
            out.append(piece)
        
        if j >= n:
            break

        next_i = j - overlap
        if next_i <= i:
            next_i = i + 1 
        i = next_i

    return out

def to_pgvector_str(emb) -> str:
    # Flatten if embedding is nested like [[...]]
    if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
        if len(emb) == 1:
            emb = emb[0]
        else:
            raise ValueError("Embedding is 2D with multiple rows; expected 1D vector.")

    if not isinstance(emb, list) or not emb:
        raise ValueError("Embedding is empty or not a list.")

    # Convert to pgvector literal string: [0.1,0.2,...]
    return "[" + ",".join(str(float(x)) for x in emb) + "]"


class IngestTextRequest(BaseModel):
    doc_name: str
    text: str

class IngestResponse(BaseModel):
    document_id: str
    chunks_ingested: int

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class Hit(BaseModel):
    chunk_id: int
    doc_name: str
    chunk_index: Optional[int]
    content: str
    score: float

class QueryResponse(BaseModel):
    query: str
    hits: List[Hit]


@app.get("/health")
def health():
    # postgres check
    try:
        ensure_schema()
        with conn() as c:
            with c.cursor() as cur:
                cur.execute("SELECT 1;")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")
    
    # Ollama check
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama connection error: {e}")
    
    return {"status": "ok"}

@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest):
    ensure_schema()
    chunks = chunk_text(req.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text to ingest")
    
    document_id = str(uuid.uuid4())
    with conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (id, name) VALUES (%s, %s);",
                (document_id, req.doc_name),
            )
            
            for idx, chunk in enumerate(chunks):
                emb = embed(chunk)
                vec_str = to_pgvector_str(emb)
                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (document_id, idx, chunk, vec_str),
                )
        c.commit()

    return IngestResponse(document_id=document_id, chunks_ingested=len(chunks))

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    ensure_schema()
    q_emb = embed(req.query)
    vec_str = to_pgvector_str(q_emb)
    
    with conn() as c:
        with c.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT 
                    c.id AS chunk_id,
                    d.name AS doc_name,
                    c.chunk_index,
                    c.content,
                    (c.embedding <=> %s::vector) AS distance
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY (c.embedding <=> %s::vector)
                LIMIT %s;
                """,
                (vec_str, vec_str, req.top_k))
            rows = cur.fetchall()
    
    hits = []
    for row in rows:
        dist = float(row["distance"])
        score = 1.0 / (1.0 + dist)  # convert distance to score
        hits.append(
            Hit(
                chunk_id=int(row["chunk_id"]),
                doc_name=str(row["doc_name"]),
                chunk_index=int(row["chunk_index"]) if row["chunk_index"] is not None else None,
                content=str(row["content"]),
                score=score,
            )
        )

    return QueryResponse(query=req.query, hits=hits)