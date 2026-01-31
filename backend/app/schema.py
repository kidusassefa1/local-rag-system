from .db import conn

def ensure_schema():
    with conn() as c:
        with c.cursor() as cur:
            cur.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS documents (
              id UUID PRIMARY KEY,
              name TEXT NOT NULL,
              created_at TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS chunks (
              id BIGSERIAL PRIMARY KEY,
              document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
              chunk_index INT,
              content TEXT NOT NULL,
              embedding vector(768),
              created_at TIMESTAMPTZ DEFAULT now()
            );

            -- index for vector search (we'll tune later)
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """)
            c.commit()
