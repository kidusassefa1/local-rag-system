# Local RAG Backend

FastAPI service that:
- ingests text into Postgres + pgvector
- queries by semantic similarity using Ollama embeddings

## Run (local or server)
1) Copy `.env.example` to `.env` and fill values
2) Install dependencies
3) Start server