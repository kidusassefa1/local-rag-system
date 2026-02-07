# Local RAG System (GPU-Accelerated, Homelab-Based)

This project is a **locally deployed Retrieval-Augmented Generation (RAG) system** built to gain hands-on, real-world experience with AI/ML engineering, backend systems, and infrastructure.

The system runs fully on a **self-hosted Proxmox homelab** using LXC containers and a GPU-enabled VM. It demonstrates how modern AI systems are designed, deployed, and integrated in a production-like environment without relying on managed cloud services.

---

## 🔍 What This Project Does

- Ingests text documents and stores them as vector embeddings
- Performs semantic search using **pgvector** in PostgreSQL
- Uses a **local GPU-powered LLM (Ollama)** to answer questions
- Grounds answers strictly in retrieved context
- Returns citations pointing to the exact stored chunks used

This is a full **end-to-end RAG pipeline**:
> ingestion → embedding → vector search → context building → LLM response

---

## 🧱 Architecture Overview

**Infrastructure (Proxmox):**
- **PostgreSQL LXC**
  - Debian-based
  - pgvector extension enabled
- **Backend API LXC**
  - FastAPI application
  - Handles ingestion, retrieval, and chat endpoints
- **GPU VM**
  - Ubuntu-based
  - Ollama running locally
  - NVIDIA GPU passthrough for inference and embeddings

**Data Flow:**
1. Text is chunked and embedded using a local embedding model
2. Embeddings are stored in PostgreSQL (pgvector)
3. User queries are embedded and compared via cosine similarity
4. Top relevant chunks are retrieved
5. Retrieved context is injected into an LLM prompt
6. The model generates an answer with citations

---

## 🧠 Tech Stack

**Backend**
- Python
- FastAPI
- Uvicorn

**AI / ML**
- Ollama (local inference)
- `nomic-embed-text` (embeddings)
- `llama3.1` (chat model)

**Database**
- PostgreSQL
- pgvector (vector similarity search)

**Infrastructure**
- Proxmox VE
- LXC containers
- GPU passthrough (NVIDIA)

**Dev & Ops**
- Git + GitHub
- Linux (Debian / Ubuntu)
- Virtual environments (`venv`)

---

## 📂 Project Structure

local-rag-system/
├── backend/
│ ├── app/
│ │ ├── main.py # FastAPI endpoints
│ │ ├── ollama.py # Ollama integration
│ │ ├── db.py # PostgreSQL connection
│ │ ├── schema.py # pgvector schema
│ │ └── config.py # Environment config
│ ├── requirements.txt
│ ├── .env.example
│ └── README.md
└── README.md