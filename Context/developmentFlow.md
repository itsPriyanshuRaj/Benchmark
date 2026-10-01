# RAGBench AI — Development Flow & Progress

This document tracks all completed tasks, current status, and the roadmap for the RAGBench AI project.

---

## Current Phase: MVP (Minimal Viable Product)
**Objective**: Build a reliable, minimal end-to-end RAG pipeline without advanced benchmarking or complex agent features.

**Pipeline**:
$$\text{PDF Upload} \to \text{PDF Loader} \to \text{Recursive Chunking} \to \text{Embeddings} \to \text{Chroma Vector Store} \to \text{Top-K Retrieval} \to \text{Local LLM (Ollama)} \to \text{Answer + Sources}$$

---

## Progress Log

### Step 1: Context & Architecture Analysis
- [x] Read and analyzed `Benchmark_Context.md`.
- [x] Confirmed constraints:
  - Python + Flask only.
  - Step-by-step incremental development (no monolithic dumps).
  - Clear explanations of "why" before introducing components.
  - Verification after each step before moving to the next.

### Step 2: Dependency Management & Environment Setup
- [x] Created `requirements.txt` with minimal required packages:
  - `flask`: HTTP web server & API endpoints.
  - `pypdf`: Text extraction from PDF documents.
  - `langchain` & `langchain-community`: Document structures and utilities.
  - `sentence-transformers` & `langchain-huggingface`: Local dense vector embeddings.
  - `chromadb` & `langchain-chroma`: Local vector database for chunk indexing and search.
  - `langchain-ollama`: Connection to local Ollama LLMs.
  - `python-dotenv`: Environment configuration.
- [x] Created isolated virtual environment (`.venv`) using Python 3.11.9.
- [x] Installed all dependencies from `requirements.txt`.
- [x] Verified package installations via test imports (`flask`, `pypdf`, `sentence_transformers`, `chromadb`, `langchain_ollama`).

### Step 3: Configuration & Flask Application Skeleton
- [x] Created `config.py`:
  - Storage paths (`uploads/`, `chroma_db/`).
  - Default RAG parameters (embedding model `all-MiniLM-L6-v2`, local LLM `llama3`, chunk size `1000`, overlap `200`, top-k `3`).
  - Flask settings (`SECRET_KEY`, upload size limits).
- [x] Created initial `app.py`:
  - Flask application factory (`create_app`).
  - Directory initialization for uploads and Chroma persistence.
  - Basic `/` and `/health` routes.
- [x] Created placeholder `templates/index.html`.
- [x] Tested app factory and routes with HTTP status 200.

---

## Upcoming Tasks (MVP Roadmap)
- [ ] **Step 4: PDF Loader (`rag/loader.py`)**
  - Extract text and page metadata from uploaded PDF files using `pypdf`.
- [ ] **Step 5: Document Splitter (`rag/splitter.py`)**
  - Implement recursive character chunking with configurable size and overlap.
- [ ] **Step 6: Embeddings (`rag/embeddings.py`)**
  - Initialize local HuggingFace / Sentence Transformers embedding model.
- [ ] **Step 7: Vector Store (`rag/vectorstore.py`)**
  - Initialize and persist Chroma vector store; support adding chunks and querying top-K.
- [ ] **Step 8: Generator (`rag/generator.py`)**
  - Connect to local Ollama instance and build the QA prompt template.
- [ ] **Step 9: Flask Endpoints (`routes/upload.py`, `routes/query.py`)**
  - Implement `/upload` (process & index PDF) and `/query` (retrieve & answer).
- [ ] **Step 10: Minimal UI (`templates/index.html`)**
  - File upload form, question input, and display for generated answer + retrieved chunks.
- [ ] **Step 11: End-to-End Pipeline Verification**
  - Full smoke test with a sample PDF.
