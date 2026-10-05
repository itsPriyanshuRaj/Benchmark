import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Storage paths
UPLOAD_FOLDER = BASE_DIR / "uploads"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# RAG Settings (MVP defaults)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 3

# Flask Settings
SECRET_KEY = os.getenv("SECRET_KEY", "ragbench-mvp-secret-key")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit

# Phase 2: Chunking Benchmark Presets
DEFAULT_CHUNKING_EXPERIMENTS = [
    {"id": "small", "name": "Small Chunks (300 / 50)", "chunk_size": 300, "chunk_overlap": 50},
    {"id": "medium", "name": "Medium Chunks (800 / 150)", "chunk_size": 800, "chunk_overlap": 150},
    {"id": "large", "name": "Large Chunks (1500 / 300)", "chunk_size": 1500, "chunk_overlap": 300}
]

# Phase 2: Top-K Benchmark Presets
DEFAULT_TOP_K_PRESETS = [1, 3, 5]

# Phase 2: Embedding Models for Benchmarking
AVAILABLE_EMBEDDING_MODELS = [
    {"id": "minilm-l6", "name": "all-MiniLM-L6-v2 (Fast / 6 layers)", "model_name": "all-MiniLM-L6-v2"},
    {"id": "minilm-l12", "name": "all-MiniLM-L12-v2 (Deep / 12 layers)", "model_name": "all-MiniLM-L12-v2"}
]

# Phase 3: Retrieval Quality (Reranking & Hybrid) Settings
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_CANDIDATES = 10
RRF_K_CONSTANT = 60

# Phase 4: Evaluation & LLM-as-a-Judge Settings
DEFAULT_EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", DEFAULT_OLLAMA_MODEL)
EVALUATION_METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
FAITHFULNESS_THRESHOLD = 0.7
ANSWER_RELEVANCE_THRESHOLD = 0.7
CONTEXT_PRECISION_THRESHOLD = 0.6



