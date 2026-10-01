import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Storage paths
UPLOAD_FOLDER = BASE_DIR / "uploads"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# RAG Settings (MVP defaults)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 3

# Flask Settings
SECRET_KEY = os.getenv("SECRET_KEY", "ragbench-mvp-secret-key")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit
