from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """
    Loads a PDF file and returns a list of pages as documents.
    Each document contains the page text and metadata (like page number).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File {path.name} is not a valid PDF.")

    loader = PyPDFLoader(str(path))
    documents = loader.load()
    return documents
