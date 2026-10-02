from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """
    Splits documents into smaller text chunks for embedding.
    Uses chunk_size and chunk_overlap from config by default.
    """
    if chunk_size is None:
        chunk_size = config.DEFAULT_CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.DEFAULT_CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)
    return chunks
