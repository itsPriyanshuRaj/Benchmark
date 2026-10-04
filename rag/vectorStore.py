from langchain_chroma import Chroma
from rag.embedding import get_embedding_model
import config

def get_vectorstore(collection_name="ragbench_docs", embedding_model_name=None):
    """
    Connects to or creates a named Chroma collection on disk.
    Supports specifying a custom embedding model for multi-model benchmarks.
    """
    embedding_model = get_embedding_model(embedding_model_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(config.CHROMA_PERSIST_DIR)
    )
    return vectorstore

def add_documents(chunks, collection_name="ragbench_docs", embedding_model_name=None):
    """
    Stores document chunks into the specified Chroma collection using the chosen embedding model.
    """
    if not chunks:
        return 0

    vectorstore = get_vectorstore(collection_name=collection_name, embedding_model_name=embedding_model_name)
    vectorstore.add_documents(chunks)
    return len(chunks)

def query_similar_documents(query, top_k=None, collection_name="ragbench_docs", embedding_model_name=None):
    """
    Searches a specific collection and returns the top-K most relevant chunks.
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    vectorstore = get_vectorstore(collection_name=collection_name, embedding_model_name=embedding_model_name)
    return vectorstore.similarity_search(query, k=top_k)

