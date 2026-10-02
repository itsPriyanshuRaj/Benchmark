from langchain_chroma import Chroma
from rag.embedding import get_embedding_model
import config

def get_vectorstore():
    """
    Connects to or creates the local Chroma vector database on disk.
    """
    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        collection_name="ragbench_docs",
        embedding_function=embedding_model,
        persist_directory=str(config.CHROMA_PERSIST_DIR)
    )
    return vectorstore

def add_documents(chunks):
    """
    Stores document chunks into the Chroma vector database.
    """
    if not chunks:
        return 0

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)

def query_similar_documents(query, top_k=None):
    """
    Searches the database and returns the top-K most relevant chunks for a question.
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=top_k)

