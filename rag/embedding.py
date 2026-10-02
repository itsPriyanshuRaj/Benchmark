from langchain_huggingface import HuggingFaceEmbeddings
import config

def get_embedding_model(model_name=None):
    """
    Loads and returns the local Sentence Transformers embedding model.
    Uses the default model from config.py if no model name is passed.
    """
    if model_name is None:
        model_name = config.DEFAULT_EMBEDDING_MODEL

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

