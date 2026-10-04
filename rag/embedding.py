from langchain_huggingface import HuggingFaceEmbeddings
import config

# In-memory model cache to prevent reloading PyTorch weights from disk repeatedly
_EMBEDDING_CACHE = {}

def get_embedding_model(model_name=None):
    """
    Returns a cached Sentence Transformers embedding model instance.
    Loads weights from disk only once; subsequent calls return the in-memory instance instantly.
    """
    if model_name is None:
        model_name = config.DEFAULT_EMBEDDING_MODEL

    if model_name not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[model_name] = HuggingFaceEmbeddings(model_name=model_name)

    return _EMBEDDING_CACHE[model_name]

