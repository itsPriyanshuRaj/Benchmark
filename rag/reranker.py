from sentence_transformers import CrossEncoder
import config

# In-memory cache for reranker model
_RERANKER_CACHE = {}

def get_reranker_model(model_name=None):
    """
    Returns a cached Cross-Encoder model for reranking.
    Loads model weights once into memory and reuses the instance.
    """
    if model_name is None:
        model_name = config.DEFAULT_RERANKER_MODEL

    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)

    return _RERANKER_CACHE[model_name]

def rerank_documents(query, documents, top_k=None, model_name=None):
    """
    Re-scores and re-ranks retrieved candidate documents using a Cross-Encoder.
    Returns the top_k most relevant documents with rerank_score attached to metadata.
    """
    if not documents:
        return []

    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    model = get_reranker_model(model_name)

    # Form (query, document_text) pairs for cross-attention
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)

    scored_docs = []
    for doc, score in zip(documents, scores):
        # Create a shallow copy of metadata to avoid mutating original
        new_meta = dict(doc.metadata)
        new_meta["rerank_score"] = round(float(score), 4)
        doc.metadata = new_meta
        scored_docs.append((doc, float(score)))

    # Sort descending by cross-encoder score
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return top_k documents
    return [doc for doc, score in scored_docs[:top_k]]

