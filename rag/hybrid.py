import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

import config
from rag.vectorStore import get_vectorstore, query_similar_documents

def tokenize(text):
    """
    Splits text into lowercase alphanumeric tokens for BM25.
    """
    return re.findall(r"\w+", text.lower())

def hybrid_search(query, top_k=None, collection_name="ragbench_docs", rrf_k=None):
    """
    Executes Hybrid Search combining Dense Vector Search (Chroma) and
    Sparse Keyword Search (BM25) fused via Reciprocal Rank Fusion (RRF).
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K
    if rrf_k is None:
        rrf_k = config.RRF_K_CONSTANT

    candidates_n = max(top_k * 2, config.DEFAULT_RERANK_CANDIDATES)

    # 1. Dense Vector Search (Semantic similarity)
    dense_docs = query_similar_documents(query, top_k=candidates_n, collection_name=collection_name)

    # 2. Sparse BM25 Search (Keyword matching)
    vectorstore = get_vectorstore(collection_name=collection_name)
    raw_data = vectorstore.get()

    if not raw_data or not raw_data.get("documents"):
        return dense_docs[:top_k]

    all_texts = raw_data["documents"]
    all_metas = raw_data.get("metadatas") or [{}] * len(all_texts)
    all_docs = [Document(page_content=t, metadata=dict(m or {})) for t, m in zip(all_texts, all_metas)]

    tokenized_corpus = [tokenize(doc.page_content) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_ranked = sorted(zip(all_docs, bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_docs = [doc for doc, score in bm25_ranked[:candidates_n] if score > 0]

    # 3. Reciprocal Rank Fusion (RRF)
    doc_scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs, 1):
        content = doc.page_content
        doc_map[content] = doc
        score = 1.0 / (rrf_k + rank)
        doc_scores[content] = doc_scores.get(content, 0.0) + score
        doc.metadata["dense_rank"] = rank

    for rank, doc in enumerate(bm25_docs, 1):
        content = doc.page_content
        if content not in doc_map:
            doc_map[content] = doc
        score = 1.0 / (rrf_k + rank)
        doc_scores[content] = doc_scores.get(content, 0.0) + score
        doc_map[content].metadata["bm25_rank"] = rank

    sorted_contents = sorted(doc_scores.keys(), key=lambda c: doc_scores[c], reverse=True)

    results = []
    for c in sorted_contents[:top_k]:
        doc = doc_map[c]
        doc.metadata["rrf_score"] = round(doc_scores[c], 5)
        results.append(doc)

    return results

