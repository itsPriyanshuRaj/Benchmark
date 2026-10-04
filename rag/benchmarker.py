import time
import config
from rag.splitter import split_documents
from rag.vectorStore import add_documents, query_similar_documents
from rag.generator import generate_answer
from rag.reranker import rerank_documents
from rag.hybrid import hybrid_search

# ================= 1. CHUNKING EXPERIMENTS =================

def get_chunk_collection_name(exp_id):
    return f"exp_chunk_{exp_id}"

def index_experiments(documents, configs=None):
    """
    Indexes the same document across multiple chunking configurations.
    """
    if configs is None:
        configs = config.DEFAULT_CHUNKING_EXPERIMENTS

    results = []
    for exp in configs:
        exp_id = exp["id"]
        chunk_size = exp["chunk_size"]
        chunk_overlap = exp["chunk_overlap"]
        collection_name = get_chunk_collection_name(exp_id)

        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        indexed_count = add_documents(chunks, collection_name=collection_name)

        results.append({
            "id": exp_id,
            "name": exp["name"],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunks_count": indexed_count
        })

    return results

def run_chunking_comparison(question, top_k=None, configs=None):
    """
    Runs the question against each chunking experiment collection and measures latency.
    """
    if configs is None:
        configs = config.DEFAULT_CHUNKING_EXPERIMENTS
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    comparisons = []
    for exp in configs:
        exp_id = exp["id"]
        collection_name = get_chunk_collection_name(exp_id)

        # Measure Retrieval Latency
        t_start = time.perf_counter()
        retrieved_chunks = query_similar_documents(question, top_k=top_k, collection_name=collection_name)
        t_retrieved = time.perf_counter()
        retrieval_ms = round((t_retrieved - t_start) * 1000, 1)

        # Measure Generation Latency
        answer = generate_answer(question, retrieved_chunks)
        t_generated = time.perf_counter()
        generation_ms = round((t_generated - t_retrieved) * 1000, 1)
        total_ms = round((t_generated - t_start) * 1000, 1)

        chunks_data = []
        for chunk in retrieved_chunks:
            chunks_data.append({
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "Unknown"),
                "page": chunk.metadata.get("page", 0) + 1,
                "char_count": len(chunk.page_content)
            })

        comparisons.append({
            "id": exp_id,
            "name": exp["name"],
            "chunk_size": exp["chunk_size"],
            "chunk_overlap": exp["chunk_overlap"],
            "answer": answer,
            "retrieved_chunks": chunks_data,
            "latency": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms
            }
        })

    return comparisons

# ================= 2. TOP-K EXPERIMENTS =================

def run_topk_comparison(question, k_values=None, collection_name="ragbench_docs"):
    """
    Compares different retrieval depths (e.g. K=1, K=3, K=5) on the same collection.
    """
    if k_values is None:
        k_values = config.DEFAULT_TOP_K_PRESETS

    comparisons = []
    for k in k_values:
        t_start = time.perf_counter()
        retrieved_chunks = query_similar_documents(question, top_k=k, collection_name=collection_name)
        t_retrieved = time.perf_counter()
        retrieval_ms = round((t_retrieved - t_start) * 1000, 1)

        answer = generate_answer(question, retrieved_chunks)
        t_generated = time.perf_counter()
        generation_ms = round((t_generated - t_retrieved) * 1000, 1)
        total_ms = round((t_generated - t_start) * 1000, 1)

        chunks_data = []
        for chunk in retrieved_chunks:
            chunks_data.append({
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "Unknown"),
                "page": chunk.metadata.get("page", 0) + 1,
                "char_count": len(chunk.page_content)
            })

        comparisons.append({
            "k": k,
            "name": f"Top-K = {k}",
            "answer": answer,
            "retrieved_chunks": chunks_data,
            "latency": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms
            }
        })

    return comparisons

# ================= 3. MULTI-EMBEDDING EXPERIMENTS =================

def get_embedding_collection_name(model_id):
    return f"exp_embed_{model_id}"

def index_embedding_experiments(documents, models=None):
    """
    Indexes the document across multiple embedding models.
    """
    if models is None:
        models = config.AVAILABLE_EMBEDDING_MODELS

    results = []
    # Standard chunking for embedding comparison
    chunks = split_documents(documents, chunk_size=config.DEFAULT_CHUNK_SIZE, chunk_overlap=config.DEFAULT_CHUNK_OVERLAP)

    for m in models:
        model_id = m["id"]
        model_name = m["model_name"]
        collection_name = get_embedding_collection_name(model_id)

        t_start = time.perf_counter()
        indexed_count = add_documents(chunks, collection_name=collection_name, embedding_model_name=model_name)
        indexing_ms = round((time.perf_counter() - t_start) * 1000, 1)

        results.append({
            "id": model_id,
            "name": m["name"],
            "model_name": model_name,
            "chunks_count": indexed_count,
            "indexing_ms": indexing_ms
        })

    return results

def run_embedding_comparison(question, top_k=None, models=None):
    """
    Compares retrieval and generation across different embedding models.
    """
    if models is None:
        models = config.AVAILABLE_EMBEDDING_MODELS
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    comparisons = []
    for m in models:
        model_id = m["id"]
        model_name = m["model_name"]
        collection_name = get_embedding_collection_name(model_id)

        t_start = time.perf_counter()
        retrieved_chunks = query_similar_documents(
            question,
            top_k=top_k,
            collection_name=collection_name,
            embedding_model_name=model_name
        )
        t_retrieved = time.perf_counter()
        retrieval_ms = round((t_retrieved - t_start) * 1000, 1)

        answer = generate_answer(question, retrieved_chunks)
        t_generated = time.perf_counter()
        generation_ms = round((t_generated - t_retrieved) * 1000, 1)
        total_ms = round((t_generated - t_start) * 1000, 1)

        chunks_data = []
        for chunk in retrieved_chunks:
            chunks_data.append({
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "Unknown"),
                "page": chunk.metadata.get("page", 0) + 1,
                "char_count": len(chunk.page_content)
            })

        comparisons.append({
            "id": model_id,
            "name": m["name"],
            "model_name": model_name,
            "answer": answer,
            "retrieved_chunks": chunks_data,
            "latency": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms
            }
        })

    return comparisons

# ================= 4. RETRIEVAL QUALITY (HYBRID & RERANKING) =================

def format_chunks_data(chunks):
    """
    Standardizes chunk data extraction with scores and ranks.
    """
    data = []
    for chunk in chunks:
        meta = dict(chunk.metadata or {})
        item = {
            "content": chunk.page_content,
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", 0) + 1,
            "char_count": len(chunk.page_content)
        }
        if "rerank_score" in meta:
            item["rerank_score"] = meta["rerank_score"]
        if "rrf_score" in meta:
            item["rrf_score"] = meta["rrf_score"]
        if "dense_rank" in meta:
            item["dense_rank"] = meta["dense_rank"]
        if "bm25_rank" in meta:
            item["bm25_rank"] = meta["bm25_rank"]
        data.append(item)
    return data

def run_retrieval_quality_comparison(question, top_k=None, collection_name="ragbench_docs"):
    """
    Compares 4 Retrieval Strategies side-by-side:
    1. Standard Dense Vector Search (Baseline)
    2. Hybrid Search (Dense + BM25 Fusion)
    3. Vector Search + Cross-Encoder Reranker
    4. Hybrid Search + Cross-Encoder Reranker
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K
    candidates_n = max(top_k * 2, config.DEFAULT_RERANK_CANDIDATES)

    strategies = []

    # Strategy 1: Standard Dense Vector Search (Baseline)
    t0 = time.perf_counter()
    v_chunks = query_similar_documents(question, top_k=top_k, collection_name=collection_name)
    t_retr_v = time.perf_counter()
    retr_ms_v = round((t_retr_v - t0) * 1000, 1)

    ans_v = generate_answer(question, v_chunks)
    t_gen_v = time.perf_counter()
    gen_ms_v = round((t_gen_v - t_retr_v) * 1000, 1)
    tot_ms_v = round((t_gen_v - t0) * 1000, 1)

    strategies.append({
        "id": "vector",
        "name": "1. Standard Dense Vector",
        "badge": "Vector Search (MiniLM)",
        "answer": ans_v,
        "retrieved_chunks": format_chunks_data(v_chunks),
        "latency": {"retrieval_ms": retr_ms_v, "rerank_ms": 0.0, "generation_ms": gen_ms_v, "total_ms": tot_ms_v}
    })

    # Strategy 2: Hybrid Search (Dense + BM25)
    t0 = time.perf_counter()
    h_chunks = hybrid_search(question, top_k=top_k, collection_name=collection_name)
    t_retr_h = time.perf_counter()
    retr_ms_h = round((t_retr_h - t0) * 1000, 1)

    ans_h = generate_answer(question, h_chunks)
    t_gen_h = time.perf_counter()
    gen_ms_h = round((t_gen_h - t_retr_h) * 1000, 1)
    tot_ms_h = round((t_gen_h - t0) * 1000, 1)

    strategies.append({
        "id": "hybrid",
        "name": "2. Hybrid Search (Dense + BM25)",
        "badge": "RRF Rank Fusion",
        "answer": ans_h,
        "retrieved_chunks": format_chunks_data(h_chunks),
        "latency": {"retrieval_ms": retr_ms_h, "rerank_ms": 0.0, "generation_ms": gen_ms_h, "total_ms": tot_ms_h}
    })

    # Strategy 3: Vector Search + Cross-Encoder Reranker
    t0 = time.perf_counter()
    v_candidates = query_similar_documents(question, top_k=candidates_n, collection_name=collection_name)
    t_retr_vr = time.perf_counter()
    retr_ms_vr = round((t_retr_vr - t0) * 1000, 1)

    vr_chunks = rerank_documents(question, v_candidates, top_k=top_k)
    t_rerank_vr = time.perf_counter()
    rerank_ms_vr = round((t_rerank_vr - t_retr_vr) * 1000, 1)

    ans_vr = generate_answer(question, vr_chunks)
    t_gen_vr = time.perf_counter()
    gen_ms_vr = round((t_gen_vr - t_rerank_vr) * 1000, 1)
    tot_ms_vr = round((t_gen_vr - t0) * 1000, 1)

    strategies.append({
        "id": "vector_rerank",
        "name": "3. Vector + Cross-Encoder",
        "badge": "Dense + MiniLM-L6 Rerank",
        "answer": ans_vr,
        "retrieved_chunks": format_chunks_data(vr_chunks),
        "latency": {"retrieval_ms": retr_ms_vr, "rerank_ms": rerank_ms_vr, "generation_ms": gen_ms_vr, "total_ms": tot_ms_vr}
    })

    # Strategy 4: Hybrid Search + Cross-Encoder Reranker
    t0 = time.perf_counter()
    h_candidates = hybrid_search(question, top_k=candidates_n, collection_name=collection_name)
    t_retr_hr = time.perf_counter()
    retr_ms_hr = round((t_retr_hr - t0) * 1000, 1)

    hr_chunks = rerank_documents(question, h_candidates, top_k=top_k)
    t_rerank_hr = time.perf_counter()
    rerank_ms_hr = round((t_rerank_hr - t_retr_hr) * 1000, 1)

    ans_hr = generate_answer(question, hr_chunks)
    t_gen_hr = time.perf_counter()
    gen_ms_hr = round((t_gen_hr - t_rerank_hr) * 1000, 1)
    tot_ms_hr = round((t_gen_hr - t0) * 1000, 1)

    strategies.append({
        "id": "hybrid_rerank",
        "name": "4. Hybrid + Cross-Encoder",
        "badge": "Dense + BM25 + Cross-Encoder",
        "answer": ans_hr,
        "retrieved_chunks": format_chunks_data(hr_chunks),
        "latency": {"retrieval_ms": retr_ms_hr, "rerank_ms": rerank_ms_hr, "generation_ms": gen_ms_hr, "total_ms": tot_ms_hr}
    })

    return strategies


