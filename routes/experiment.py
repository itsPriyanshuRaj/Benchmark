from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

import config
from rag.loader import load_pdf
from rag.benchmarker import (
    index_experiments,
    run_chunking_comparison,
    run_topk_comparison,
    index_embedding_experiments,
    run_embedding_comparison,
    run_retrieval_quality_comparison
)

experiment_bp = Blueprint("experiment", __name__, url_prefix="/experiment")

@experiment_bp.route("/configs", methods=["GET"])
def get_experiment_configs():
    """
    Returns available experiment configurations for chunking, top-k, and embeddings.
    """
    return jsonify({
        "status": "success",
        "chunking_presets": config.DEFAULT_CHUNKING_EXPERIMENTS,
        "top_k_presets": config.DEFAULT_TOP_K_PRESETS,
        "embedding_models": config.AVAILABLE_EMBEDDING_MODELS
    }), 200

@experiment_bp.route("/collections", methods=["GET"])
def get_experiment_collections():
    """
    Returns all active collections in Chroma along with their chunk counts.
    """
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    colls = []
    for c in client.list_collections():
        colls.append({
            "name": c.name,
            "count": c.count()
        })
    colls.sort(key=lambda x: x["count"], reverse=True)
    return jsonify({
        "status": "success",
        "collections": colls
    }), 200

# ================= 1. CHUNKING BENCHMARK =================

@experiment_bp.route("/upload", methods=["POST"])
def upload_for_experiments():
    """
    Uploads a PDF and indexes it across all chunking presets (or custom configs).
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"status": "error", "message": "Only PDF files are supported"}), 400

    try:
        save_path = config.UPLOAD_FOLDER / filename
        file.save(str(save_path))

        documents = load_pdf(save_path)
        if not documents:
            return jsonify({"status": "error", "message": "Could not read pages from PDF."}), 400

        experiment_results = index_experiments(documents)

        return jsonify({
            "status": "success",
            "filename": filename,
            "pages_loaded": len(documents),
            "experiments": experiment_results
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@experiment_bp.route("/query", methods=["POST"])
@experiment_bp.route("/query/chunking", methods=["POST"])
def query_chunking_experiments():
    """
    Runs query comparison across chunking presets with latency breakdown.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"status": "error", "message": "The 'question' field is required."}), 400

    top_k = data.get("top_k", config.DEFAULT_TOP_K)

    try:
        comparisons = run_chunking_comparison(question, top_k=top_k)
        return jsonify({
            "status": "success",
            "question": question,
            "comparisons": comparisons
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ================= 2. TOP-K BENCHMARK =================

@experiment_bp.route("/query/topk", methods=["POST"])
def query_topk_experiments():
    """
    Runs query comparison across different Top-K retrieval depths (K=1, 3, 5).
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"status": "error", "message": "The 'question' field is required."}), 400

    k_values = data.get("k_values", config.DEFAULT_TOP_K_PRESETS)
    collection_name = data.get("collection_name", "ragbench_docs")

    try:
        comparisons = run_topk_comparison(question, k_values=k_values, collection_name=collection_name)
        return jsonify({
            "status": "success",
            "question": question,
            "comparisons": comparisons
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ================= 3. MULTI-EMBEDDING BENCHMARK =================

@experiment_bp.route("/upload/embedding", methods=["POST"])
def upload_embedding_experiments():
    """
    Uploads a PDF and indexes it across multiple embedding models.
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"status": "error", "message": "Only PDF files are supported"}), 400

    try:
        save_path = config.UPLOAD_FOLDER / filename
        file.save(str(save_path))

        documents = load_pdf(save_path)
        if not documents:
            return jsonify({"status": "error", "message": "Could not read pages from PDF."}), 400

        results = index_embedding_experiments(documents)
        return jsonify({
            "status": "success",
            "filename": filename,
            "pages_loaded": len(documents),
            "experiments": results
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@experiment_bp.route("/query/embedding", methods=["POST"])
def query_embedding_experiments():
    """
    Runs query comparison across multiple embedding models with latency breakdown.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"status": "error", "message": "The 'question' field is required."}), 400

    top_k = data.get("top_k", config.DEFAULT_TOP_K)

    try:
        comparisons = run_embedding_comparison(question, top_k=top_k)
        return jsonify({
            "status": "success",
            "question": question,
            "comparisons": comparisons
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ================= 4. CUSTOM EXPERIMENT BUILDER =================

@experiment_bp.route("/custom", methods=["POST"])
def run_custom_benchmark():
    """
    Indexes a previously uploaded document or uploaded file with user-specified custom configs.
    """
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    configs = data.get("configs", [])
    question = data.get("question", "").strip()
    top_k = data.get("top_k", config.DEFAULT_TOP_K)

    if not filename:
        return jsonify({"status": "error", "message": "Filename is required"}), 400

    save_path = config.UPLOAD_FOLDER / secure_filename(filename)
    if not save_path.exists():
        return jsonify({"status": "error", "message": f"File {filename} not found in uploads."}), 404

    try:
        documents = load_pdf(save_path)
        index_experiments(documents, configs=configs)
        comparisons = run_chunking_comparison(question, top_k=top_k, configs=configs)

        return jsonify({
            "status": "success",
            "question": question,
            "comparisons": comparisons
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ================= 5. RETRIEVAL QUALITY (HYBRID & RERANKING) =================

@experiment_bp.route("/query/quality", methods=["POST"])
def query_retrieval_quality():
    """
    Compares 4 Retrieval Strategies side-by-side:
    1. Standard Vector Search (Baseline)
    2. Hybrid Search (Dense + BM25)
    3. Vector Search + Cross-Encoder Reranker
    4. Hybrid Search + Cross-Encoder Reranker
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"status": "error", "message": "The 'question' field is required."}), 400

    top_k = data.get("top_k", config.DEFAULT_TOP_K)
    collection_name = data.get("collection_name", "exp_chunk_medium")

    try:
        comparisons = run_retrieval_quality_comparison(question, top_k=top_k, collection_name=collection_name)
        return jsonify({
            "status": "success",
            "question": question,
            "comparisons": comparisons
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


