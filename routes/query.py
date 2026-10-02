from flask import Blueprint, request, jsonify

import config
from rag.vectorStore import query_similar_documents
from rag.generator import generate_answer

query_bp = Blueprint("query", __name__)

@query_bp.route("/query", methods=["POST"])
def query_rag():
    """
    Receives a question, retrieves the top-K chunks from Chroma,
    and generates an answer via the local LLM.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"status": "error", "message": "The 'question' field is required."}), 400

    top_k = data.get("top_k", config.DEFAULT_TOP_K)

    try:
        # 1. Retrieve top-K relevant chunks from Chroma
        retrieved_chunks = query_similar_documents(question, top_k=top_k)

        # 2. Generate answer using local LLM based on retrieved context
        answer = generate_answer(question, retrieved_chunks)

        # 3. Format retrieved chunks for response display
        chunks_info = []
        for chunk in retrieved_chunks:
            chunks_info.append({
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "Unknown"),
                "page": chunk.metadata.get("page", 0) + 1
            })

        return jsonify({
            "status": "success",
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks_info
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

