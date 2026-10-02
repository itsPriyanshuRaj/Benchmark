from pathlib import Path
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

import config
from rag.loader import load_pdf
from rag.splitter import split_documents
from rag.vectorStore import add_documents

upload_bp = Blueprint("upload", __name__)

@upload_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Handles PDF upload, parses pages, chunks content, and indexes in Chroma.
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
        # 1. Save uploaded PDF safely to disk
        save_path = config.UPLOAD_FOLDER / filename
        file.save(str(save_path))

        # 2. Load PDF pages into Document objects
        documents = load_pdf(save_path)

        # 3. Split documents into smaller text chunks
        chunks = split_documents(documents)
        if not chunks:
            return jsonify({
                "status": "error",
                "message": "No extractable text found in the uploaded PDF."
            }), 400

        # 4. Embed and store chunks in Chroma vector store
        chunks_indexed = add_documents(chunks)

        return jsonify({
            "status": "success",
            "filename": filename,
            "pages_loaded": len(documents),
            "chunks_indexed": chunks_indexed
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
