from flask import Flask, render_template, jsonify
import config

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["UPLOAD_FOLDER"] = str(config.UPLOAD_FOLDER)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # Ensure storage directories exist
    config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "service": "RAGBench AI",
            "upload_folder": str(config.UPLOAD_FOLDER),
            "chroma_dir": str(config.CHROMA_PERSIST_DIR)
        })

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
