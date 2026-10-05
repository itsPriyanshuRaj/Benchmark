import csv
import io
import json
from flask import Blueprint, request, jsonify, Response
from rag.tracker import (
    get_experiment_runs,
    get_experiment_run_details,
    delete_experiment_run,
    clear_all_runs,
    get_dashboard_analytics
)

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")

@dashboard_bp.route("/runs", methods=["GET"])
def list_runs():
    """
    Returns list of experiment runs with optional filtering by type and question search.
    """
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    exp_type = request.args.get("type", None)
    search = request.args.get("search", None)

    runs = get_experiment_runs(limit=limit, offset=offset, experiment_type=exp_type, search=search)
    return jsonify({
        "status": "success",
        "count": len(runs),
        "runs": runs
    }), 200

@dashboard_bp.route("/runs/<run_id>", methods=["GET"])
def get_run(run_id):
    """
    Returns full details and comparisons array for a specific experiment run.
    """
    details = get_experiment_run_details(run_id)
    if not details:
        return jsonify({"status": "error", "message": f"Run {run_id} not found."}), 404

    return jsonify({
        "status": "success",
        "run": details["run"],
        "comparisons": details["comparisons"]
    }), 200

@dashboard_bp.route("/runs/<run_id>", methods=["DELETE"])
def delete_run(run_id):
    """
    Deletes a specific experiment run.
    """
    success = delete_experiment_run(run_id)
    if not success:
        return jsonify({"status": "error", "message": f"Failed to delete run {run_id}."}), 400

    return jsonify({
        "status": "success",
        "message": f"Run {run_id} deleted successfully."
    }), 200

@dashboard_bp.route("/clear", methods=["POST", "DELETE"])
def clear_history():
    """
    Clears all saved experiment runs and comparison items.
    """
    success = clear_all_runs()
    if not success:
        return jsonify({"status": "error", "message": "Failed to clear experiment history."}), 500

    return jsonify({
        "status": "success",
        "message": "All experiment runs cleared successfully."
    }), 200

@dashboard_bp.route("/stats", methods=["GET"])
def get_stats():
    """
    Returns aggregated analytics across all past experiment runs.
    """
    analytics = get_dashboard_analytics()
    return jsonify({
        "status": "success",
        "analytics": analytics
    }), 200

@dashboard_bp.route("/export", methods=["GET"])
def export_data():
    """
    Exports all experiment runs as a JSON or CSV download.
    """
    export_format = request.args.get("format", "json").lower()
    runs = get_experiment_runs(limit=1000)

    if export_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Run ID", "Timestamp", "Type", "Question", "Collection",
            "Top-K", "Strategies Tested", "Fastest Strategy",
            "Fastest Latency (ms)", "Best Verdict", "Avg Quality Score"
        ])
        for r in runs:
            writer.writerow([
                r.get("run_id"),
                r.get("timestamp"),
                r.get("experiment_type"),
                r.get("question"),
                r.get("collection_name"),
                r.get("top_k"),
                r.get("strategy_count"),
                r.get("fastest_strategy"),
                r.get("fastest_latency_ms"),
                r.get("best_verdict"),
                r.get("avg_quality_score")
            ])
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=ragbench_experiments.csv"}
        )

    # Default: JSON export
    full_export = []
    for r in runs:
        details = get_experiment_run_details(r["run_id"])
        if details:
            full_export.append(details)

    return jsonify({
        "status": "success",
        "exported_count": len(full_export),
        "experiments": full_export
    }), 200

