from flask import Blueprint, request, jsonify
import config
from rag.optimizer import (
    STRATEGY_METADATA,
    diagnose_configuration,
    recommend_next_experiment,
    run_auto_optimization_cycle,
    diagnose_past_run,
    execute_strategy_query
)
from rag.evaluator import evaluate_rag_response

optimizer_bp = Blueprint("optimizer", __name__, url_prefix="/optimizer")

@optimizer_bp.route("/strategies", methods=["GET"])
def get_strategies():
    """
    Returns available strategy metadata and optimization search space.
    """
    return jsonify({
        "status": "success",
        "strategies": STRATEGY_METADATA,
        "default_baseline": getattr(config, "OPTIMIZER_DEFAULT_BASELINE_STRATEGY", "vector"),
        "default_k": getattr(config, "OPTIMIZER_DEFAULT_BASELINE_K", 3)
    }), 200

@optimizer_bp.route("/diagnose", methods=["POST"])
def diagnose_run():
    """
    Diagnoses an existing experiment run or a live question query.
    Accepts:
    - { "run_id": "..." } to diagnose an existing past run from tracker
    - OR { "question": "...", "collection_name": "...", "strategy": "vector", "top_k": 3 }
    """
    data = request.get_json() or {}
    run_id = data.get("run_id")

    if run_id:
        result = diagnose_past_run(run_id)
        if not result:
            return jsonify({"status": "error", "message": f"Run '{run_id}' not found or has no comparisons."}), 404
        return jsonify({
            "status": "success",
            "mode": "past_run",
            "data": result
        }), 200

    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"status": "error", "message": "Question is required for diagnosis."}), 400

    collection_name = data.get("collection_name") or "ragbench_docs"
    strategy = data.get("strategy") or "vector"
    top_k = int(data.get("top_k", 3))
    ground_truth = (data.get("ground_truth") or "").strip() or None
    eval_mode = data.get("eval_mode") or "heuristic"

    try:
        # Run baseline
        baseline_res = execute_strategy_query(
            strategy_id=strategy,
            question=question,
            top_k=top_k,
            collection_name=collection_name
        )
        contexts = [c.get("content", "") for c in baseline_res.get("retrieved_chunks", [])]
        evaluation = evaluate_rag_response(
            question=question,
            answer=baseline_res.get("answer", ""),
            contexts=contexts,
            ground_truth=ground_truth,
            mode=eval_mode
        )
        baseline_res["evaluation"] = evaluation

        diagnosis = diagnose_configuration(
            evaluation_data=evaluation,
            latency_data=baseline_res.get("latency"),
            strategy_id=strategy,
            top_k=top_k
        )

        recommendation = recommend_next_experiment(
            diagnosis=diagnosis,
            current_strategy=strategy,
            current_k=top_k
        )

        return jsonify({
            "status": "success",
            "mode": "live",
            "data": {
                "question": question,
                "collection_name": collection_name,
                "baseline": baseline_res,
                "diagnosis": diagnosis,
                "recommendation": recommendation
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"Diagnosis failed: {str(e)}"}), 500

@optimizer_bp.route("/run", methods=["POST"])
def run_optimization():
    """
    Executes an end-to-end auto-optimization cycle:
    Baseline -> Diagnosis -> Recommendation -> Optimized Execution -> Evaluation -> Before vs After Comparison.
    """
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"status": "error", "message": "Question is required for auto-optimization."}), 400

    collection_name = data.get("collection_name") or "ragbench_docs"
    baseline_strategy = data.get("baseline_strategy") or "vector"
    baseline_k = int(data.get("baseline_k", 3))
    ground_truth = (data.get("ground_truth") or "").strip() or None
    eval_mode = data.get("eval_mode") or "heuristic"

    try:
        result = run_auto_optimization_cycle(
            question=question,
            baseline_strategy=baseline_strategy,
            baseline_k=baseline_k,
            collection_name=collection_name,
            ground_truth=ground_truth,
            eval_mode=eval_mode
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"Optimization execution failed: {str(e)}"}), 500

