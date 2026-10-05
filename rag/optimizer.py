import time
import json
from pathlib import Path
import config
from rag.vectorStore import query_similar_documents
from rag.hybrid import hybrid_search
from rag.reranker import rerank_documents
from rag.generator import generate_answer
from rag.evaluator import evaluate_rag_response
from rag.benchmarker import format_chunks_data
from rag.tracker import log_experiment_run, get_experiment_run_details

# Human-readable strategy definitions
STRATEGY_METADATA = {
    "vector": {
        "name": "Standard Dense Vector",
        "badge": "Vector Search (MiniLM)",
        "desc": "Baseline dense semantic retrieval using Bi-Encoder vector embeddings."
    },
    "hybrid": {
        "name": "Hybrid Search (Dense + BM25)",
        "badge": "RRF Rank Fusion",
        "desc": "Reciprocal Rank Fusion merging dense embeddings and sparse BM25 keyword matching."
    },
    "vector_rerank": {
        "name": "Vector + Cross-Encoder Reranker",
        "badge": "Dense + MiniLM-L6 Rerank",
        "desc": "Dense vector candidate retrieval re-scored by a Cross-Encoder for higher precision."
    },
    "hybrid_rerank": {
        "name": "Hybrid + Cross-Encoder Reranker",
        "badge": "Dense + BM25 + Cross-Encoder",
        "desc": "Two-stage retrieval: broad hybrid candidate recall followed by deep cross-attention precision reranking."
    }
}

def diagnose_configuration(evaluation_data, latency_data=None, strategy_id="vector", top_k=3):
    """
    Analyzes an evaluation payload and latency measurements to identify RAG pipeline weaknesses,
    bottlenecks, and performance opportunities.
    """
    bottlenecks = []
    strengths = []

    metrics = (evaluation_data.get("metrics") or {}) if isinstance(evaluation_data, dict) else {}
    
    def _get_val(key):
        item = metrics.get(key)
        if isinstance(item, dict):
            return item.get("score")
        elif isinstance(item, (int, float)):
            return float(item)
        return None

    faithfulness = _get_val("faithfulness")
    relevance = _get_val("answer_relevance")
    precision = _get_val("context_precision")
    recall = _get_val("context_recall")
    overall_score = evaluation_data.get("overall_score") if isinstance(evaluation_data, dict) else None

    # Latency parsing
    latency = latency_data or {}
    total_ms = float(latency.get("total_ms", 0.0) or 0.0)
    retrieval_ms = float(latency.get("retrieval_ms", 0.0) or 0.0)
    generation_ms = float(latency.get("generation_ms", 0.0) or 0.0)

    # Threshold checks
    faith_thresh = getattr(config, "FAITHFULNESS_THRESHOLD", 0.70)
    rel_thresh = getattr(config, "ANSWER_RELEVANCE_THRESHOLD", 0.70)
    prec_thresh = getattr(config, "CONTEXT_PRECISION_THRESHOLD", 0.60)
    lat_thresh = getattr(config, "OPTIMIZER_LATENCY_THRESHOLD_MS", 2500.0)

    # 1. Context Precision Diagnosis (Signal-to-Noise Ratio)
    if precision is not None:
        if precision < prec_thresh:
            bottlenecks.append({
                "id": "LOW_CONTEXT_PRECISION",
                "name": "Low Context Precision (Noisy Retrieval)",
                "severity": "HIGH" if precision < 0.45 else "MEDIUM",
                "metric": "context_precision",
                "score": precision,
                "description": f"Retrieved context contains excessive irrelevant text (score {int(precision*100)}%). Irrelevant chunks dilute answer quality."
            })
        elif precision >= 0.8:
            strengths.append(f"High context precision ({int(precision*100)}%) — retrieved text is directly relevant.")

    # 2. Context Recall Diagnosis (Information Coverage)
    if recall is not None:
        if recall < 0.65:
            bottlenecks.append({
                "id": "LOW_CONTEXT_RECALL",
                "name": "Low Context Recall (Missing Information)",
                "severity": "HIGH" if recall < 0.4 else "MEDIUM",
                "metric": "context_recall",
                "score": recall,
                "description": f"Retrieved context missed key factual details required to answer completely (score {int(recall*100)}%)."
            })
        elif recall >= 0.85:
            strengths.append(f"Strong context recall ({int(recall*100)}%) — key facts were captured.")

    # 3. Faithfulness / Hallucination Risk
    if faithfulness is not None:
        if faithfulness < faith_thresh:
            bottlenecks.append({
                "id": "HALLUCINATION_RISK",
                "name": "Faithfulness Risk (Ungrounded Claims)",
                "severity": "HIGH",
                "metric": "faithfulness",
                "score": faithfulness,
                "description": f"Generated answer makes assertions not strictly supported by retrieved context (score {int(faithfulness*100)}%)."
            })
        elif faithfulness >= 0.85:
            strengths.append(f"High factual faithfulness ({int(faithfulness*100)}%) — answer is well-grounded in evidence.")

    # 4. Answer Relevance
    if relevance is not None:
        if relevance < rel_thresh:
            bottlenecks.append({
                "id": "LOW_ANSWER_RELEVANCE",
                "name": "Low Answer Relevance (Query Drift)",
                "severity": "MEDIUM",
                "metric": "answer_relevance",
                "score": relevance,
                "description": f"The answer does not directly address the user question (score {int(relevance*100)}%)."
            })
        elif relevance >= 0.85:
            strengths.append(f"Direct answer relevance ({int(relevance*100)}%) — response stays tightly focused on the query.")

    # 5. Latency Bottleneck
    if total_ms > lat_thresh:
        bottlenecks.append({
            "id": "HIGH_LATENCY",
            "name": "High End-to-End Latency",
            "severity": "HIGH" if total_ms > 4000 else "MEDIUM",
            "metric": "latency",
            "score": total_ms,
            "description": f"Pipeline took {total_ms}ms (threshold: {lat_thresh}ms). Retrieval: {retrieval_ms}ms, Generation: {generation_ms}ms."
        })

    # Overall Health Score calculation (0 to 100)
    if overall_score is not None:
        health_score = int(round(overall_score * 100))
    else:
        # Fallback based on bottleneck count
        health_score = max(20, 100 - (len(bottlenecks) * 25))

    if len(bottlenecks) == 0:
        health_status = "OPTIMAL"
    elif any(b["severity"] == "HIGH" for b in bottlenecks):
        health_status = "CRITICAL_ISSUES"
    else:
        health_status = "NEEDS_OPTIMIZATION"

    return {
        "health_status": health_status,
        "health_score": health_score,
        "current_strategy": strategy_id,
        "current_top_k": top_k,
        "bottlenecks": bottlenecks,
        "strengths": strengths
    }

def recommend_next_experiment(diagnosis, current_strategy="vector", current_k=3):
    """
    Synthesizes the optimal next RAG configuration to test based on the diagnosed bottlenecks.
    """
    bottleneck_ids = {b["id"] for b in diagnosis.get("bottlenecks", [])}

    recommended_strategy = "hybrid_rerank"
    recommended_k = current_k
    hypothesis = ""
    expected_improvements = []
    reasoning = ""

    # Decision tree logic based on diagnosed weaknesses
    has_precision_issue = "LOW_CONTEXT_PRECISION" in bottleneck_ids
    has_recall_issue = "LOW_CONTEXT_RECALL" in bottleneck_ids
    has_hallucination = "HALLUCINATION_RISK" in bottleneck_ids
    has_latency_issue = "HIGH_LATENCY" in bottleneck_ids

    if current_strategy == "vector":
        if has_precision_issue and not has_recall_issue:
            recommended_strategy = "vector_rerank"
            recommended_k = max(2, min(current_k, 3))
            hypothesis = "Applying a Cross-Encoder Reranker to dense candidates will score query-chunk cross-attention directly, filtering out noise and boosting context precision."
            expected_improvements = ["+25% to +40% Context Precision", "Reduced generator confusion", "Cleaner context window"]
            reasoning = "Dense retrieval found relevant documents but included too much noisy filler. A Cross-Encoder reranker re-orders chunks by true relevance."

        elif has_recall_issue and not has_precision_issue:
            recommended_strategy = "hybrid"
            recommended_k = max(current_k, 5)
            hypothesis = "Enabling BM25 sparse lexical search alongside dense vectors and increasing Top-K will retrieve exact keyword matches and technical terminology missed by pure vector embeddings."
            expected_improvements = ["+30% to +50% Context Recall", "Broadened factual coverage", "Captures exact keywords and entities"]
            reasoning = "Vector embeddings often miss rare tokens, acronyms, or proper nouns. Hybrid BM25 fusion ensures sparse lexical terms are retrieved."

        elif has_precision_issue or has_recall_issue or has_hallucination:
            recommended_strategy = "hybrid_rerank"
            recommended_k = max(current_k, 3)
            hypothesis = "Employing a two-stage Hybrid + Cross-Encoder pipeline will maximize recall via BM25 + Dense fusion in Stage 1, while Cross-Encoder reranking in Stage 2 prunes irrelevant chunks to ensure high precision and eliminate hallucination."
            expected_improvements = ["+35% Overall Answer Quality", "+30% Context Precision", "+25% Context Recall", "Substantially reduced hallucination risk"]
            reasoning = "The baseline configuration suffers from compound retrieval deficiencies. Two-stage hybrid retrieval with cross-encoder reranking is the gold standard for RAG accuracy."

        else:
            # Current pipeline is okay, suggest reranking for peak accuracy
            recommended_strategy = "vector_rerank"
            recommended_k = current_k
            hypothesis = "Adding Cross-Encoder reranking will further polish chunk ranking and ensure the single most authoritative snippet is passed at rank #1."
            expected_improvements = ["Higher confidence ranking", "+10% Groundedness"]
            reasoning = "Incremental quality uplift via cross-attention reranking."

    elif current_strategy == "hybrid":
        if has_precision_issue or has_hallucination:
            recommended_strategy = "hybrid_rerank"
            recommended_k = max(2, min(current_k, 4))
            hypothesis = "Hybrid search successfully retrieved diverse candidate chunks, but needs Cross-Encoder reranking to eliminate irrelevant candidates before generator ingestion."
            expected_improvements = ["+30% Context Precision", "Eliminates spurious BM25 keyword hits", "High factual faithfulness"]
            reasoning = "BM25 can introduce spurious keyword overlaps. A Cross-Encoder reranker ensures only truly semantic matches reach the LLM."
        elif has_latency_issue:
            recommended_strategy = "vector"
            recommended_k = 3
            hypothesis = "Reverting to pure dense vector search with K=3 reduces retrieval latency while maintaining acceptable semantic coverage."
            expected_improvements = ["-40% Retrieval Latency", "Faster query cycle"]
            reasoning = "BM25 tokenization and fusion add latency overhead that may not be needed if vector embeddings suffice."
        else:
            recommended_strategy = "hybrid_rerank"
            recommended_k = current_k
            hypothesis = "Adding cross-attention reranking completes the two-stage hybrid pipeline."
            expected_improvements = ["+20% Precision"]
            reasoning = "RRF gives strong candidates; Cross-Encoder provides exact ordering."

    elif current_strategy == "vector_rerank":
        if has_recall_issue:
            recommended_strategy = "hybrid_rerank"
            recommended_k = max(current_k, 4)
            hypothesis = "Cross-Encoder reranking is working well, but the candidate generator is missing sparse keywords. Upgrading candidate retrieval to Hybrid (Dense + BM25) will supply better candidates for reranking."
            expected_improvements = ["+35% Context Recall", "Resolves missing lexical matches"]
            reasoning = "Rerankers can only rank what is retrieved in Stage 1. Hybrid search feeds a much richer candidate pool."
        elif has_latency_issue:
            recommended_strategy = "vector"
            recommended_k = 3
            hypothesis = "Disabling cross-encoder inference cuts latency significantly for time-critical scenarios."
            expected_improvements = ["-300ms to -800ms Latency"]
            reasoning = "Cross-Encoder inference requires dedicated forward passes over all candidate pairs."
        else:
            recommended_strategy = "hybrid_rerank"
            recommended_k = current_k
            hypothesis = "Injecting BM25 lexical candidates into the reranker pool maximizes factual recall."
            expected_improvements = ["+15% Recall"]
            reasoning = "Completes hybrid multi-stage retrieval architecture."

    else: # hybrid_rerank already
        if has_latency_issue:
            recommended_strategy = "vector_rerank"
            recommended_k = 3
            hypothesis = "Switching from Hybrid+Reranker to Vector+Reranker with Top-K=3 streamlines candidate retrieval and reduces token overhead."
            expected_improvements = ["-25% Total Latency", "Streamlined pipeline"]
            reasoning = "Fine-tuning candidate generation to balance millisecond latency and quality."
        elif has_recall_issue:
            recommended_k = min(current_k + 2, 7)
            recommended_strategy = "hybrid_rerank"
            hypothesis = f"Increasing Top-K from {current_k} to {recommended_k} provides broader context to satisfy complex multi-part queries."
            expected_improvements = ["+20% Context Recall", "Full query coverage"]
            reasoning = "The retrieval pipeline is already optimal; increasing K provides more facts."
        else:
            recommended_strategy = "hybrid_rerank"
            recommended_k = current_k
            hypothesis = "Pipeline configuration is already optimal. Maintaining current parameters."
            expected_improvements = ["Consistent top-tier performance"]
            reasoning = "Hybrid + Cross-Encoder represents the strongest achievable retrieval architecture."

    meta = STRATEGY_METADATA.get(recommended_strategy, STRATEGY_METADATA["vector"])

    return {
        "recommended_strategy": recommended_strategy,
        "recommended_strategy_name": meta["name"],
        "recommended_badge": meta["badge"],
        "recommended_top_k": recommended_k,
        "hypothesis": hypothesis,
        "expected_improvements": expected_improvements,
        "reasoning": reasoning
    }

def execute_strategy_query(strategy_id, question, top_k=3, collection_name="ragbench_docs"):
    """
    Executes a single RAG strategy and returns measured answer and latencies.
    """
    if top_k is None:
        top_k = getattr(config, "DEFAULT_TOP_K", 3)
    candidates_n = max(top_k * 2, getattr(config, "DEFAULT_RERANK_CANDIDATES", 10))

    t0 = time.perf_counter()
    retrieval_ms = 0.0
    rerank_ms = 0.0

    if strategy_id == "vector":
        raw_chunks = query_similar_documents(question, top_k=top_k, collection_name=collection_name)
        t_retr = time.perf_counter()
        retrieval_ms = round((t_retr - t0) * 1000, 1)

    elif strategy_id == "hybrid":
        raw_chunks = hybrid_search(question, top_k=top_k, collection_name=collection_name)
        t_retr = time.perf_counter()
        retrieval_ms = round((t_retr - t0) * 1000, 1)

    elif strategy_id == "vector_rerank":
        candidates = query_similar_documents(question, top_k=candidates_n, collection_name=collection_name)
        t_retr = time.perf_counter()
        retrieval_ms = round((t_retr - t0) * 1000, 1)

        raw_chunks = rerank_documents(question, candidates, top_k=top_k)
        t_rerank = time.perf_counter()
        rerank_ms = round((t_rerank - t_retr) * 1000, 1)

    elif strategy_id == "hybrid_rerank":
        candidates = hybrid_search(question, top_k=candidates_n, collection_name=collection_name)
        t_retr = time.perf_counter()
        retrieval_ms = round((t_retr - t0) * 1000, 1)

        raw_chunks = rerank_documents(question, candidates, top_k=top_k)
        t_rerank = time.perf_counter()
        rerank_ms = round((t_rerank - t_retr) * 1000, 1)

    else:
        # Fallback to vector
        raw_chunks = query_similar_documents(question, top_k=top_k, collection_name=collection_name)
        t_retr = time.perf_counter()
        retrieval_ms = round((t_retr - t0) * 1000, 1)

    # Generation
    t_gen_start = time.perf_counter()
    answer = generate_answer(question, raw_chunks)
    t_gen_end = time.perf_counter()
    generation_ms = round((t_gen_end - t_gen_start) * 1000, 1)
    total_ms = round((t_gen_end - t0) * 1000, 1)

    chunks_data = format_chunks_data(raw_chunks)
    meta = STRATEGY_METADATA.get(strategy_id, STRATEGY_METADATA["vector"])

    return {
        "id": strategy_id,
        "name": meta["name"],
        "badge": meta["badge"],
        "top_k": top_k,
        "answer": answer,
        "retrieved_chunks": chunks_data,
        "latency": {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms
        }
    }

def run_auto_optimization_cycle(
    question,
    baseline_strategy="vector",
    baseline_k=3,
    collection_name="ragbench_docs",
    ground_truth=None,
    eval_mode="heuristic",
    precomputed_baseline=None
):
    """
    Orchestrates the entire automated optimization workflow:
    1. Executes baseline configuration (or accepts precomputed baseline).
    2. Evaluates baseline with LLM-as-a-judge.
    3. Runs diagnosis to pinpoint bottlenecks.
    4. Synthesizes an optimization hypothesis & recommends next experiment.
    5. Automatically executes the recommended configuration.
    6. Evaluates the optimized response with LLM-as-a-judge.
    7. Computes before vs after delta metrics and verdict.
    8. Persists the optimization run in the Phase 5 experiment tracker.
    """
    # Step 1: Baseline Execution & Evaluation
    if precomputed_baseline:
        baseline_res = precomputed_baseline
        baseline_eval = baseline_res.get("evaluation")
        if not baseline_eval:
            contexts = [c.get("content", "") for c in baseline_res.get("retrieved_chunks", [])]
            baseline_eval = evaluate_rag_response(
                question=question,
                answer=baseline_res.get("answer", ""),
                contexts=contexts,
                ground_truth=ground_truth,
                mode=eval_mode
            )
            baseline_res["evaluation"] = baseline_eval
    else:
        baseline_res = execute_strategy_query(
            strategy_id=baseline_strategy,
            question=question,
            top_k=baseline_k,
            collection_name=collection_name
        )
        contexts = [c.get("content", "") for c in baseline_res.get("retrieved_chunks", [])]
        baseline_eval = evaluate_rag_response(
            question=question,
            answer=baseline_res.get("answer", ""),
            contexts=contexts,
            ground_truth=ground_truth,
            mode=eval_mode
        )
        baseline_res["evaluation"] = baseline_eval

    # Step 2: Diagnostic Analysis
    diagnosis = diagnose_configuration(
        evaluation_data=baseline_eval,
        latency_data=baseline_res.get("latency"),
        strategy_id=baseline_strategy,
        top_k=baseline_k
    )

    # Step 3: Optimization Recommendation
    recommendation = recommend_next_experiment(
        diagnosis=diagnosis,
        current_strategy=baseline_strategy,
        current_k=baseline_k
    )

    target_strat = recommendation["recommended_strategy"]
    target_k = recommendation["recommended_top_k"]

    # Step 4: Execute Optimized Experiment
    opt_res = execute_strategy_query(
        strategy_id=target_strat,
        question=question,
        top_k=target_k,
        collection_name=collection_name
    )

    # Step 5: Evaluate Optimized Output
    opt_contexts = [c.get("content", "") for c in opt_res.get("retrieved_chunks", [])]
    opt_eval = evaluate_rag_response(
        question=question,
        answer=opt_res.get("answer", ""),
        contexts=opt_contexts,
        ground_truth=ground_truth,
        mode=eval_mode
    )
    opt_res["evaluation"] = opt_eval

    # Step 6: Compute Delta Improvements
    def _metric_score(ev, metric_name):
        if not ev or not isinstance(ev, dict):
            return 0.0
        m = ev.get("metrics", {}).get(metric_name)
        if isinstance(m, dict):
            return float(m.get("score") or 0.0)
        elif isinstance(m, (int, float)):
            return float(m)
        return 0.0

    b_overall = baseline_eval.get("overall_score") or 0.0
    o_overall = opt_eval.get("overall_score") or 0.0
    delta_overall = round(o_overall - b_overall, 2)
    pct_overall_change = int(round(delta_overall * 100))

    b_faith = _metric_score(baseline_eval, "faithfulness")
    o_faith = _metric_score(opt_eval, "faithfulness")
    delta_faith = round(o_faith - b_faith, 2)

    b_prec = _metric_score(baseline_eval, "context_precision")
    o_prec = _metric_score(opt_eval, "context_precision")
    delta_prec = round(o_prec - b_prec, 2)

    b_recall = _metric_score(baseline_eval, "context_recall")
    o_recall = _metric_score(opt_eval, "context_recall")
    delta_recall = round(o_recall - b_recall, 2)

    b_rel = _metric_score(baseline_eval, "answer_relevance")
    o_rel = _metric_score(opt_eval, "answer_relevance")
    delta_rel = round(o_rel - b_rel, 2)

    b_lat = baseline_res.get("latency", {}).get("total_ms", 0.0)
    o_lat = opt_res.get("latency", {}).get("total_ms", 0.0)
    delta_lat = round(o_lat - b_lat, 1)

    # Verdict synthesis
    if delta_overall >= 0.10 or (delta_overall >= 0 and delta_lat < -200):
        verdict = "OPTIMIZATION_SUCCESS"
        verdict_badge = f"+{pct_overall_change}% Quality Uplift" if pct_overall_change > 0 else f"{delta_lat}ms Latency Reduction"
    elif delta_overall > 0:
        verdict = "PARTIAL_IMPROVEMENT"
        verdict_badge = f"+{pct_overall_change}% Improvement"
    elif delta_overall == 0:
        verdict = "NEUTRAL"
        verdict_badge = "Neutral Shift"
    else:
        verdict = "REGRESSION"
        verdict_badge = f"{pct_overall_change}% Variation"

    deltas = {
        "overall_score_delta": delta_overall,
        "overall_score_pct": pct_overall_change,
        "faithfulness_delta": delta_faith,
        "context_precision_delta": delta_prec,
        "context_recall_delta": delta_recall,
        "answer_relevance_delta": delta_rel,
        "latency_delta_ms": delta_lat,
        "verdict": verdict,
        "verdict_badge": verdict_badge
    }

    # Step 7: Persist in SQLite Tracker
    comparisons = [
        {
            "id": f"baseline_{baseline_strategy}",
            "name": f"Baseline: {baseline_res['name']}",
            "badge": f"Baseline (K={baseline_k})",
            "answer": baseline_res["answer"],
            "retrieved_chunks": baseline_res["retrieved_chunks"],
            "latency": baseline_res["latency"],
            "evaluation": baseline_eval
        },
        {
            "id": f"optimized_{target_strat}",
            "name": f"Auto-Optimized: {opt_res['name']}",
            "badge": f"Optimized (K={target_k})",
            "answer": opt_res["answer"],
            "retrieved_chunks": opt_res["retrieved_chunks"],
            "latency": opt_res["latency"],
            "evaluation": opt_eval
        }
    ]

    run_id = log_experiment_run(
        experiment_type="optimization",
        question=question,
        comparisons=comparisons,
        collection_name=collection_name,
        top_k=target_k,
        ground_truth=ground_truth
    )

    return {
        "status": "success",
        "run_id": run_id,
        "question": question,
        "collection_name": collection_name,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "baseline": baseline_res,
        "optimized": opt_res,
        "deltas": deltas
    }

def diagnose_past_run(run_id):
    """
    Extracts an existing experiment run from Phase 5 Tracker and performs diagnostic analysis.
    """
    details = get_experiment_run_details(run_id)
    if not details:
        return None

    run = details.get("run", {})
    comparisons = details.get("comparisons", [])
    if not comparisons:
        return None

    # Pick the baseline (or first/worst strategy) to diagnose
    primary_comp = comparisons[0]
    ev = primary_comp.get("evaluation") or {}
    lat = primary_comp.get("latency") or {}

    strategy_id = primary_comp.get("id") or "vector"
    top_k = run.get("top_k") or 3

    diagnosis = diagnose_configuration(
        evaluation_data=ev,
        latency_data=lat,
        strategy_id=strategy_id,
        top_k=top_k
    )

    recommendation = recommend_next_experiment(
        diagnosis=diagnosis,
        current_strategy=strategy_id,
        current_k=top_k
    )

    return {
        "run_id": run_id,
        "question": run.get("question"),
        "collection_name": run.get("collection_name"),
        "top_k": top_k,
        "baseline_strategy": primary_comp.get("name"),
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "comparison": primary_comp
    }

