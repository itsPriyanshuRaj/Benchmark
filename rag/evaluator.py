import sys
from pathlib import Path

# Ensure project root directory is in sys.path when running or inspecting directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import re
import time
from langchain_ollama import ChatOllama
import config

def _extract_context_text(context_chunks):
    """
    Extracts text content from context chunks whether they are LangChain Document objects,
    dictionaries with 'content', or raw strings.
    """
    if not context_chunks:
        return "No retrieved context available."

    lines = []
    for i, chunk in enumerate(context_chunks, 1):
        if hasattr(chunk, "page_content"):
            content = chunk.page_content
        elif isinstance(chunk, dict) and "content" in chunk:
            content = chunk["content"]
        else:
            content = str(chunk)
        lines.append(f"[Chunk {i}]: {content.strip()}")

    return "\n\n".join(lines)

def _clean_json_response(raw_text):
    """
    Strips markdown code fences and extracts a valid JSON object string.
    """
    text = str(raw_text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def _parse_metric(metric_data, default_score=0.7):
    """
    Safely extracts score and reason whether the LLM returns:
    - A dict: {"score": 0.9, "reason": "..."}
    - A number: 0.9
    - A string or None
    Prevents AttributeError: 'float' object has no attribute 'get'.
    """
    if isinstance(metric_data, dict):
        raw_score = metric_data.get("score", default_score)
        reason = metric_data.get("reason", "Reason not specified.")
    elif isinstance(metric_data, (int, float)):
        raw_score = metric_data
        reason = "Score assigned by LLM judge."
    elif isinstance(metric_data, str):
        try:
            raw_score = float(metric_data)
        except ValueError:
            raw_score = default_score
        reason = "Score assigned by LLM judge."
    else:
        raw_score = default_score
        reason = "No explanation provided."

    try:
        score = float(raw_score)
    except (ValueError, TypeError):
        score = default_score

    # Clamp between 0.0 and 1.0
    score = max(0.0, min(1.0, score))
    return score, str(reason)

def calculate_heuristic_overlap(reference_text, candidate_text):
    """
    Computes lexical keyword recall between a reference text and candidate text.
    Returns a score between 0.0 and 1.0.
    """
    if not reference_text or not candidate_text:
        return 0.0

    ref_words = set(re.findall(r"\w+", reference_text.lower()))
    cand_words = set(re.findall(r"\w+", candidate_text.lower()))

    # Ignore common short stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "of", "to", "for", "with", "it", "this", "that", "by", "as"}
    meaningful_ref = ref_words - stopwords

    if not meaningful_ref:
        return 1.0

    intersection = meaningful_ref.intersection(cand_words)
    return round(len(intersection) / len(meaningful_ref), 3)

def evaluate_rag_response(question, context_chunks=None, answer="", ground_truth=None, model_name=None, mode="heuristic", contexts=None):
    """
    Evaluates a RAG response across 4 core metrics using local LLM-as-a-Judge:
    1. Faithfulness (Groundedness): Answer claims are supported by retrieved context without hallucination.
    2. Answer Relevance: Answer directly addresses the user question.
    3. Context Precision: Retrieved chunks contain information relevant to the question.
    4. Context Recall: Retrieved context contains the facts necessary to answer completely (or matches ground truth).
    """
    if context_chunks is None:
        context_chunks = contexts or []

    if model_name is None:
        model_name = getattr(config, "DEFAULT_EVALUATOR_MODEL", getattr(config, "DEFAULT_OLLAMA_MODEL", "llama3.2:3b"))

    context_str = _extract_context_text(context_chunks)
    ground_truth_clause = ""
    if ground_truth and ground_truth.strip():
        ground_truth_clause = f"\nGround Truth (Reference Answer):\n{ground_truth.strip()}\n"

    prompt = f"""You are an expert impartial evaluation judge for Retrieval-Augmented Generation (RAG) systems.
Analyze the provided Question, Retrieved Context, and Generated Answer below.
{ground_truth_clause}
Question:
{question}

Retrieved Context:
{context_str}

Generated Answer:
{answer}

Evaluate the response across these 4 RAG metrics on a scale of 0.0 to 1.0:
1. "faithfulness": Does the answer rely strictly on facts in the retrieved context without hallucinating external or false claims? (1.0 = completely faithful, 0.0 = completely hallucinated)
2. "answer_relevance": Does the answer directly address the user's specific question completely and concisely? (1.0 = fully relevant, 0.0 = completely irrelevant/off-topic)
3. "context_precision": Are the retrieved chunks relevant and focused on the question, rather than noisy or irrelevant text? (1.0 = high signal, 0.0 = all irrelevant noise)
4. "context_recall": Does the retrieved context contain all the necessary details/facts needed to answer the question? (1.0 = comprehensive context, 0.0 = missing all key facts)

Return ONLY a valid JSON object matching this exact schema:
{{
  "faithfulness": {{"score": 0.9, "reason": "brief explanation"}},
  "answer_relevance": {{"score": 0.9, "reason": "brief explanation"}},
  "context_precision": {{"score": 0.8, "reason": "brief explanation"}},
  "context_recall": {{"score": 0.8, "reason": "brief explanation"}},
  "verdict": "EXCELLENT",
  "critique": "one sentence overall evaluation summary"
}}
Permitted verdicts: "EXCELLENT", "GOOD", "POTENTIAL_HALLUCINATION", "RETRIEVAL_FAILURE", "LOW_RELEVANCE"."""

    t0 = time.perf_counter()
    try:
        llm = ChatOllama(model=model_name, temperature=0.0, format="json")
        response = llm.invoke(prompt)
        t_eval = round((time.perf_counter() - t0) * 1000, 1)

        cleaned = _clean_json_response(response.content)
        parsed = json.loads(cleaned)

        f_score, f_reason = _parse_metric(parsed.get("faithfulness"), default_score=0.7)
        ar_score, ar_reason = _parse_metric(parsed.get("answer_relevance"), default_score=0.7)
        cp_score, cp_reason = _parse_metric(parsed.get("context_precision"), default_score=0.7)
        cr_score, cr_reason = _parse_metric(parsed.get("context_recall"), default_score=0.7)

        # Overall weighted composite score
        overall = round((f_score * 0.35) + (ar_score * 0.35) + (cp_score * 0.15) + (cr_score * 0.15), 3)

        # Determine diagnostic verdict
        verdict = str(parsed.get("verdict", "")).upper()
        faithfulness_thresh = getattr(config, "FAITHFULNESS_THRESHOLD", 0.7)
        relevance_thresh = getattr(config, "ANSWER_RELEVANCE_THRESHOLD", 0.7)
        precision_thresh = getattr(config, "CONTEXT_PRECISION_THRESHOLD", 0.6)

        if f_score < faithfulness_thresh and ar_score >= relevance_thresh:
            verdict = "POTENTIAL_HALLUCINATION"
        elif cp_score < precision_thresh or cr_score < 0.5:
            verdict = "RETRIEVAL_FAILURE"
        elif ar_score < 0.5:
            verdict = "LOW_RELEVANCE"
        elif overall >= 0.85:
            verdict = "EXCELLENT"
        elif overall >= 0.70:
            verdict = "GOOD"
        elif verdict not in ["EXCELLENT", "GOOD", "POTENTIAL_HALLUCINATION", "RETRIEVAL_FAILURE", "LOW_RELEVANCE"]:
            verdict = "MODERATE"

        lexical_recall = None
        if ground_truth and ground_truth.strip():
            lexical_recall = calculate_heuristic_overlap(ground_truth, answer)

        return {
            "status": "success",
            "metrics": {
                "faithfulness": {
                    "score": round(f_score, 2),
                    "percentage": int(round(f_score * 100)),
                    "reason": f_reason
                },
                "answer_relevance": {
                    "score": round(ar_score, 2),
                    "percentage": int(round(ar_score * 100)),
                    "reason": ar_reason
                },
                "context_precision": {
                    "score": round(cp_score, 2),
                    "percentage": int(round(cp_score * 100)),
                    "reason": cp_reason
                },
                "context_recall": {
                    "score": round(cr_score, 2),
                    "percentage": int(round(cr_score * 100)),
                    "reason": cr_reason
                }
            },
            "overall_score": overall,
            "overall_percentage": int(round(overall * 100)),
            "verdict": verdict,
            "critique": parsed.get("critique", "Evaluation completed successfully."),
            "lexical_ground_truth_recall": lexical_recall,
            "evaluation_time_ms": t_eval
        }
    except Exception as e:
        t_eval = round((time.perf_counter() - t0) * 1000, 1)
        # Fallback evaluation on parser or LLM failure
        return {
            "status": "error",
            "message": str(e),
            "metrics": {
                "faithfulness": {"score": 0.5, "percentage": 50, "reason": f"Evaluation error: {str(e)}"},
                "answer_relevance": {"score": 0.5, "percentage": 50, "reason": f"Evaluation error: {str(e)}"},
                "context_precision": {"score": 0.5, "percentage": 50, "reason": f"Evaluation error: {str(e)}"},
                "context_recall": {"score": 0.5, "percentage": 50, "reason": f"Evaluation error: {str(e)}"}
            },
            "overall_score": 0.5,
            "overall_percentage": 50,
            "verdict": "EVALUATION_ERROR",
            "critique": f"Failed to run automated LLM judge: {str(e)}",
            "evaluation_time_ms": t_eval
        }

if __name__ == "__main__":
    print("Testing evaluator directly...")
    sample_q = "What is self-attention?"
    sample_ctx = ["The Transformer relies entirely on an attention mechanism called self-attention to compute representations."]
    sample_ans = "Self-attention is an attention mechanism that allows the model to compute representations of input sequences."
    sample_gt = "Self-attention computes representations without using recurrence or convolutions."

    eval_out = evaluate_rag_response(sample_q, sample_ctx, sample_ans, ground_truth=sample_gt)
    print("Direct Execution Result:")
    print(json.dumps(eval_out, indent=2))
