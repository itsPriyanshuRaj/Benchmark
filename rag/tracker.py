import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import sqlite3
import json
import uuid
from datetime import datetime
import config

def get_db_connection():
    """
    Returns a connection to the SQLite experiment database with row_factory set to dict-like rows.
    """
    db_path = getattr(config, "DATABASE_PATH", project_root / "experiments.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """
    Initializes the SQLite database schema for storing experiment runs and comparison items.
    """
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                experiment_type TEXT NOT NULL,
                question TEXT NOT NULL,
                collection_name TEXT,
                top_k INTEGER,
                ground_truth TEXT,
                strategy_count INTEGER DEFAULT 0,
                fastest_strategy TEXT,
                fastest_latency_ms REAL,
                best_verdict TEXT,
                avg_quality_score REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiment_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                strategy_id TEXT,
                strategy_name TEXT NOT NULL,
                badge TEXT,
                answer TEXT,
                retrieval_ms REAL DEFAULT 0.0,
                rerank_ms REAL DEFAULT 0.0,
                generation_ms REAL DEFAULT 0.0,
                total_ms REAL DEFAULT 0.0,
                chunks_count INTEGER DEFAULT 0,
                chunks_data_json TEXT,
                faithfulness REAL,
                answer_relevance REAL,
                context_precision REAL,
                context_recall REAL,
                overall_score REAL,
                verdict TEXT,
                critique TEXT,
                evaluation_json TEXT,
                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE
            )
        """)

        # Indices for fast filtering and ordering
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON experiment_runs(timestamp DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_type ON experiment_runs(experiment_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_items_run_id ON experiment_items(run_id)")
    conn.close()

# Auto-initialize DB on import
init_db()

def _extract_metric_score(val):
    """
    Safely extracts a numeric score whether the metric is a dict ({"score": 0.9}) or a direct float/int.
    """
    if isinstance(val, dict):
        score = val.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return None
    elif isinstance(val, (int, float)):
        return float(val)
    return None

def log_experiment_run(experiment_type, question, comparisons, collection_name=None, top_k=None, ground_truth=None):
    """
    Persists a full experiment run and each of its comparative strategy outputs into SQLite.
    """
    if not comparisons:
        return None

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    now_iso = datetime.now().isoformat(timespec="seconds")

    # Compute summary analytics across comparisons
    fastest_strat = None
    min_latency = float("inf")
    quality_scores = []
    verdicts = []

    for comp in comparisons:
        lat_obj = comp.get("latency") or {}
        lat = lat_obj.get("total_ms", 0.0) if isinstance(lat_obj, dict) else 0.0
        lat = float(lat) if isinstance(lat, (int, float)) else 0.0

        if 0 < lat < min_latency:
            min_latency = lat
            fastest_strat = comp.get("name")

        ev = comp.get("evaluation")
        if ev and isinstance(ev, dict) and ev.get("overall_score") is not None:
            try:
                quality_scores.append(float(ev["overall_score"]))
            except (ValueError, TypeError):
                pass
            if ev.get("verdict"):
                verdicts.append(str(ev["verdict"]))

    fastest_latency_ms = min_latency if min_latency != float("inf") else 0.0
    avg_quality = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else None

    # Priority ranking for best verdict: EXCELLENT > GOOD > MODERATE
    best_verdict = None
    if "EXCELLENT" in verdicts:
        best_verdict = "EXCELLENT"
    elif "GOOD" in verdicts:
        best_verdict = "GOOD"
    elif verdicts:
        best_verdict = verdicts[0]

    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                INSERT INTO experiment_runs (
                    run_id, timestamp, experiment_type, question, collection_name,
                    top_k, ground_truth, strategy_count, fastest_strategy,
                    fastest_latency_ms, best_verdict, avg_quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, now_iso, experiment_type, question, collection_name,
                top_k, ground_truth, len(comparisons), fastest_strat,
                fastest_latency_ms, best_verdict, avg_quality
            ))

            for comp in comparisons:
                lat = comp.get("latency") or {}
                if not isinstance(lat, dict):
                    lat = {}
                chunks = comp.get("retrieved_chunks") or comp.get("chunks") or []
                if not isinstance(chunks, list):
                    chunks = []
                ev = comp.get("evaluation")

                faithfulness = None
                answer_relevance = None
                context_precision = None
                context_recall = None
                overall_score = None
                verdict = None
                critique = None
                eval_json_str = None

                if ev and isinstance(ev, dict):
                    try:
                        eval_json_str = json.dumps(ev)
                    except Exception:
                        eval_json_str = None
                    critique = ev.get("critique")
                    verdict = ev.get("verdict")
                    overall_score = ev.get("overall_score")
                    metrics = ev.get("metrics") or {}
                    if isinstance(metrics, dict):
                        faithfulness = _extract_metric_score(metrics.get("faithfulness"))
                        answer_relevance = _extract_metric_score(metrics.get("answer_relevance"))
                        context_precision = _extract_metric_score(metrics.get("context_precision"))
                        context_recall = _extract_metric_score(metrics.get("context_recall"))

                badge = comp.get("badge") or (
                    f"Size: {comp.get('chunk_size')} / Overlap: {comp.get('chunk_overlap')}"
                    if comp.get("chunk_size") is not None
                    else (comp.get("model_name") or (f"K = {comp.get('k')}" if comp.get("k") is not None else "Strategy"))
                )

                try:
                    chunks_json_str = json.dumps(chunks)
                except Exception:
                    chunks_json_str = "[]"

                conn.execute("""
                    INSERT INTO experiment_items (
                        run_id, strategy_id, strategy_name, badge, answer,
                        retrieval_ms, rerank_ms, generation_ms, total_ms,
                        chunks_count, chunks_data_json, faithfulness,
                        answer_relevance, context_precision, context_recall,
                        overall_score, verdict, critique, evaluation_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    str(comp.get("id") or ""),
                    comp.get("name", "Unnamed Strategy"),
                    badge,
                    comp.get("answer", ""),
                    float(lat.get("retrieval_ms", 0.0) or 0.0),
                    float(lat.get("rerank_ms", 0.0) or 0.0),
                    float(lat.get("generation_ms", 0.0) or 0.0),
                    float(lat.get("total_ms", 0.0) or 0.0),
                    len(chunks),
                    chunks_json_str,
                    faithfulness,
                    answer_relevance,
                    context_precision,
                    context_recall,
                    overall_score,
                    verdict,
                    critique,
                    eval_json_str
                ))
        return run_id
    except Exception as e:
        print(f"[Tracker] Error logging run: {e}")
        return None
    finally:
        conn.close()

def get_experiment_runs(limit=50, offset=0, experiment_type=None, search=None):
    """
    Retrieves recent experiment runs with summary metrics and optional filtering.
    """
    conn = get_db_connection()
    try:
        query = "SELECT * FROM experiment_runs WHERE 1=1"
        params = []

        if experiment_type and experiment_type.lower() != "all":
            query += " AND experiment_type = ?"
            params.append(experiment_type)

        if search and search.strip():
            query += " AND (question LIKE ? OR run_id LIKE ?)"
            term = f"%{search.strip()}%"
            params.extend([term, term])

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([int(limit) if limit is not None else 50, int(offset) if offset is not None else 0])

        rows = conn.execute(query, params).fetchall()
        runs = [dict(r) for r in rows]
        return runs
    finally:
        conn.close()

def get_experiment_run_details(run_id):
    """
    Fetches full metadata and all comparison cards for a specific run.
    Reconstructs the comparisons object for direct rendering by frontend card components.
    """
    conn = get_db_connection()
    try:
        run_row = conn.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,)).fetchone()
        if not run_row:
            return None

        run = dict(run_row)
        items_rows = conn.execute("SELECT * FROM experiment_items WHERE run_id = ? ORDER BY id ASC", (run_id,)).fetchall()

        comparisons = []
        for row in items_rows:
            r = dict(row)

            # Reconstruct chunks
            chunks = []
            if r.get("chunks_data_json"):
                try:
                    chunks = json.loads(r["chunks_data_json"])
                except Exception:
                    chunks = []

            # Reconstruct evaluation
            evaluation = None
            if r.get("evaluation_json"):
                try:
                    evaluation = json.loads(r["evaluation_json"])
                except Exception:
                    evaluation = None
            elif r.get("overall_score") is not None:
                faith = r.get("faithfulness")
                relev = r.get("answer_relevance")
                prec = r.get("context_precision")
                rec = r.get("context_recall")
                ov = r.get("overall_score")
                evaluation = {
                    "verdict": r.get("verdict"),
                    "overall_score": ov,
                    "overall_percentage": int(round(ov * 100)) if ov is not None else 0,
                    "critique": r.get("critique"),
                    "metrics": {
                        "faithfulness": {"score": faith, "percentage": int(round(faith * 100)) if faith is not None else 0},
                        "answer_relevance": {"score": relev, "percentage": int(round(relev * 100)) if relev is not None else 0},
                        "context_precision": {"score": prec, "percentage": int(round(prec * 100)) if prec is not None else 0},
                        "context_recall": {"score": rec, "percentage": int(round(rec * 100)) if rec is not None else 0}
                    }
                }

            comparisons.append({
                "id": r.get("strategy_id"),
                "name": r.get("strategy_name"),
                "badge": r.get("badge"),
                "answer": r.get("answer"),
                "retrieved_chunks": chunks,
                "latency": {
                    "retrieval_ms": float(r.get("retrieval_ms") or 0.0),
                    "rerank_ms": float(r.get("rerank_ms") or 0.0),
                    "generation_ms": float(r.get("generation_ms") or 0.0),
                    "total_ms": float(r.get("total_ms") or 0.0)
                },
                "evaluation": evaluation
            })

        return {
            "run": run,
            "comparisons": comparisons
        }
    finally:
        conn.close()

def delete_experiment_run(run_id):
    """
    Deletes a specific experiment run and its items.
    """
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("DELETE FROM experiment_items WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM experiment_runs WHERE run_id = ?", (run_id,))
        return True
    except Exception as e:
        print(f"[Tracker] Error deleting run: {e}")
        return False
    finally:
        conn.close()

def clear_all_runs():
    """
    Clears all saved experiment history.
    """
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("DELETE FROM experiment_items")
            conn.execute("DELETE FROM experiment_runs")
        return True
    except Exception as e:
        print(f"[Tracker] Error clearing runs: {e}")
        return False
    finally:
        conn.close()

def get_dashboard_analytics():
    """
    Calculates aggregated statistics across all historical runs:
    - Total runs and breakdown by experiment type
    - Average total latency across all runs
    - Fastest and slowest recorded strategies
    - Average evaluation metrics (faithfulness, answer relevance)
    - Strategy performance rankings
    """
    conn = get_db_connection()
    try:
        total_runs_row = conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()
        total_runs = total_runs_row[0] if total_runs_row else 0

        # Breakdown by experiment type
        type_rows = conn.execute("""
            SELECT experiment_type, COUNT(*) as count 
            FROM experiment_runs 
            GROUP BY experiment_type
        """).fetchall()
        type_counts = {r["experiment_type"]: r["count"] for r in type_rows}

        # Overall average latency and quality
        overall_stats = conn.execute("""
            SELECT 
                ROUND(AVG(total_ms), 1) as avg_latency,
                ROUND(AVG(faithfulness), 2) as avg_faithfulness,
                ROUND(AVG(answer_relevance), 2) as avg_relevance,
                ROUND(AVG(overall_score), 2) as avg_overall
            FROM experiment_items
        """).fetchone()

        # Latency & quality breakdown per strategy
        strat_rows = conn.execute("""
            SELECT 
                strategy_name,
                COUNT(*) as test_count,
                ROUND(AVG(total_ms), 1) as avg_latency,
                ROUND(AVG(retrieval_ms), 1) as avg_retrieval,
                ROUND(AVG(generation_ms), 1) as avg_generation,
                ROUND(AVG(overall_score), 2) as avg_quality
            FROM experiment_items
            GROUP BY strategy_name
            ORDER BY avg_latency ASC
        """).fetchall()
        strategy_stats = [dict(r) for r in strat_rows]

        # Verdict counts
        verdict_rows = conn.execute("""
            SELECT verdict, COUNT(*) as count
            FROM experiment_items
            WHERE verdict IS NOT NULL
            GROUP BY verdict
        """).fetchall()
        verdict_counts = {r["verdict"]: r["count"] for r in verdict_rows}

        avg_lat = 0.0
        avg_faith = 0.0
        avg_rel = 0.0
        avg_ov = 0.0
        if overall_stats:
            avg_lat = overall_stats["avg_latency"] or 0.0
            avg_faith = overall_stats["avg_faithfulness"] or 0.0
            avg_rel = overall_stats["avg_relevance"] or 0.0
            avg_ov = overall_stats["avg_overall"] or 0.0

        return {
            "total_runs": total_runs,
            "type_counts": type_counts,
            "avg_latency_ms": avg_lat,
            "avg_faithfulness": avg_faith,
            "avg_relevance": avg_rel,
            "avg_overall_score": avg_ov,
            "strategy_stats": strategy_stats,
            "verdict_counts": verdict_counts
        }
    finally:
        conn.close()

if __name__ == "__main__":
    print("Checking tracker database connection...")
    init_db()
    analytics = get_dashboard_analytics()
    print("Database connected successfully! Total runs:", analytics["total_runs"])
