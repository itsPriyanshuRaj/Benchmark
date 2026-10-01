# RAGBench AI — Development Context

## Project
RAGBench AI is a simple RAG evaluation/experimentation platform built with **Python + Flask**. The goal is to start with a minimal working RAG MVP and incrementally turn it into a tool for comparing RAG configurations.

## User Preferences
- Keep explanations short and simple.
- Build incrementally; do NOT dump the whole project at once.
- Explain the purpose/why before introducing each component.
- Prefer practical implementation over textbook theory.
- Keep MVP intentionally small.
- Do not introduce advanced components until the basic version works.
- This project is **Python/Flask only**; do not switch to Java/Spring Boot.

## Current Phase: MVP
The MVP goal is a basic end-to-end RAG pipeline:

PDF upload
→ PDF loading
→ Recursive chunking
→ embeddings
→ Chroma vector store
→ question
→ retrieve top-K chunks
→ local LLM
→ answer + retrieved chunks

### MVP Features
1. Upload PDF
2. Process and index PDF
3. Ask a question
4. Retrieve relevant chunks
5. Generate answer
6. Display answer and retrieved chunks

### Explicitly NOT in MVP
- LangGraph
- Reranking
- Multiple embedding models
- Multiple chunking strategies
- Evaluation metrics
- Dashboard
- Automated optimization
- Complex agent workflows

## Planned MVP Stack
- Python
- Flask
- LangChain
- Chroma
- Sentence Transformers embeddings
- Ollama + local LLM (Qwen/Llama)
- Simple HTML/JS frontend

## Initial Project Structure
ragbench/
├── app.py
├── config.py
├── routes/
│   ├── upload.py
│   └── query.py
├── rag/
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── generator.py
├── templates/
│   └── index.html
├── uploads/
└── requirements.txt

## Development Rules
- Work one step/component at a time.
- After each meaningful step, verify it works before moving on.
- Prefer simple code and clear module boundaries.
- Avoid premature abstraction.
- Keep dependencies minimal.
- If an error occurs, debug the current step before adding new architecture.
- Preserve working code when moving to the next phase.

## Future Phases (Do Not Build Yet)

### Phase 2 — RAG Bench
Compare configurations:
- chunk size / overlap
- embedding models
- top-K
- retrieval strategies

### Phase 3 — Retrieval Quality
- reranking
- hybrid retrieval
- better retrieval experiments

### Phase 4 — Evaluation
- retrieval recall
- context precision
- answer relevance
- faithfulness
- latency
- failure cases

### Phase 5 — Experiment Dashboard
Store and compare experiment runs and results.

### Phase 6 — Auto Optimization
Use an agent/workflow to identify weak RAG configurations and suggest/run the next experiment.

## Current Decision
We are starting with **MVP only**. The immediate objective is to get one PDF → question → retrieved context → LLM answer pipeline working reliably before adding any benchmarking features.

## Handoff Instruction
When continuing this project, first read this file and identify:
1. Current phase
2. What has already been implemented
3. What is the next smallest step

Do not assume future phases are implemented just because they are listed here.
