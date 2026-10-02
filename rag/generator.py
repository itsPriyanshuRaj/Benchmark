from langchain_ollama import ChatOllama
import config

def format_context(context_chunks):
    """
    Combines the text of retrieved chunks into a single readable context string.
    """
    if not context_chunks:
        return "No relevant context found."

    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", 0) + 1
        parts.append(f"[Chunk {i} | Source: {source}, Page: {page}]\n{chunk.page_content}")

    return "\n\n".join(parts)

def generate_answer(question, context_chunks, model_name=None):
    """
    Generates an answer using the local Ollama LLM based on retrieved context.
    """
    if model_name is None:
        model_name = config.DEFAULT_OLLAMA_MODEL

    context_text = format_context(context_chunks)

    prompt = f"""You are a helpful AI assistant. Answer the user's question based strictly on the context provided below.
If the context does not contain enough information to answer the question, state that the answer is not available in the provided document.

Context:
{context_text}

Question:
{question}

Answer:"""

    try:
        llm = ChatOllama(model=model_name, temperature=0.2)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error communicating with local LLM (Ollama): {str(e)}"
