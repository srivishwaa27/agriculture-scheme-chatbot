"""
STEP 5: RAG Pipeline (Retrieval-Augmented Generation)
Core module: loads FAISS, creates retriever, calls HuggingFace LLM.
Can be imported by both the CLI chatbot and the Streamlit app.
"""

from __future__ import annotations

import os
from functools import lru_cache

from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ─────────────────────────────────────────
# CONFIGURATION  (override via env vars)
# ─────────────────────────────────────────
FAISS_DB_PATH   = os.getenv("FAISS_DB_PATH",   "data/faiss_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",  "sentence-transformers/all-MiniLM-L6-v2")
HF_MODEL        = os.getenv("HF_MODEL",         "Qwen/Qwen2.5-72B-Instruct")
TOP_K           = int(os.getenv("TOP_K", "12"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "1024"))

SYSTEM_PROMPT = (
    "You are an AI assistant that helps farmers understand "
    "Indian government agricultural schemes and policies. "
    "Be clear, concise, and practical."
)

RAG_PROMPT_TEMPLATE = """\
You are an expert on Indian government agricultural schemes.

Answer the question clearly and concisely. Base your answer primarily on the context provided below. If you have additional relevant knowledge about the scheme, you may include it to provide a complete answer.

Context:
{context}

Question:
{question}

Answer:"""


# ─────────────────────────────────────────
# LOAD RESOURCES (cached)
# ─────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_embedding_model() -> HuggingFaceEmbeddings:
    print(f"Loading embedding model: {EMBEDDING_MODEL} …")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _load_vector_db() -> FAISS:
    emb = _load_embedding_model()
    print(f"Loading FAISS index from: {FAISS_DB_PATH} …")
    return FAISS.load_local(
        FAISS_DB_PATH,
        emb,
        allow_dangerous_deserialization=True,
    )


def get_retriever():
    """Return a LangChain retriever backed by the FAISS index."""
    db = _load_vector_db()
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt(context: str, question: str) -> str:
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


# ─────────────────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────────────────
def rag_answer(question: str, hf_api_key: str) -> dict:
    """
    Run the full RAG pipeline.

    Parameters
    ----------
    question   : user's natural-language question
    hf_api_key : Hugging Face Inference API token

    Returns
    -------
    dict with keys:
        answer   : str  — generated answer
        sources  : list — retrieved document metadata
        context  : str  — concatenated context passed to the LLM
    """
    # 0. Basic query expansion for poor acronym understanding in small embedding models
    import re
    search_query = re.sub(r'\bPM\b', 'Pradhan Mantri', question, flags=re.IGNORECASE)

    # 1. Retrieve relevant chunks
    retriever = get_retriever()
    docs      = retriever.invoke(search_query)

    import re
    import json
    import os
    
    titles_map = {}
    titles_path = "scheme_titles.json"
    if os.path.exists(titles_path):
        try:
            with open(titles_path, "r", encoding="utf-8") as f:
                titles_map = json.load(f)
        except Exception:
            pass

    for doc in docs:
        source_file = doc.metadata.get("source", "")
        # Extract the scheme id, e.g., "scheme_496" from "scheme_496_chunk_1.txt"
        match = re.search(r"(scheme_\d+)", source_file)
        if match:
            scheme_id = match.group(1)
            if scheme_id in titles_map:
                real_title = titles_map[scheme_id]
                # Replace the source name with the real title for the UI
                doc.metadata["source"] = real_title
                # Replace the abstract "scheme_X" with the real title in the content context for the LLM
                doc.page_content = doc.page_content.replace(
                    f"Scheme Name : {scheme_id}",
                    f"Scheme Name : {real_title}"
                )

    context   = format_docs(docs)

    # 2. Build prompt
    prompt = build_prompt(context, question)

    # 3. Call LLM
    client = InferenceClient(api_key=hf_api_key)
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
    )

    answer  = response.choices[0].message.content
    sources = [d.metadata.get("source", "unknown") for d in docs]

    return {
        "answer":  answer,
        "sources": sources,
        "context": context,
    }
