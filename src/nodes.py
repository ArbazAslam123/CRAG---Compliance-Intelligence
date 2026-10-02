from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from state import CRAGState
from config import config
from telemetry import logger, track_node
from graders import batch_doc_grader_chain
from tools import execute_web_search
from database import KnowledgeBaseManager

# =====================================================================
# 1. RETRIEVER SINGLETON & LLM INITIALIZATION
# =====================================================================

# Singleton instance so we never re-index embeddings during app reruns
_retriever_instance = None

def get_retriever():
    """Lazily initializes and caches the Qdrant retriever in memory."""
    global _retriever_instance
    if _retriever_instance is None:
        logger.info("Initializing vector store retriever singleton...")
        kb_manager = KnowledgeBaseManager("data/enterprise_policy.txt")
        vector_store = kb_manager.build_vector_store()
        _retriever_instance = vector_store.as_retriever(search_kwargs={"k": 3})
    return _retriever_instance


# Generator LLM: High fidelity synthesis
generator_llm = ChatGoogleGenerativeAI(
    model=config.generator_model,
    google_api_key=config.gemini_api_key,
    temperature=0.1,
    max_output_tokens=1000
)

# Query Rewriter LLM: Fast, deterministic rewrites
rewrite_llm = ChatGoogleGenerativeAI(
    model=config.grader_model,
    google_api_key=config.gemini_api_key,
    temperature=0.0,
    max_output_tokens=150
)


# =====================================================================
# 2. STATE MACHINE NODES
# =====================================================================

@track_node("retriever")
def retriever_node(state: CRAGState) -> Dict[str, Any]:
    """Queries the cached Qdrant vector database using the current user question."""
    question = state["question"]
    logger.info(f"Retrieving candidate chunks for: '{question}'")

    retriever = get_retriever()
    documents = retriever.invoke(question)
    logger.info(f"Retrieved {len(documents)} raw chunks from knowledge base.")

    return {"documents": documents}


@track_node("grade_documents")
def grade_documents_node(state: CRAGState) -> Dict[str, Any]:
    """
    Grades all retrieved chunks in a single batched network call.
    Filters out noise and determines whether fallback web search is required.
    """
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.warning("No documents present to grade. Triggering web search.")
        return {"documents": [], "web_search_needed": True}

    # Format all retrieved chunks into a single indexed text block
    formatted_chunks = []
    for idx, doc in enumerate(documents):
        formatted_chunks.append(f"[Chunk Index {idx + 1}]\n{doc.page_content}")
    chunks_block = "\n\n---\n\n".join(formatted_chunks)

    filtered_docs: List[Document] = []
    web_search_needed = False

    try:
        logger.info(f"Batch grading {len(documents)} chunks in a single LLM call...")
        batch_result: Any = batch_doc_grader_chain.invoke({
            "question": question,
            "documents": chunks_block
        })

        # Map decisions by chunk index
        eval_map = {item.chunk_index: item for item in batch_result.evaluations}

        for idx, doc in enumerate(documents):
            chunk_num = idx + 1
            evaluation = eval_map.get(chunk_num)

            if evaluation and evaluation.binary_score == "yes":
                logger.info(f"Chunk {chunk_num}: RELEVANT. Reason: {evaluation.explanation}")
                filtered_docs.append(doc)
            else:
                reason = evaluation.explanation if evaluation else "Not evaluated"
                logger.warning(f"Chunk {chunk_num}: IRRELEVANT. Reason: {reason}")

    except Exception as exc:
        logger.error(f"Batch grading failed: {str(exc)}. Retaining all chunks defensively.", exc_info=True)
        filtered_docs = documents

    if not filtered_docs:
        logger.warning("All chunks scored irrelevant. Routing to Web Search.")
        web_search_needed = True
    else:
        logger.info(f"{len(filtered_docs)}/{len(documents)} chunks passed relevance threshold.")

    return {
        "documents": filtered_docs,
        "web_search_needed": web_search_needed
    }


@track_node("rewrite_query")
def rewrite_query_node(state: CRAGState) -> Dict[str, Any]:
    """Optimizes user question for external search engines by stripping internal references."""
    question = state["question"]
    logger.info(f"Transforming query for external search: '{question}'")

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert query optimizer. The user asked a question that could not be "
            "answered by internal company documents. Convert the question into a concise, keyword-rich "
            "search engine query suitable for finding up-to-date facts on Google/Tavily. "
            "Do NOT include company-specific internal codes in the query. Return ONLY the rewritten query text."
        )),
        ("human", "Original Question:\n{question}\n\nOptimized Search Query:")
    ])

    chain = rewrite_prompt | rewrite_llm | StrOutputParser()
    rewritten_query = chain.invoke({"question": question}).strip()

    logger.info(f"Query rewritten: '{question}' -> '{rewritten_query}'")
    return {"question": rewritten_query}


@track_node("web_search")
def web_search_node(state: CRAGState) -> Dict[str, Any]:
    """Executes external web search via Tavily and appends results to state documents."""
    question = state["question"]
    logger.info(f"Executing web search fallback for: '{question}'")

    web_docs = execute_web_search(query=question, max_results=3)
    existing_docs = state.get("documents", [])

    return {"documents": existing_docs + web_docs}


@track_node("generate")
def generate_node(state: CRAGState) -> Dict[str, Any]:
    """Synthesizes the final answer using Gemini, grounded strictly on the verified context."""
    original_question = state.get("original_question", state["question"])
    documents = state.get("documents", [])

    context_blocks = []
    for doc in documents:
        src = doc.metadata.get("source", "internal_policy")
        chunk_id = doc.metadata.get("chunk_id", "doc")
        context_blocks.append(f"[Source: {src} | ID: {chunk_id}]\n{doc.page_content}")

    formatted_context = "\n\n---\n\n".join(context_blocks)

    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict, professional corporate compliance assistant.\n"
            "Answer the user's question using ONLY the provided verified context.\n"
            "Rules:\n"
            "1. If facts or numbers are provided, state them accurately.\n"
            "2. Cite your sources inline using [Source: <name> | ID: <id>].\n"
            "3. If the context does not contain enough information, state: "
            "'I do not have enough verified information to answer this question accurately.'\n"
            "4. Do NOT speculate or extrapolate beyond the provided text."
        )),
        ("human", (
            "VERIFIED CONTEXT:\n{context}\n\n"
            "USER QUESTION:\n{question}\n\n"
            "ANSWER:"
        ))
    ])

    chain = generation_prompt | generator_llm | StrOutputParser()
    answer = chain.invoke({
        "context": formatted_context,
        "question": original_question
    })

    logger.info("Successfully synthesized grounded answer.")
    return {"generation": answer}