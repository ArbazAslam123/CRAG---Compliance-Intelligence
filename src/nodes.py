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

# Cache retriever singleton so we never re-index in memory
_retriever_instance = None

def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        logger.info("Initializing cached vector store...")
        kb_manager = KnowledgeBaseManager("data/enterprise_policy.txt")
        vector_store = kb_manager.build_vector_store()
        _retriever_instance = vector_store.as_retriever(search_kwargs={"k": 3})
    return _retriever_instance

generator_llm = ChatGoogleGenerativeAI(
    model=config.generator_model,
    google_api_key=config.gemini_api_key,
    temperature=0.1,
    max_output_tokens=1000
)

rewrite_llm = ChatGoogleGenerativeAI(
    model=config.grader_model,
    google_api_key=config.gemini_api_key,
    temperature=0.0,
    max_output_tokens=150
)

@track_node("retrieve")
def retriever_node(state: CRAGState) -> Dict[str, Any]:
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents}

@track_node("grade_documents")
def grade_documents_node(state: CRAGState) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {"documents": [], "web_search_needed": True}

    # Format all retrieved chunks into a single indexed text block
    formatted_chunks = [
        f"[Chunk Index {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(documents)
    ]
    chunks_block = "\n\n---\n\n".join(formatted_chunks)

    filtered_docs: List[Document] = []
    web_search_needed = False

    try:
        # SINGLE LLM call instead of 3 separate calls
        batch_result: Any = batch_doc_grader_chain.invoke({
            "question": question,
            "documents": chunks_block
        })

        eval_map = {item.chunk_index: item for item in batch_result.evaluations}

        for idx, doc in enumerate(documents):
            chunk_num = idx + 1
            evaluation = eval_map.get(chunk_num)
            if evaluation and evaluation.binary_score == "yes":
                filtered_docs.append(doc)

    except Exception as exc:
        logger.error(f"Batch grading error: {str(exc)}. Retaining chunks defensively.", exc_info=True)
        filtered_docs = documents

    if not filtered_docs:
        web_search_needed = True

    return {
        "documents": filtered_docs,
        "web_search_needed": web_search_needed
    }

@track_node("rewrite_query")
def rewrite_query_node(state: CRAGState) -> Dict[str, Any]:
    question = state["question"]
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert query optimizer. Convert the question into a concise keyword query "
            "for search engines. Remove company-specific references. Return ONLY the search query."
        )),
        ("human", "Original Question:\n{question}\n\nSearch Query:")
    ])
    chain = rewrite_prompt | rewrite_llm | StrOutputParser()
    rewritten_query = chain.invoke({"question": question}).strip()
    return {"question": rewritten_query}

@track_node("web_search")
def web_search_node(state: CRAGState) -> Dict[str, Any]:
    question = state["question"]
    web_docs = execute_web_search(query=question, max_results=3)
    existing_docs = state.get("documents", [])
    return {"documents": existing_docs + web_docs}

@track_node("generate")
def generate_node(state: CRAGState) -> Dict[str, Any]:
    original_question = state.get("original_question", state['question'])
    documents = state.get("documents", [])

    context_blocks = [
        f"[Source: {doc.metadata.get('source', 'internal')} | ID: {doc.metadata.get('chunk_id', 'doc')}]\n{doc.page_content}"
        for doc in documents
    ]
    formatted_context = "\n\n---\n\n".join(context_blocks)

    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict corporate compliance assistant. Answer using ONLY the verified context.\n"
            "Cite sources inline using [Source: <name> | ID: <id>]. If information is missing, state it."
        )),
        ("human", "VERIFIED CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:")
    ])
    chain = generation_prompt | generator_llm | StrOutputParser()
    answer = chain.invoke({
        "context": formatted_context,
        "question": original_question
    })
    return {"generation": answer}