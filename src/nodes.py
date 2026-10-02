from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from state import CRAGState
from config import config
from telemetry import logger, track_node
from graders import doc_grader_chain
from tools import execute_web_search
from database import KnowledgeBaseManager

# 1. Initialize Shared Retriever & Generator LLM

# Initialize the vector database once so it stays warm in memory
kb_manager = KnowledgeBaseManager("data/enterprise_policy.txt")
vector_store = kb_manager.build_vector_store()
# retireve the top 3 most semantically similar chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Synthesis LLM: Generate the final grounded search queries
generator_llm = ChatGoogleGenerativeAI(
    model = config.generator_model,
    google_api_key = config.gemini_api_key,
    temperature=0.1,
    max_output_tokens = 1000
)

# Rewrite LLM: Uses fast Flash-Lite to reformulate search queries
rewrite_llm = ChatGoogleGenerativeAI(
    model = config.grader_model,
    google_api_key = config.gemini_api_key,
    temperature = 0.0,
    max_output_tokens = 150
)

# 2. Graph Nodes (state processors)

@track_node("retriever")
def retriever_node(state: CRAGState) -> Dict[str, Any]:
    """
    Queries the Qdrant vector database using the current user question.
    """
    question = state["question"]
    logger.info(f"Retrieving chunks form Qdrant for question: '{question}'")

    documents = retriever.invoke(question)
    logger.info(f"Retrieved {len(documents)} raw chunks from knowledge base.")

    # Update the 'documents' list in state
    return {"documents": documents}

@track_node("grade_documents")
def grade_documents_node(state: CRAGState) -> Dict[str, Any]:
    """
    Grades every retrieved document for relevance using Gemini.
    Filters out noise and decide if a fallback web search is needed.
    """
    question = state["question"]
    documents = state.get("documents", [])

    filtered_docs: List[Document] = []
    web_search_needed = False

    logger.info(f"Grading {len(documents)} retrieved documents against question...")

    for idx, doc in enumerate(documents):
        try:
            # Call the structured Pydantic grader chain built in src/graders.py
            # The chain's inferred return type may be a dict or a Pydantic model;
            # both expose the structured grader fields at runtime.
            grade: Any = doc_grader_chain.invoke({
                "question": question,
                "document": doc.page_content
            })

            score = grade.binary_score
            explanation = grade.explanation

            if score == "yes":
                logger.info(f"Chunk {idx + 1}: RELEVANT. Reason: {explanation}")
                filtered_docs.append(doc)
            else:
                logger.warning(f"Chunks {idx + 1} Irrelevant. Reason: {explanation}")

        except Exception as exc:
            logger.error(f"Grading failed for chunk {idx + 1}: {str(exc)}. Retaining chunk by default")
            # Defensive design: Keep the chunk if evaluation fails unexpectedly
            filtered_docs.append(doc)

    # CRAG Decision Rule: If no documents survived grading, trigger web search
    if not filtered_docs:
        logger.warning("No retrieved documents passed relevance threshold. Triggering Web Search.")
        web_search_needed = True
    else:
        logger.info(f"{len(filtered_docs)}/{len(documents)} documents retained for generation.")

    return {"documents": filtered_docs,
            "web_search_needed": web_search_needed
            }

@track_node("rewrite_query")
def rewrite_query_node(state: CRAGState) -> Dict[str, Any]:
    """
    Rewrites the user question into an optimized web search query.
    Extracts core entities and removes internal conversational phrasing.
    """
    question = state["question"]
    logger.info(f"Transforming query for external search: '{question}'")

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert query optimizer. The user asked a question that could not be"
            "answered by internal company documents. COnvert the question into a concise, keyword-rich"
            "search engine query suitable for finding up to date facts on Google/Tavily."
            "Do NOT include company specific internal codes in the query. Return ONLY the rewritten query text."
        )),
        ("human", "Original Question:\n{question}\n\nOptimized Search Query:")
    ])

    chain = rewrite_prompt | rewrite_llm | StrOutputParser()
    rewritten_query = chain.invoke({"question": question}).strip()

    logger.info(f"Query rewritten: '{question}' -> '{rewritten_query}'")
    return {"question": rewritten_query}

@track_node("web_search")
def web_search_node(state: CRAGState) -> Dict[str, Any]:
    """
    Executes external web search via Tavily and appends results to state documents.
    """
    question = state["question"]
    logger.info(f"Executing web search fallback for: '{question}'")

    web_docs = execute_web_search(query=question, max_results=3)
    existing_docs = state.get("documents", [])

    # combine surviving local documents (if any) with the fresh web snippets
    combined_docs = existing_docs + web_docs
    logger.info(f"Added {len(web_docs)} web search chunks to context.")

    return {"documents": combined_docs}

@track_node("generate")
def generate_node(state: CRAGState) -> Dict[str, Any]:
    """
    Synthesizes the final answer using Gemini, grounded strictly on the verified context.
    """
    original_question = state.get("original_question", state['question'])
    documents = state.get("documents", [])

    # Format all chunk contents and their source citations into a clean context block
    context_blocks = []
    for doc in documents:
        src=doc.metadata.get("source", "internal_policy")
        chunk_id = doc.metadata.get("chunk_id", "doc")
        context_blocks.append(f"[source: {src} | ID: {chunk_id}]\n{doc.page_content}")

    formatted_context = "\n\n---\n\n".join(context_blocks)

    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict, professional corporate complaince assistant.\n"
            "Answer the user's question usng ONLY the provided verified context.\n"
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