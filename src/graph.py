from langgraph.graph import StateGraph, START, END
from state import CRAGState
from nodes import (
    retriever_node,
    grade_documents_node,
    rewrite_query_node,
    web_search_node,
    generate_node
)
from graders import hallucination_grader_chain
from telemetry import logger

# 1. Conditional Routing Functions (Edge Logics)

def decide_to_generate(state: CRAGState) -> str:
    """
    Evaluates whether internal retrieved documents are sufficient or if 
    the state machine must branch to web search fallback.
    """
    web_search_needed = state.get("web_search_needed", False)

    if web_search_needed:
        logger.info("DECISION: No relevant internal context found. Routing to 'rewrite_query'.")
        return "rewrite_query"

    logger.info("DECISION: Relevant context confirmed. Routing directly to 'generate'.")
    return "generate"

def grade_generation_grounding(state: CRAGState)-> str:
    """
    Audits the generated response against source context to catch hallucinations.
    If the response is grounded, it terminates at END.
    If an internal only generation hallucinated, it branches to web search for correction.
    """
    logger.info("AUDIT: Evaluating generated answer for factual grounding...")
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    web_search_needed = state.get("web_search_needed", False)

    # if context is empty, no further grounding audit is possible.
    if not documents:
        logger.info("AUDIT: Context empty. Terminating to END.")
        return "useful"

    # Concatenate document context for the evaluator
    context_text = "\n\n".join([doc.page_content for doc in documents])

    try:
        verdict = hallucination_grader_chain.invoke({
            "documents": context_text,
            "generation": generation
        })

        if isinstance(verdict, dict):
            score = verdict.get("binary_score")
            explanation = verdict.get("explanation", "")
        else:
            score = getattr(verdict, "binary_score", None)
            explanation = getattr(verdict, "explanation", "")

        if score == "yes":
            logger.info(f"AUDIT PASSED: Grounded in context. Reason: {explanation}")
            return "useful"
        else:
            logger.warning(f"AUDIT FAILED: Hallucination detected. Reason: {explanation}")

            # if we haven't tried web search yet, route to search to find factual ground truth
            if not web_search_needed:
                logger.info("Self-Correction: Triggering web search to remediate hallucination.")
                return "retry_with_web"

            # if web search was already run and failed, exit cleanly to prevent infinite loops
            return "useful"

    except Exception as exc:
        logger.error(f"Grounding audit encountered an error: {str(exc)}. Routing to END.", exc_info=True)
        return "useful"
# 2. State Graph Builder & Assembly

# Initilize the StateGraph with our structured memory schema
builder = StateGraph(CRAGState)

# Register the functional nodes
builder.add_node("retrieve", retriever_node)
builder.add_node("grade_documents", grade_documents_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

# Set up fixed transitions
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_documents")

# Set up conditional routing from grading
builder.add_conditional_edges(
     "grade_documents",
     decide_to_generate,
     {
         "rewrite_query": "rewrite_query",
         "generate": "generate"
     }
)

# setup fallback pipeline transitions
builder.add_edge("rewrite_query", "web_search")
builder.add_edge("web_search", "generate")

# Set up grounding verification conditional edge
builder.add_conditional_edges(
    "generate",
    grade_generation_grounding,
    {
        "useful": END,
        "retry_with_web": "rewrite_query"
    }
)

# Compile into an executable Runnable StateGraph
crag_app = builder.compile()