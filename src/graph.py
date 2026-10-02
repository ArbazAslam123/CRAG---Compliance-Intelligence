from langgraph.graph import StateGraph, START, END
from state import CRAGState
from nodes import (
    retriever_node,
    grade_documents_node,
    rewrite_query_node,
    web_search_node,
    generate_node
)
from telemetry import logger

# =====================================================================
# 1. CONDITIONAL ROUTING FUNCTION
# =====================================================================

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


# =====================================================================
# 2. STATE GRAPH BUILDER & ASSEMBLY
# =====================================================================

# Initialize the StateGraph with the shared memory schema
builder = StateGraph(CRAGState)

# Register functional nodes
builder.add_node("retrieve", retriever_node)
builder.add_node("grade_documents", grade_documents_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

# Fixed entry pipeline
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_documents")

# Conditional fork: internal generation vs external web fallback
builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "rewrite_query": "rewrite_query",
        "generate": "generate"
    }
)

# Web fallback pipeline transitions
builder.add_edge("rewrite_query", "web_search")
builder.add_edge("web_search", "generate")

# Terminate immediately after generation (eliminates redundant audit delay)
builder.add_edge("generate", END)

# Compile into executable Runnable StateGraph
crag_app = builder.compile()