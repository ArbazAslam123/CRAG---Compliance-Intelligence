from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document

class CRAGState(TypedDict):
    """
    The shared memory schema passed between every node in the LangGraph workflow.
    Every key represents a piece of information the agent needs to track.
    """
    question: str                         # The active question (may be rewritten for web search)
    original_question: str                # The raw user question preserved for provenance
    documents: List[Document]             # The list of chunks that survived relevance grading
    web_search_needed: bool               # A boolean flag: True if local documents were irrelevant
    generation: str                       # The final generated response string
    iteration_count: int                  # Safety counter to prevent infinite self-correction loops
    hallucination_verdict: Optional[str]  # Evaluation verdict: 'grounded' or 'hallucinated'