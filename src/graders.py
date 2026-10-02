import warnings
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import config

# Silence non-critical Google SDK function-calling warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google")

# =====================================================================
# 1. ULTRA-LEAN EVALUATION SCHEMAS (Pydantic Data Models)
# =====================================================================

class LeanDocGrading(BaseModel):
    """
    Returns only the 1-based indices of relevant chunks.
    Generating ~5 tokens instead of ~300 eliminates API throttling.
    """
    relevant_indices: List[int] = Field(
        default_factory=list,
        description="List of 1-based chunk numbers (e.g., [1, 3] or [] if none) containing facts relevant to answering the question."
    )


class GradeHallucinations(BaseModel):
    """Binary score checking whether the answer is grounded in source context."""
    binary_score: Literal["yes", "no"] = Field(
        description="Grounding score: 'yes' if all facts are strictly supported by the text, 'no' if facts are invented."
    )
    explanation: str = Field(
        description="Reasoning on whether the facts in the answer exist in the source context."
    )


class GradeAnswer(BaseModel):
    """Binary score to evaluate if the answer directly resolves the question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Resolution score: 'yes' if the answer directly addresses the question, otherwise 'no'."
    )
    explanation: str = Field(
        description="Brief justification of whether the user question was resolved."
    )


# =====================================================================
# 2. MODEL INITIALIZATION
# =====================================================================

# Fast Flash-Lite evaluator restricted to a tiny output token window
evaluator_llm = ChatGoogleGenerativeAI(
    model=config.grader_model,
    google_api_key=config.gemini_api_key,
    temperature=0.0,
    max_output_tokens=100
)


# =====================================================================
# 3. LEAN BATCH GRADER CHAIN
# =====================================================================

fast_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert compliance relevance filter.\n"
        "Evaluate the numbered candidate text chunks against the user question.\n"
        "Identify which chunks contain policies, keywords, or facts directly relevant to answering the question.\n"
        "Return ONLY the list of 1-based chunk indices (e.g., [1, 2] or [] if none are relevant)."
    )),
    ("human", (
        "USER QUESTION:\n{question}\n\n"
        "CANDIDATE CHUNKS:\n{documents}"
    ))
])

fast_doc_grader_chain = fast_grader_prompt | evaluator_llm.with_structured_output(
    LeanDocGrading,
    method="json_schema"
)

# Hallucination chain retained for audit compatibility
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict compliance auditor assessing whether an AI-generated answer is grounded in the provided facts.\n"
        "Review the source documents and the generation. If every claim in the generation is supported by the facts, "
        "score it 'yes'. If the generation introduces external facts or unverified assertions not found in the documents, score it 'no'."
    )),
    ("human", (
        "SOURCE DOCUMENTS:\n{documents}\n\n"
        "GENERATED ANSWER:\n{generation}"
    ))
])

hallucination_grader_chain = hallucination_prompt | evaluator_llm.with_structured_output(
    GradeHallucinations,
    method="json_schema"
)