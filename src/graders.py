import warnings
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import config

# Silence non-critical Google SDK function-calling warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google")

# =====================================================================
# 1. EVALUATION SCHEMAS (Pydantic Data Models)
# =====================================================================

class ChunkEvaluation(BaseModel):
    """Relevance verdict for an individual document chunk."""
    chunk_index: int = Field(
        description="The 1-based index number corresponding to the chunk (e.g., 1, 2, 3)."
    )
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if the chunk contains keywords or concepts related to the question, otherwise 'no'."
    )
    explanation: str = Field(
        description="One brief sentence explaining why this specific chunk is or is not relevant."
    )


class BatchGradeDocuments(BaseModel):
    """Batch evaluation container for all candidate chunks in one call."""
    evaluations: List[ChunkEvaluation] = Field(
        description="A list containing the relevance evaluation for each retrieved chunk."
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

# Use Gemini 2.5 Flash-Lite for fast, low-latency evaluation
evaluator_llm = ChatGoogleGenerativeAI(
    model=config.grader_model,
    google_api_key=config.gemini_api_key,
    temperature=0.0,
    max_output_tokens=500
)


# =====================================================================
# 3. PROMPTS & STRUCTURED CHAINS
# =====================================================================

# --- OPTIMIZED BATCH GRADER: Evaluates all retrieved chunks at once ---
batch_doc_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert compliance auditor grading the relevance of retrieved document chunks to a user question.\n"
        "Carefully evaluate each numbered chunk independently.\n"
        "Rules:\n"
        "- If a chunk contains keywords, policies, or partial topical overlap to the question, grade it 'yes'.\n"
        "- If a chunk is completely irrelevant to the question, grade it 'no'.\n"
        "- You must return an evaluation entry for EVERY provided chunk index."
    )),
    ("human", (
        "USER QUESTION:\n{question}\n\n"
        "RETRIEVED DOCUMENT CHUNKS:\n{documents}"
    ))
])

batch_doc_grader_chain = batch_doc_grader_prompt | evaluator_llm.with_structured_output(
    BatchGradeDocuments,
    method="json_schema"
)


# --- GRADER 2: Hallucination / Grounding Audit ---
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


# --- GRADER 3: Answer Quality / Usefulness ---
answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an evaluator assessing whether an answer directly resolves the user's question.\n"
        "If the answer provides a clear and actionable response to what was asked, score it 'yes'.\n"
        "If the answer evades the question or goes off-topic, score it 'no'."
    )),
    ("human", (
        "USER QUESTION:\n{question}\n\n"
        "GENERATED ANSWER:\n{generation}"
    ))
])

answer_grader_chain = answer_grader_prompt | evaluator_llm.with_structured_output(
    GradeAnswer,
    method="json_schema"
)