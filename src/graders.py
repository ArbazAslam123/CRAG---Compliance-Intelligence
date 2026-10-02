from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
from config import config

# 1. EVALUATION SCHEMAS (pydantic Data Models)

class GradeDocuments(BaseModel):
    """Binary score for checking whether a retrieved document is relevant to the question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if the document contains keywords or semantic meaning related to the question, otherwise 'no'."
    )
    explanation: str = Field(
        description="One brief sentence explaining why the document is or is not relevant"
    )

class GradeHallucinations(BaseModel):
    """Binary score for checking whether an answer is grounded in the retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Grounding score: 'yes' if the answer is strictly supported by the retrieved facts, 'no' if it invents information."
    )
    explanation: str = Field(
        description="Reasoning on whether the facts in the answer exist in the source context."
    )

class GradeAnswer(BaseModel):
    """Binary score to evaluate if the answer actually resolves the user's question."""
    binary_score: Literal['yes', 'no'] = Field(
        description="Resolution score:'yes' if the asnwer directly and fully addresses the question, otherwise 'no'."
    )
    explanation: str = Field (
        description="Brief justification of whether the user question was resolved."
    )

# 2. Model Initialization

# we use gemini 2.5 flash with temperature=0 for deterministic evaluation
evaluator_llm = ChatGoogleGenerativeAI(
    model=config.grader_model,
    google_api_key=config.gemini_api_key,
    temperature=0.0,
    max_output_tokens=300,
)

# 3. Prompts & Structured Chains

# Grader 1: Document Relevance
doc_grader_prompt= ChatPromptTemplate.from_messages([
    ("system",(
        "You are an expert compliance auditor grading the relevance of a retrieved document to a user question.\n"
        "Carefully evaluate if the document contains keywords, concepts, or policies related to the user question.\n"
        "If it contains relevant information, grade it as 'yes'. If it is completely unrelated, grade it as 'no'.\n"
        "Do not be overly strict if the document has even partial topical overlap, score it 'yes'."
    )),
    ("human", (
        "RETRIEVED DOCUMENT:\n{document}\n\n"
        "USER QUESTION:\n{question}"
    ))
])

# Bind the Pydantic schema using Gemini's native JSON schema generation
doc_grader_chain = doc_grader_prompt | evaluator_llm.with_structured_output(
    GradeDocuments,
    method="json_schema"
)

# Grader 2: Hallucination / Grounding
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict compliance auditor assessing whether an AI generated answer is grounded in the provided facts.\n"
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

# Grader 3: Answer Quality / Usefulness
answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an evaluator assessing whether an answer directly resolves the user's question.\n"
        "If the answer provides a clear and actionable response to what was asked, score it 'yes'.\n"
        "If the answer evades the question, states missing context without resolution, or goes off-topic, score it 'no'."
    )),
    ("human", (
        "USER QUESTION:\n{question}\n\n"
        "GENERATED ANSWER:\n{generation}"
    ))
])

answer_grader_chain = answer_grader_prompt | evaluator_llm.with_structured_output(
    GradeAnswer,
    method = "json_schema"
)


# Silence the non-critical Google GenAI Automatic Function Calling warning
warnings.filterwarnings("ignore", category=UserWarning, module="google")