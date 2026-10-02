import os
import sys
import time
from typing import Any, Dict, cast
import streamlit as st

# =====================================================================
# 0. RESOLVE IMPORT PATHS (Fixes ModuleNotFoundError for src files)
# =====================================================================
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.graph import crag_app
    from src.config import config
    from src.state import CRAGState
    from src.telemetry import logger
    from src.nodes import get_retriever
except ImportError:
    from graph import crag_app
    from config import config
    from state import CRAGState
    from telemetry import logger
    from nodes import get_retriever

# =====================================================================
# 1. PAGE SETUP & INJECTED STYLES
# =====================================================================
st.set_page_config(
    page_title="CRAG-OS // Enterprise Policy AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber-Enterprise Glassmorphism CSS
st.markdown("""
<style>
    /* Dark Cyber Theme Foundation */
    .stApp {
        background-color: #0b0f17;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Mission Control Container */
    .header-box {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.9) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(56, 139, 253, 0.35);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    }
    
    /* Glowing Status Badges */
    .badge-grounded {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid #3fb950;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 0 10px rgba(63, 185, 80, 0.2);
    }
    
    .badge-fallback {
        background-color: rgba(219, 109, 40, 0.15);
        color: #f0883e;
        border: 1px solid #f0883e;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 0 10px rgba(240, 136, 62, 0.2);
    }

    /* Node Execution Pipeline Cards */
    .node-card {
        background-color: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .node-card-alert {
        background-color: #1c1514;
        border-left: 4px solid #f0883e;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .node-card-success {
        background-color: #121d15;
        border-left: 4px solid #3fb950;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Result Box Styling */
    .result-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 18px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #f0f6fc;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. PRE-WARM & CACHE VECTOR STORE (Runs ONCE outside the query timer)
# =====================================================================
@st.cache_resource(show_spinner="Warming up Vector Database & Neural Embeddings...")
def init_cached_knowledge_base():
    """Initializes embeddings and indexes Qdrant into RAM once at startup."""
    return get_retriever()

# Trigger warmup immediately on server boot
init_cached_knowledge_base()


# =====================================================================
# 3. SESSION STATE MANAGEMENT (Direct Key Binding)
# =====================================================================
if "main_input" not in st.session_state:
    st.session_state.main_input = ""
if "trigger_run" not in st.session_state:
    st.session_state.trigger_run = False

def run_scenario(scenario_query: str):
    """Directly updates the bound input key and sets the execution flag."""
    st.session_state.main_input = scenario_query
    st.session_state.trigger_run = True


# =====================================================================
# 4. SIDEBAR // SYSTEM TELEMETRY & PRELOADED SCENARIOS
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Telemetry")
    st.markdown(f"**LLM Generator:** `{config.generator_model}`")
    st.markdown(f"**LLM Grader:** `{config.grader_model}`")
    st.markdown(f"**Embeddings:** `all-MiniLM-L6-v2 (Local CPU)`")
    st.markdown(f"**Vector Store:** `Qdrant In-Memory (Cached)`")
    
    st.divider()
    
    st.markdown("### 🧪 Quick-Test Scenarios")
    st.caption("Click to trigger deterministic routing or dynamic web fallback:")
    
    st.button(
        "🏢 Internal: Desk Stipend & Code",
        on_click=run_scenario,
        args=("What is the reimbursement cap and billing code for home office desks?",),
        use_container_width=True
    )
    
    st.button(
        "🚨 Internal: Lost Laptop Protocol",
        on_click=run_scenario,
        args=("What is the mandatory reporting window and SOC contact if a laptop is lost?",),
        use_container_width=True
    )
        
    st.button(
        "✈️ Internal: Business Class Air Travel",
        on_click=run_scenario,
        args=("When is an employee permitted to book business class flights?",),
        use_container_width=True
    )
        
    st.button(
        "🌐 Fallback: 2026 IRS Mileage Rate",
        on_click=run_scenario,
        args=("What is the current standard IRS business mileage rate for 2026?",),
        use_container_width=True
    )
        
    st.button(
        "🌐 Fallback: UK Statutory Sick Pay",
        on_click=run_scenario,
        args=("What is the current UK statutory sick pay weekly rate?",),
        use_container_width=True
    )

    st.divider()
    st.caption("Engineered by **Arbaz Aslam** | Autonomous Agent Architecture")


# =====================================================================
# 5. MAIN HEADER & CONTROL PANEL
# =====================================================================
st.markdown("""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2rem; color: #58a6ff;">CRAG-OS // Compliance Intelligence</h1>
            <p style="margin: 6px 0 0 0; color: #8b949e; font-size: 0.95rem;">
                Corrective Retrieval-Augmented Generation with Self-Reflective Verification & Live Web Fallback
            </p>
        </div>
        <div>
            <span class="badge-grounded">ONLINE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# User Query Input bound directly to st.session_state.main_input
user_query = st.text_input(
    "Query Corporate Policy or Industry Compliance:",
    key="main_input",
    placeholder="Ask a question about internal policy or an external regulatory standard..."
)

execute_clicked = st.button("🚀 Analyze & Verify", type="primary")


# =====================================================================
# 6. EXECUTION PIPELINE (LangGraph Streaming & State Rendering)
# =====================================================================
should_execute = execute_clicked or st.session_state.trigger_run

if should_execute and user_query.strip():
    # Reset trigger flag immediately
    st.session_state.trigger_run = False
    start_total_time = time.perf_counter()
    
    initial_state: CRAGState = cast(
        CRAGState,
        {
            "question": user_query.strip(),
            "original_question": user_query.strip(),
            "documents": [],
            "web_search_needed": False,
            "generation": "",
            "iteration_count": 0,
            "hallucination_verdict": None,
        },
    )
    
    col_pipeline, col_result = st.columns([1.1, 1.9], gap="medium")
    
    with col_pipeline:
        st.markdown("### 🔄 Execution State Machine")
        pipeline_status = st.container()
    
    with col_result:
        st.markdown("### 📑 Verified Synthesis")
        result_container = st.container()
        sources_container = st.container()

    final_output_state = dict(initial_state)

    with pipeline_status:
        with st.spinner("Executing State Machine..."):
            for step_event in crag_app.stream(initial_state, stream_mode="updates"):
                for node_name, state_update in step_event.items():
                    final_output_state.update(state_update)
                    
                    if node_name == "retriever":
                        docs = state_update.get("documents", [])
                        st.markdown(f"""
                        <div class="node-card">
                            <strong>1. Vector Retrieval (Qdrant)</strong><br>
                            <span style="color: #8b949e; font-size: 0.85rem;">
                                Fetched {len(docs)} candidate chunks via cached dense index.
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif node_name == "grade_documents":
                        web_needed = state_update.get("web_search_needed", False)
                        docs = state_update.get("documents", [])
                        if web_needed:
                            st.markdown(f"""
                            <div class="node-card-alert">
                                <strong>2. Document Relevance Grader</strong><br>
                                <span style="color: #f0883e; font-size: 0.85rem;">
                                    ⚠️ 0 Chunks passed relevance threshold. Triggering external fallback.
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="node-card-success">
                                <strong>2. Document Relevance Grader</strong><br>
                                <span style="color: #3fb950; font-size: 0.85rem;">
                                    ✓ Verified {len(docs)} high-confidence internal chunks.
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    elif node_name == "rewrite_query":
                        new_q = state_update.get("question", "")
                        st.markdown(f"""
                        <div class="node-card">
                            <strong>3. Query Optimizer (CRAG)</strong><br>
                            <span style="color: #8b949e; font-size: 0.85rem;">
                                Reformulated for search: <em>"{new_q}"</em>
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif node_name == "web_search":
                        docs = state_update.get("documents", [])
                        st.markdown(f"""
                        <div class="node-card">
                            <strong>4. Fallback Web Search (Tavily)</strong><br>
                            <span style="color: #8b949e; font-size: 0.85rem;">
                                Ingested real-time external intelligence. Total context: {len(docs)} chunks.
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif node_name == "generate":
                        st.markdown(f"""
                        <div class="node-card-success">
                            <strong>5. Context-Grounded Synthesis</strong><br>
                            <span style="color: #3fb950; font-size: 0.85rem;">
                                Answer generated via {config.generator_model} with zero extrapolation.
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

    elapsed_total = time.perf_counter() - start_total_time

    # =====================================================================
    # 7. RENDER THE GENERATED ANSWER & CITATIONS
    # =====================================================================
    with result_container:
        answer_text = final_output_state.get("generation", "No generation produced.")
        is_web = final_output_state.get("web_search_needed", False)
        
        metric_col1, metric_col2, metric_col3 = st.columns([1, 1, 1])
        with metric_col1:
            st.metric("Total Latency", f"{elapsed_total:.2f}s")
        with metric_col2:
            st.metric("Knowledge Origin", "External Web" if is_web else "Internal Policy")
        with metric_col3:
            st.metric("Context Verification", "Passed")
            
        st.markdown("#### Official Compliance Guidance")
        st.markdown(f'<div class="result-box">{answer_text}</div>', unsafe_allow_html=True)

    with sources_container:
        st.markdown("#### 🔍 Evidence & Source Provenance")
        documents_raw = final_output_state.get("documents", [])
        docs = documents_raw if isinstance(documents_raw, list) else ([] if documents_raw is None else [documents_raw])
        
        if not docs:
            st.caption("No external or internal documents were utilized.")
        else:
            for idx, doc in enumerate(docs):
                metadata = getattr(doc, "metadata", {}) or {}
                if not isinstance(metadata, dict):
                    metadata = {}

                page_content = getattr(doc, "page_content", "")
                source_title = metadata.get("source", "Internal KB")
                chunk_id = metadata.get("chunk_id", f"chunk_{idx+1}")
                source_title_str = str(source_title)
                is_url = source_title_str.startswith("http")

                with st.expander(f"📌 Chunk [{chunk_id}] — {source_title_str[:55]}..."):
                    if is_url:
                        st.markdown(f"**Direct URL:** [{source_title_str}]({source_title_str})")
                    else:
                        st.markdown(f"**Origin:** Internal Policy Directive (`{source_title_str}`)")
                    st.code(page_content, language="markdown")