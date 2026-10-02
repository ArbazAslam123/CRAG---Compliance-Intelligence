# 🛡️ CRAG-OS: Autonomous Enterprise Policy & Compliance Agent

An enterprise-grade **Corrective Retrieval-Augmented Generation (CRAG)** intelligence system built with **LangGraph**, **Google Gemini 2.5**, and **Qdrant**.

Unlike standard linear RAG pipelines that blindly trust vector database output, CRAG-OS operates as a self-reflective state machine. It evaluates retrieved chunks for relevance, discards ungrounded information, optimizes queries for search engines, and dynamically executes live web searches through **Tavily** whenever internal documentation is insufficient.

---

## 🏗️ System Architecture

```text
                         ┌─────────────────────────┐
                         │       User Query        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Retrieve From Qdrant   │
                         │    Dense Embeddings     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Document Relevance Node │
                         │  Gemini 2.5 Flash-Lite  │
                         └────────────┬────────────┘
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
       ┌──────────────────────┐                ┌──────────────────────┐
       │ All Chunks           │                │ Relevant Context     │
       │ Irrelevant           │                │ Found                │
       └──────────┬───────────┘                └──────────┬───────────┘
                  │                                       │
                  ▼                                       │
       ┌──────────────────────┐                           │
       │ Rewrite Search Query │                           │
       └──────────┬───────────┘                           │
                  │                                       │
                  ▼                                       │
       ┌──────────────────────┐                           │
       │ Fallback Web Search  │                           │
       │     Tavily API       │                           │
       └──────────┬───────────┘                            │
                  │                                       │
                  └───────────────────┬───────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Context Synthesis Node │
                         │    Gemini 2.5 Flash     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Grounding & Fact Audit  │
                         │  Hallucination Checker  │
                         └────────────┬────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                    ┌─────────┐            ┌──────────────────┐
                    │ Grounded│            │  Hallucinated /  │
                    │         │            │  Unsupported     │
                    └────┬────┘            └────────┬─────────┘
                         │                           │
                         ▼                           ▼
                  Final Response            Self-Correction
                                                 Loop
```

---

## ⚡ Core Technical Features

### 🧠 Stateful Graph Control

**LangGraph** orchestrates multi-step retrieval, conditional branching, and self-correction cycles using a typed `CRAGState` contract.

### 🎯 Deterministic Relevance Grading

Structured JSON-schema evaluations powered by **Pydantic** ensure consistent relevance grading and prevent conversational drift during routing decisions.

### 🌐 Autonomous Web Fallback

When internal enterprise policy documents cannot provide sufficient information, CRAG-OS automatically rewrites the search query and performs a live search using the **Tavily Search API**.

### 🔍 Deterministic Grounding Audit

After response generation, a dedicated grounding stage verifies that factual claims are supported by the retrieved evidence before the response reaches the user.

### 💾 Local In-Memory Vector Storage

Uses **Qdrant** with local **Sentence Transformers** embeddings (`all-MiniLM-L6-v2`) running on CPU, reducing infrastructure costs and avoiding vendor lock-in.

### 📊 Production-Grade Telemetry

Every graph node records execution timing and state mutations. Telemetry is persisted to:

```text
logs/crag.log
```

---

## 🛠️ Tech Stack

| Category                       | Technology                                     |
| ------------------------------ | ---------------------------------------------- |
| **Language**                   | Python 3.10+                                   |
| **State Machine & Agent Flow** | LangGraph, LangChain Core                      |
| **Reasoning Models**           | Google Gemini 2.5 Flash, Gemini 2.5 Flash-Lite |
| **Vector Database**            | Qdrant — In-Memory Engine                      |
| **Embedding Model**            | `sentence-transformers/all-MiniLM-L6-v2`       |
| **External Web Intelligence**  | Tavily AI Search API                           |
| **Frontend**                   | Streamlit — Cyber-Enterprise Glassmorphism UI  |
| **Data Validation**            | Pydantic v2                                    |

---

## 📁 Repository Structure

```text
crag-enterprise/
│
├── data/
│   └── enterprise_policy.txt
│       └── Master corporate policy knowledge base
│
├── logs/
│   └── crag.log
│       └── Telemetry and execution audit log
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │   └── Immutable application configuration
│   │
│   ├── database.py
│   │   └── Ingestion, chunking, and Qdrant vector indexing
│   │
│   ├── graders.py
│   │   └── Pydantic schemas and evaluation chains
│   │
│   ├── graph.py
│   │   └── LangGraph state machine assembly and routing
│   │
│   ├── nodes.py
│   │   └── Atomic node executors with performance hooks
│   │
│   ├── state.py
│   │   └── TypedDict state contract definition
│   │
│   ├── telemetry.py
│   │   └── Performance timing decorator and dual logger
│   │
│   └── tools.py
│       └── External Tavily search client
│
├── .env.example
├── .gitignore
├── app.py
│   └── Mission Control Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

# 🚀 Local Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/crag-enterprise-assistant.git
cd crag-enterprise-assistant
```

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows — Command Prompt / PowerShell

```bash
python -m venv venv
.\venv\Scripts\activate
```

## 3. Install Dependencies

Upgrade `pip` and install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

> **Important:** Never commit your `.env` file or API keys to GitHub.

## 5. Launch the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🌐 Deploying to Streamlit Community Cloud

## Step 1 — Prepare Your GitHub Repository

If Git is not already initialized:

```bash
git init
```

Stage the required project files:

```bash
git add src/ data/ app.py requirements.txt .gitignore README.md
```

Create the initial commit:

```bash
git commit -m "feat: complete production CRAG enterprise state machine"
```

Verify that sensitive files are not staged:

```bash
git status
```

Make sure `.env` and `logs/` are excluded through `.gitignore`.

---

## Step 2 — Create and Push the GitHub Repository

Create a new repository on GitHub, then connect your local project:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

Set the main branch:

```bash
git branch -M main
```

Push the project:

```bash
git push -u origin main
```

---

## Step 3 — Configure `requirements.txt`

Streamlit Community Cloud installs dependencies directly from `requirements.txt`.

Use compatible dependency ranges such as:

```text
streamlit>=1.38.0
langgraph>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-google-genai>=2.0.0
qdrant-client>=1.11.0
sentence-transformers>=3.0.0
tavily-python>=0.5.0
pydantic>=2.8.0
python-dotenv>=1.0.0
```

---

## Step 4 — Create the Streamlit Cloud App

Go to **Streamlit Community Cloud** and sign in with your GitHub account.

Create a new application and configure:

```text
Repository:    YOUR_USERNAME/YOUR_REPO_NAME
Branch:        main
Main file:     app.py
App URL:       crag-enterprise-ai.streamlit.app
```

The exact UI labels may vary slightly depending on the current Streamlit Cloud interface.

---

## Step 5 — Configure Streamlit Secrets

Since `.env` should remain excluded from Git, add your production API credentials through Streamlit's **Secrets** manager.

Open:

```text
App Settings → Secrets
```

Add:

```toml
GEMINI_API_KEY = "AIzaSy..."
TAVILY_API_KEY = "tvly-..."
```

Streamlit exposes these secrets through the environment, allowing your application to continue using:

```python
import os

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
```

No additional changes should be required in `src/config.py` if it already reads the credentials through `os.getenv()`.

---

## Step 6 — Deploy

Click **Deploy** and wait for Streamlit Community Cloud to install the dependencies and start the application.

Once deployment completes, your CRAG-OS application will be accessible through your Streamlit Cloud URL.

---

# 👤 Author

**Arbaz Aslam**
AI Engineer & Data Scientist

---

## ⭐ Project Highlights

CRAG-OS demonstrates a production-oriented approach to enterprise RAG by combining:

* **Corrective Retrieval-Augmented Generation**
* **Self-reflective LangGraph workflows**
* **Relevance grading**
* **Query rewriting**
* **Autonomous web fallback**
* **Grounding verification**
* **Hallucination detection**
* **Typed state management**
* **Pydantic structured outputs**
* **Local vector storage**
* **Execution telemetry**
* **Streamlit deployment**

The result is a RAG architecture designed to **retrieve, evaluate, correct, verify, and respond** rather than simply retrieve and generate.
