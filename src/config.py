import os 
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # I assign Gemini 2.5 Flash for both Grading and generation
    # It is fast enough for quick classification and large enough for complex synthesis
    grader_model: str = "gemini-3.1-flash-lite"
    generator_model: str = "gemini-3.1-flash-lite"

    # Embedding model used to convert text chunks into dense vectors
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Internal name for the collection inside the Qdrant database
    collection_name: str = "crag_enterprise_kb"

# initializing a single global object config object to import across files
config = AppConfig()