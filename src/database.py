import os 
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import config
from telemetry import logger

class KnowledgeBaseManager:
    """Handles document chunking, metadata injection, and Qdrant indexing."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        # Initializing Local Embeddings
        logger.info(f"Loading local embedding model: {config.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name= config.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None

    def load_and_chunk(self) -> List[Document]:
        """Reads policy text and slices it with metadata preservation."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Knowledge document not found at: {self.file_path}")

        # Load the raw document
        loader = TextLoader(self.file_path, encoding="utf-8")
        raw_documents = loader.load()

        # Split into 600 character chunks with a 100 character overlap
        splitter= RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\nSection ","\n\n"," ",""]
        )
        chunks=splitter.split_documents(raw_documents)

        # Inject unique chunk IDs into metadata for traceability
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"policy_doc_chunk_{index + 1}"
            chunk.metadata["source"] = os.path.basename(self.file_path)

        logger.info(f"Chunked document into {len(chunks)} traceable segments.")
        return chunks

    def build_vector_store(self) -> QdrantVectorStore:
        """embeds chunks and creates an in memory Qdrant vector store."""
        chunks = self.load_and_chunk()

        # Create an In memory Qdrant Client
        client = QdrantClient(location=":memory:")

        # 384 dimension matches all-MiniLM-L6_v2 vector output size
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )        

        # Wrap in Langchains vector store abstraction
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embedding=self.embeddings
        )

        # Add the documents and calculate vectors
        self.vector_store.add_documents(chunks)
        logger.info("Successfully indexed knowledge base into in memory Qdrant.")
        return self.vector_store