import os
from pathlib import Path
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

class Settings:
    """Application settings loaded from environment variables."""
    
    # Document processing
    DOCS_DIR: str = os.getenv("DOCS_DIR", "/Users/kahingleung/Downloads/edu-doc/science/ch8")
    FAISS_DIR: str = os.getenv("FAISS_DIR", "./faiss_index")
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./chunks_metadata.db")
    
    # Embeddings configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "/models/google/embeddinggemma-300m")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "http://spark-cda3.local:8001/v1")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "EMPTY")
    
    # LLM configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "/models/Qwen/Qwen3-Omni-30B-A3B-Instruct")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://spark-cda3.local:8000/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "EMPTY")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Database configuration (for PostgreSQL/Neo4j if needed)
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    PG_CONNECTION_STRING: str = os.getenv(
        "PG_CONNECTION_STRING", 
        "postgresql+psycopg://myuser:mypassword@localhost:5432/quiz_vector_db"
    )
    
    # OpenAI API (if using OpenAI directly)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Create a singleton instance
settings = Settings()
