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
    # Normalize SQLITE_DB_PATH: convert relative paths to absolute
    # In Docker, resolve relative paths relative to /app (WORKDIR), not current working directory
    _sqlite_db_path = os.getenv("SQLITE_DB_PATH", "./chunks_metadata.db")
    if os.path.isabs(_sqlite_db_path):
        SQLITE_DB_PATH: str = _sqlite_db_path
    else:
        # For relative paths, resolve relative to /app if it exists (Docker), otherwise use current dir
        base_dir = "/app" if os.path.exists("/app") else os.getcwd()
        SQLITE_DB_PATH: str = os.path.join(base_dir, _sqlite_db_path)
    
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
    # If PG_CONNECTION_STRING is not set, construct it from individual components
    _pg_conn_str = os.getenv("PG_CONNECTION_STRING")
    if _pg_conn_str:
        PG_CONNECTION_STRING: str = _pg_conn_str
    else:
        # Construct from individual components
        pg_user = os.getenv("POSTGRES_USER", "myuser")
        pg_password = os.getenv("POSTGRES_PASSWORD", "mypassword")
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB", "quiz_vector_db")
        PG_CONNECTION_STRING: str = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    
    # OpenAI API (if using OpenAI directly)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Create a singleton instance
settings = Settings()
