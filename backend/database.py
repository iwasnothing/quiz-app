import os
import sqlite3
import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import settings
import requests

# --- Embeddings ---
# Initialize embeddings with error handling
_embeddings = None

def get_embeddings():
    """Get embeddings instance, with lazy initialization and connection testing."""
    global _embeddings
    if _embeddings is None:
        try:
            # Test connection before initializing
            test_url = settings.EMBEDDING_BASE_URL.replace('/v1', '/health') if '/v1' in settings.EMBEDDING_BASE_URL else f"{settings.EMBEDDING_BASE_URL}/health"
            try:
                response = requests.get(test_url, timeout=2)
                if response.status_code == 200:
                    print(f"✓ Embedding service is reachable at {settings.EMBEDDING_BASE_URL}")
            except requests.exceptions.RequestException as e:
                print(f"⚠ Warning: Cannot reach embedding service at {settings.EMBEDDING_BASE_URL}: {e}")
                print(f"  Attempting to initialize anyway - connection will be tested on first use")
            
            _embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=settings.EMBEDDING_BASE_URL,
                api_key=settings.EMBEDDING_API_KEY,
            )
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize embeddings: {e}")
            raise
    return _embeddings

# For backward compatibility, create embeddings at module level but with error handling
try:
    embeddings = get_embeddings()
except Exception as e:
    print(f"⚠ Warning: Embeddings initialization failed: {e}")
    # Create a placeholder that will raise errors when used
    embeddings = None

# --- Vector Store (FAISS) ---
def get_faiss_vectorstore(docs=None):
    """Get or create FAISS vectorstore."""
    faiss_path = Path(settings.FAISS_DIR)
    
    # Get embeddings instance (with error handling)
    emb = get_embeddings()
    if emb is None:
        raise ValueError("Embeddings not available. Cannot load FAISS vectorstore.")
    
    if faiss_path.exists():
        print(f"Loading existing FAISS index from {faiss_path}")
        try:
            vectorstore = FAISS.load_local(
                str(faiss_path),
                emb,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"⚠ Warning: Failed to load FAISS index: {e}")
            raise
        # Add new documents if provided
        if docs and len(docs) > 0:
            print(f"Adding {len(docs)} new documents to existing FAISS index...")
            vectorstore.add_documents(docs)
    else:
        if not docs or len(docs) == 0:
            raise ValueError("Cannot build FAISS index - no document chunks provided")
        
        print("Building new FAISS index...")
        vectorstore = FAISS.from_documents(docs, embedding=emb)
        faiss_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(faiss_path))
        print(f"Saved new FAISS index to {faiss_path}")
    
    return vectorstore

def save_faiss_index(vectorstore, faiss_path: str = None):
    """Explicitly save FAISS index to disk."""
    if faiss_path is None:
        faiss_path = settings.FAISS_DIR
    try:
        path = Path(faiss_path)
        path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(path))
        print(f"✓ FAISS index saved to {faiss_path}")
    except Exception as e:
        print(f"Warning: Failed to save FAISS index: {e}")


# --- SQLite Database Operations ---
def init_sqlite_db(db_path: str = None):
    """Initialize SQLite database with chunks table."""
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table for storing document chunks with metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_name TEXT NOT NULL,
            sub_concepts TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            source TEXT,
            doc_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index on topic_name for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_topic_name ON document_chunks(topic_name)
    """)
    
    # Create index on source for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_source ON document_chunks(source)
    """)
    
    # Create index on doc_type for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_doc_type ON document_chunks(doc_type)
    """)
    
    # Add doc_type column to existing tables if it doesn't exist (for migration)
    cursor.execute("""
        PRAGMA table_info(document_chunks)
    """)
    columns = [col[1] for col in cursor.fetchall()]
    if 'doc_type' not in columns:
        cursor.execute("""
            ALTER TABLE document_chunks ADD COLUMN doc_type TEXT
        """)
        print("Added doc_type column to existing table")
    
    conn.commit()
    conn.close()
    print(f"SQLite database initialized at {db_path}")

def insert_chunks_to_sqlite(chunks, db_path: str = None):
    """Insert document chunks with metadata into SQLite database. Commits after each insert."""
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    inserted_count = 0
    
    for chunk in chunks:
        topic_name = chunk.metadata.get("topic_name", "Unknown")
        sub_concepts = chunk.metadata.get("sub_concepts", [])
        chunk_text = chunk.page_content
        source = chunk.metadata.get("source", "Unknown")
        doc_type = chunk.metadata.get("doc_type", None)
        
        # Convert sub_concepts list to JSON string
        sub_concepts_json = json.dumps(sub_concepts)
        
        try:
            cursor.execute("""
                INSERT INTO document_chunks (topic_name, sub_concepts, chunk_text, source, doc_type)
                VALUES (?, ?, ?, ?, ?)
            """, (topic_name, sub_concepts_json, chunk_text, source, doc_type))
            # Commit after each insert to ensure data is saved immediately
            conn.commit()
            inserted_count += 1
        except Exception as e:
            print(f"Warning: Failed to insert chunk: {e}")
            # Rollback on error to maintain consistency
            conn.rollback()
            continue
    
    conn.close()
    print(f"Inserted {inserted_count} chunks into SQLite database (saved after each insert)")

def query_all_topics(db_path: str = None):
    """Query all unique topics from the database. Returns list of topic names."""
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT topic_name
        FROM document_chunks
        ORDER BY topic_name
    """)
    
    results = cursor.fetchall()
    conn.close()
    return [topic[0] for topic in results]

def query_all_concepts(db_path: str = None):
    """Query all unique concepts from the database."""
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT sub_concepts
        FROM document_chunks
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    # Extract all concepts from JSON strings
    all_concepts = set()
    for (sub_concepts_json,) in results:
        try:
            concepts = json.loads(sub_concepts_json)
            all_concepts.update(concepts)
        except:
            continue
    
    return sorted(list(all_concepts))

def query_all_topics_with_counts(db_path: str = None):
    """Query all unique topics with their counts from the database."""
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT topic_name, COUNT(*) as count
        FROM document_chunks
        GROUP BY topic_name
        ORDER BY count DESC
    """)
    
    results = cursor.fetchall()
    conn.close()
    return results


def get_main_topics_from_sqlite(db_path: str = None):
    """
    Return main topics from SQLite for the /topics API.
    Each topic has topic_name and aggregated unique sub_concepts from all chunks.
    Returns list of dicts: [{"topic_name": str, "sub_concepts": list[str]}, ...]
    """
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH

    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT topic_name, sub_concepts
        FROM document_chunks
        ORDER BY topic_name
    """)
    rows = cursor.fetchall()
    conn.close()

    # Aggregate by topic_name: collect unique sub_concepts
    by_topic: dict[str, set[str]] = {}
    for topic_name, sub_concepts_json in rows:
        if topic_name not in by_topic:
            by_topic[topic_name] = set()
        try:
            concepts = json.loads(sub_concepts_json)
            if isinstance(concepts, list):
                by_topic[topic_name].update(c for c in concepts if isinstance(c, str))
        except (json.JSONDecodeError, TypeError):
            continue

    return [
        {"topic_name": name, "sub_concepts": sorted(concepts)}
        for name, concepts in sorted(by_topic.items())
    ]

def query_chunks_by_topic(db_path: str, topic_name: str):
    """Query all chunks for a specific topic."""
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, topic_name, sub_concepts, chunk_text, source, created_at
        FROM document_chunks
        WHERE topic_name = ?
        ORDER BY created_at
    """, (topic_name,))
    
    results = cursor.fetchall()
    conn.close()
    return results

def verify_sqlite_saved(db_path: str = None):
    """Verify SQLite database is properly saved."""
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            count = cursor.fetchone()[0]
            conn.close()
            print(f"✓ SQLite database verified: {count} chunks saved to {db_path}")
            return True
        else:
            print(f"Warning: SQLite database file not found at {db_path}")
            return False
    except Exception as e:
        print(f"Warning: Failed to verify SQLite database: {e}")
        return False

def get_concepts_from_question_bank(topics: list, db_path: str = None):
    """
    Get concepts from question_bank documents for given topics.
    Returns a list of concepts that are commonly tested in historical question banks.
    """
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create placeholders for IN clause
    placeholders = ','.join(['?'] * len(topics))
    
    cursor.execute(f"""
        SELECT sub_concepts
        FROM document_chunks
        WHERE doc_type = 'question_bank'
        AND topic_name IN ({placeholders})
    """, topics)
    
    results = cursor.fetchall()
    conn.close()
    
    # Extract all concepts from JSON strings and count frequency
    concept_counts = {}
    for (sub_concepts_json,) in results:
        try:
            concepts = json.loads(sub_concepts_json)
            for concept in concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        except:
            continue
    
    # Return concepts sorted by frequency (most common first)
    sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
    return [concept for concept, count in sorted_concepts]

def query_chunks_by_doc_type_and_topics(doc_type: str, topics: list, limit: int = None, db_path: str = None, allow_null: bool = False):
    """
    Query chunks by doc_type and topics from SQLite.
    Returns list of tuples: (id, topic_name, sub_concepts, chunk_text, source, doc_type, created_at)
    
    Args:
        doc_type: The doc_type to filter by
        topics: List of topic names
        limit: Optional limit on number of results
        db_path: Optional database path
        allow_null: If True, also include chunks where doc_type IS NULL
    """
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create placeholders for IN clause
    placeholders = ','.join(['?'] * len(topics))
    params = topics
    
    # Build WHERE clause: allow doc_type match or NULL if allow_null is True
    if allow_null:
        query = f"""
            SELECT id, topic_name, sub_concepts, chunk_text, source, doc_type, created_at
            FROM document_chunks
            WHERE (doc_type = ? OR doc_type IS NULL) AND topic_name IN ({placeholders})
            ORDER BY created_at
        """
        params = [doc_type] + topics
    else:
        query = f"""
            SELECT id, topic_name, sub_concepts, chunk_text, source, doc_type, created_at
            FROM document_chunks
            WHERE doc_type = ? AND topic_name IN ({placeholders})
            ORDER BY created_at
        """
        params = [doc_type] + topics
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    return results

def query_question_bank_chunks_by_topics(topics: list, limit: int = 10, db_path: str = None):
    """
    Query question_bank chunks for given topics. Used as few-shot examples.
    Returns list of chunk texts.
    """
    chunks = query_chunks_by_doc_type_and_topics('question_bank', topics, limit, db_path)
    return [chunk[3] for chunk in chunks]  # chunk_text is at index 3

def query_teaching_material_chunks_by_topics(topics: list, limit: int = None, db_path: str = None):
    """
    Query teaching_material chunks for given topics from SQLite.
    Also includes chunks where doc_type IS NULL.
    Returns list of tuples: (id, topic_name, sub_concepts, chunk_text, source, doc_type, created_at)
    """
    return query_chunks_by_doc_type_and_topics('teaching_material', topics, limit, db_path, allow_null=True)

def get_concepts_for_topic(topic: str, db_path: str = None):
    """
    Get all unique concepts (sub_concepts) for a specific topic from SQLite.
    Returns a list of unique concept strings.
    """
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT sub_concepts
        FROM document_chunks
        WHERE topic_name = ?
    """, (topic,))
    
    results = cursor.fetchall()
    conn.close()
    
    # Extract all concepts from JSON strings
    all_concepts = set()
    for (sub_concepts_json,) in results:
        try:
            concepts = json.loads(sub_concepts_json)
            if isinstance(concepts, list):
                all_concepts.update(c for c in concepts if isinstance(c, str) and c.strip())
        except (json.JSONDecodeError, TypeError):
            continue
    
    return sorted(list(all_concepts))


# --- Vector store for retrieval (used by generator) ---
# Lazy-load from FAISS when index exists. If not, run ingest first.
vector_store = None
try:
    if Path(settings.FAISS_DIR).exists():
        # Only load if embeddings are available
        if get_embeddings() is not None:
            vector_store = get_faiss_vectorstore()
        else:
            print("⚠ Warning: Embeddings not available, skipping vector store load")
    # else: vector_store stays None; generator will need to handle missing index
except Exception as e:
    import warnings
    print(f"⚠ Warning: FAISS vector store not loaded: {e}. Run ingest first.")
    warnings.warn(f"FAISS vector store not loaded: {e}. Run ingest first.")
