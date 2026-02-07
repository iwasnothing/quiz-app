# QuizGenius — AI-Powered Quiz Generator

An intelligent quiz generation platform that extracts knowledge from educational documents and creates customizable quizzes using AI. Built with FastAPI, Next.js, and LangChain.

## 🎯 Features

- **Document Ingestion**: Upload and process educational documents (PDF, DOCX, PPTX, etc.)
- **Topic Extraction**: Automatically discover topics and concepts from uploaded documents
- **AI-Powered Quiz Generation**: Generate quizzes with configurable:
  - Question count
  - Complexity distribution (Easy/Medium/Hard)
  - Format distribution (Multiple Choice/Short Answer)
- **Streaming Generation**: Real-time question generation with Server-Sent Events (SSE)
- **Interactive Canvas UI**: Drag-and-drop interface for quiz creation and editing
- **Question Refinement**: Edit, refine, and reroll questions using AI
- **Vector Search**: Semantic search using FAISS (with planned migration to PostgreSQL + pgvector)

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────────────────────┐
│   Next.js   │────▶│   FastAPI   │────▶│  SQLite (metadata + chunks)      │
│  Frontend   │     │   Backend   │     │  FAISS  (vector index)           │
└─────────────┘     └─────────────┘     └──────────────────────────────────┘
                            │                        │
                            │           ┌────────────┴──────────────┐
                            │           │  PostgreSQL + pgvector    │
                            │           │  (provisioned, not yet    │
                            │           │   used — future migration)│
                            │           └───────────────────────────┘
                            │
                            └──────────▶ LangChain/LLM
```

## 📋 Prerequisites

- **Docker** (with Docker Compose v2) and **NVIDIA Container Toolkit** (the backend image is based on `nvcr.io/nvidia/pytorch` and requests GPU access)
- An **LLM inference server** exposing an OpenAI-compatible `/v1` API (e.g. vLLM, TGI, Ollama, or the OpenAI API itself)
- An **embedding inference server** exposing an OpenAI-compatible `/v1` API (can be the same server or a separate one)
- **Make** (optional, for convenience commands)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd quiz-app
```

### 2. Configure Environment Variables

The backend reads its configuration from `backend/.env`. Copy the example below and edit values for your environment:

```bash
cp backend/.env.example backend/.env   # or create from scratch
```

**`backend/.env`** -- minimum required settings:

```bash
# ── LLM Server (Required) ─────────────────────────────────────────
# Point these to your OpenAI-compatible LLM inference endpoint.
LLM_MODEL=/models/Qwen/Qwen2.5-7B-Instruct        # model name served by your LLM server
LLM_BASE_URL=http://your-llm-host:8000/v1          # base URL (must end with /v1)
LLM_API_KEY=EMPTY                                   # API key (use "EMPTY" if auth is disabled)
LLM_TEMPERATURE=0.7

# ── Embedding Server (Required) ───────────────────────────────────
# Point these to your OpenAI-compatible embedding endpoint.
EMBEDDING_MODEL=/models/google/embeddinggemma-300m  # model name served by your embedding server
EMBEDDING_BASE_URL=http://your-emb-host:8001/v1    # base URL (must end with /v1)
EMBEDDING_API_KEY=EMPTY

# ── Data Paths ─────────────────────────────────────────────────────
DOCS_DIR=/path/to/your/documents                    # directory containing source documents
FAISS_DIR=./faiss_index                             # where FAISS index is stored (relative to /app in container)
SQLITE_DB_PATH=./chunks_metadata.db                 # SQLite metadata DB path

# ── PostgreSQL (not used yet — provisioned for future migration from SQLite/FAISS)
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mypassword
POSTGRES_DB=quiz_vector_db

# ── Optional ───────────────────────────────────────────────────────
LLM_MAX_RETRIES=5
QUIZ_CONCEPT_MIN_SIMILARITY=0.2
OPENAI_API_KEY=EMPTY                                # only needed if calling OpenAI directly
```

The frontend also needs to know where the backend API lives. This is set as a Docker Compose build argument and runtime environment variable. Override it from the shell before building if your host is not `localhost`:

```bash
export NEXT_PUBLIC_API_URL=http://your-host:8080
```

### 3. Build Docker Images

Using Make (recommended):

```bash
make build            # builds all images (frontend + backend)
```

Or using Docker Compose directly:

```bash
docker compose build
```

To rebuild from scratch without cache (e.g. after dependency changes):

```bash
make build-no-cache
# or
docker compose build --no-cache
```

You can also build images individually:

```bash
make build-backend    # docker build -t quiz-backend:latest ./backend
make build-frontend   # docker build -t quiz-frontend:latest ./frontend
```

### 4. Start the Application

```bash
make up               # starts all services in detached mode
# or
docker compose up -d
```

This launches four services:

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| **frontend** | `quiz-frontend` | `3000` | Next.js UI |
| **backend** | `quiz-backend` | `8080` | FastAPI + LangChain |
| **postgres** | `pgvector-arm` | `5432` | PostgreSQL with pgvector (not actively used yet -- provisioned for future migration) |
| **localtunnel** | `quiz-localtunnel` | -- | Optional tunnel for external access |

### 5. Access the Application

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8080
- **API Docs (Swagger)**: http://localhost:8080/docs
- **PostgreSQL**: `localhost:5432` (not actively used yet -- available for future migration)

### 6. Stop / Restart / Clean

```bash
make down             # stop all containers
make restart          # stop + start
make logs             # tail logs from all services
make clean            # stop containers AND delete volumes (WARNING: deletes all data)
make prune            # remove dangling Docker images to reclaim disk space
```

## 📚 Building the Knowledge Database from Teaching Materials

Before the app can generate quizzes, you must ingest your documents and build the normalized topic graph. This is a two-step process run via Python scripts inside the backend.

### Overview

```
Step 1: ingest.py          Step 2: build_graph.py
─────────────────          ──────────────────────
.docx / .pptx files        SQLite document_chunks
        │                          │
        ▼                          ▼
Convert to Markdown        Cluster similar topics
        │                  (word-overlap algorithm)
        ▼                          │
Chunk (2000 chars)                 ▼
        │                  normalized_topics
        ▼                  normalized_concepts
LLM: extract topic         topic_concept_edges
  + sub-concepts                   │
        │                          ▼
        ▼                  ┌──────────────────┐
LLM: resolve against       │  Ready for quiz  │
  existing entries         │    generation    │
        │                  └──────────────────┘
        ▼
Store in SQLite +
FAISS vector index
```

### Where to Put Your Documents

The ingestion scripts read documents from directories you specify. You have two options for making documents accessible to the backend:

#### Option A: Place documents inside the `backend/` folder (simplest for Docker)

Since `docker-compose.yml` mounts `./backend` into the container at `/app`, any folder you create under `backend/` is immediately visible inside the container:

```
quiz-app/
└── backend/
    ├── data/                          ← create this folder
    │   ├── Teaching-Materials/        ← put .docx / .pptx files here
    │   │   ├── chapter1.pptx
    │   │   ├── chapter2.docx
    │   │   └── ...
    │   └── Question-Bank/             ← put past exam papers here
    │       ├── exam2024.docx
    │       └── ...
    ├── main.py
    ├── ingest.py
    └── ...
```

Inside the container these become `/app/data/Teaching-Materials` and `/app/data/Question-Bank`.

#### Option B: Mount an external directory into the container

If your documents live elsewhere on the host (e.g. `/home/user/edu-docs/`), add a volume mount to the `backend` service in `docker-compose.yml`:

```yaml
backend:
  volumes:
    - ./backend:/app
    - /home/user/edu-docs:/data    # ← add this line
```

Inside the container the documents are then available at `/data/Teaching-Materials`, `/data/Question-Bank`, etc.

#### Running locally (no Docker)

When running without Docker, just use the absolute path on your machine directly:

```bash
ingest_documents('/home/user/edu-docs/Teaching-Materials', doc_type='teaching_material')
```

### Where the Database and Vector Index Are Stored

After ingestion, the following files are created. Because `./backend` is mounted at `/app` in the container, they appear in both locations simultaneously:

| File | Inside Container | On Host | Controlled By |
|------|-----------------|---------|---------------|
| SQLite metadata DB | `/app/chunks_metadata.db` | `backend/chunks_metadata.db` | `SQLITE_DB_PATH` in `.env` |
| FAISS vector index | `/app/faiss_index/` | `backend/faiss_index/` | `FAISS_DIR` in `.env` |
| Topic graph visualization | `/app/topic_graph.html` | `backend/topic_graph.html` | Argument to `visualize_graph()` |
| PostgreSQL data | `/var/lib/postgresql/data` | Docker volume `postgres_data` | Managed by Docker (not actively used yet) |

> **Tip**: Since the backend volume mount is bidirectional, the SQLite DB and FAISS index persist on your host at `backend/chunks_metadata.db` and `backend/faiss_index/` even if the container is stopped. You can back them up, copy them between machines, or inspect the SQLite DB directly with `sqlite3 backend/chunks_metadata.db`.

### Step 1: Ingest Documents (`ingest.py`)

The `ingest_documents()` function accepts a directory of office documents and a `doc_type` label. It runs through 8 internal steps:

1. Convert legacy `.doc`/`.ppt` files to `.docx`/`.pptx` (requires LibreOffice)
2. Convert all `.docx`/`.pptx` files to Markdown via MarkItDown
3. Load the resulting `.md` files
4. Chunk into ~2000-character segments with 500-character overlap
5. For each chunk, call the LLM to extract a topic and 3-7 sub-concepts
6. Resolve each extracted topic/concept against the existing database (LLM + BM25)
7. Insert all chunks with metadata into the SQLite `document_chunks` table
8. Build/update the FAISS vector index

There are two document types you should ingest:

| `doc_type` | Purpose | Used For |
|------------|---------|----------|
| `"teaching_material"` | Primary knowledge source (textbooks, lecture slides) | Context for generating quiz questions |
| `"question_bank"` | Historical exam papers, past quizzes | Few-shot examples and diversity enforcement |

**Running inside Docker** (recommended -- assuming documents are in `backend/data/`):

```bash
# Open a shell in the running backend container
make shell-backend

# Inside the container (/app is the working directory):
python -c "
from ingest import ingest_documents

# Ingest teaching materials (primary knowledge source)
ingest_documents('/app/data/Teaching-Materials', doc_type='teaching_material')

# Ingest question bank (historical questions for diversity)
ingest_documents('/app/data/Question-Bank', doc_type='question_bank')
"
```

If you mounted an external directory (Option B above), use that path instead:

```bash
python -c "
from ingest import ingest_documents

ingest_documents('/data/Teaching-Materials', doc_type='teaching_material')
ingest_documents('/data/Question-Bank', doc_type='question_bank')
"
```

**Running locally** (without Docker):

```bash
cd backend
# Ensure backend/.env is configured with LLM_BASE_URL, EMBEDDING_BASE_URL, etc.
python -c "
from ingest import ingest_documents

ingest_documents('/home/user/edu-docs/Teaching-Materials', doc_type='teaching_material')
ingest_documents('/home/user/edu-docs/Question-Bank', doc_type='question_bank')
"
```

Alternatively, edit the `if __name__ == "__main__"` block at the bottom of `backend/ingest.py` to point to your directories, then run:

```bash
cd backend
python ingest.py
```

**What it produces:**

| Output | Path (container) | Path (host) | Description |
|--------|-----------------|-------------|-------------|
| SQLite DB | `/app/chunks_metadata.db` | `backend/chunks_metadata.db` | `document_chunks` table -- one row per chunk with `topic_name`, `sub_concepts`, `chunk_text`, `source`, `doc_type` |
| FAISS index | `/app/faiss_index/` | `backend/faiss_index/` | Vector index for semantic search during quiz generation |

> **Note**: Ingestion calls the LLM twice per chunk (once for topic extraction, once for resolution), so it can take a while for large document sets. Progress is printed to the console.

### Step 2: Build the Normalized Topic Graph (`build_graph.py`)

After ingestion, run `build_graph.py` to cluster similar topics into normalized groups. This step uses a deterministic word-overlap algorithm (no LLM calls) and is fast. It reads from and writes to the same SQLite database produced by Step 1.

**Running inside Docker:**

```bash
make shell-backend

python -c "
from build_graph import build_graph, visualize_graph

# Build normalized topic tables
build_graph()

# (Optional) Generate an interactive HTML visualization of the topic graph
# Output: /app/topic_graph.html (= backend/topic_graph.html on host)
visualize_graph('topic_graph.html')
"
```

**Running locally:**

```bash
cd backend
python build_graph.py
```

**What it produces (new tables in `chunks_metadata.db`):**

| Table | Contents |
|-------|----------|
| `normalized_topics` | Maps each original topic to its normalized cluster name |
| `normalized_concepts` | Identity mapping (concepts are not further merged) |
| `topic_concept_edges` | Links `(normalized_topic, normalized_concept, original_topic)` |

It also optionally generates `topic_graph.html` (at `backend/topic_graph.html` on the host) -- an interactive PyVis visualization showing normalized topics and their original topic members.

### Full Example: End-to-End Database Build

```bash
# 1. Place your documents under backend/data/
mkdir -p backend/data/Teaching-Materials backend/data/Question-Bank
cp /path/to/your/slides/*.pptx backend/data/Teaching-Materials/
cp /path/to/your/exams/*.docx  backend/data/Question-Bank/

# 2. Start services
make up

# 3. Open a shell in the backend container
make shell-backend

# 4. Inside the container:
python -c "
from ingest import ingest_documents
from build_graph import build_graph, visualize_graph

# ── Step 1: Ingest teaching materials ──
ingest_documents('/app/data/Teaching-Materials', doc_type='teaching_material')

# ── Step 2: Ingest question bank (optional but recommended) ──
ingest_documents('/app/data/Question-Bank', doc_type='question_bank')

# ── Step 3: Build normalized topic graph ──
build_graph()

# ── Step 4: (Optional) Visualize the topic graph ──
visualize_graph('topic_graph.html')

print('Database build complete!')
"
```

After this, the `/topics` API endpoint will return the normalized topics and the frontend will be ready for quiz generation. You can verify by visiting http://localhost:8080/topics.

### Re-Ingesting or Adding More Documents

You can run `ingest_documents()` multiple times with different directories. Each run:
- **Appends** new chunks to the existing `document_chunks` table (does not delete existing data)
- **Adds** new documents to the existing FAISS index
- **Resolves** new topics/concepts against all previously ingested ones

After adding more documents, always **re-run `build_graph()`** to rebuild the normalized topic tables from scratch (it truncates and rebuilds each time).

To start completely fresh, delete the database and index files:

```bash
# On the host:
rm -f backend/chunks_metadata.db
rm -rf backend/faiss_index/
```

---

## 📖 Usage

### Upload Documents

1. Use the `/ingest` endpoint to upload educational documents:

```bash
curl -X POST "http://localhost:8080/ingest" \
  -H "accept: application/json" \
  -F "file=@your_document.pdf"
```

2. The system will:
   - Extract text from the document
   - Chunk the content into ~2000-character segments
   - Extract topics and sub-concepts via LLM
   - Resolve against existing topics/concepts to avoid duplicates
   - Generate embeddings and store in FAISS + SQLite

### Generate Quizzes

1. **Via Frontend UI**:
   - Open http://localhost:3000
   - Select topics from the Quiz DNA sidebar
   - Configure question count, complexity, and format ratios
   - Click "Generate Quiz"
   - Watch questions stream in real-time

2. **Via API** (streaming Server-Sent Events):

```bash
curl -N -X POST "http://localhost:8080/generate-quiz" \
  -H "Content-Type: application/json" \
  -d '{
    "topics": ["Photosynthesis", "Cell Biology"],
    "questionCount": 10,
    "complexityHard": 30,
    "formatMC": 70
  }'
```

### Available Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Upload and process documents |
| `GET` | `/topics` | Get all normalized topics with sub-concepts |
| `GET` | `/topics/names` | Get all unique topic names |
| `GET` | `/verify-topic?topic=...` | Debug: check if a topic maps to chunks |
| `POST` | `/generate-quiz` | Generate quiz (streaming SSE) |
| `POST` | `/generate-quiz-legacy` | Legacy non-streaming quiz generation |
| `POST` | `/refine-question` | Accept an edited question |
| `POST` | `/refine-question-advanced` | Refine a question with LLM + teacher feedback |
| `POST` | `/reroll-question` | Regenerate a specific question via AI |

## ⚙️ Configuration

### Environment Variables Reference

All backend variables are read from `backend/.env` (loaded via `python-dotenv`) and can be overridden in `docker-compose.yml`.

#### LLM Configuration (Required)

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `LLM_MODEL` | Model name on your LLM server | `/models/Qwen/Qwen3-Omni-30B-A3B-Instruct` | `gpt-4o`, `/models/Qwen/Qwen2.5-7B-Instruct` |
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `http://spark-cda3.local:8000/v1` | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API key for the LLM server | `EMPTY` | `sk-...` |
| `LLM_TEMPERATURE` | Sampling temperature (0.0-1.0) | `0.7` | `0.5` |
| `LLM_MAX_RETRIES` | Max retries on LLM failure | `5` | `3` |

#### Embedding Configuration (Required)

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `EMBEDDING_MODEL` | Model name on your embedding server | `/models/google/embeddinggemma-300m` | `text-embedding-3-small` |
| `EMBEDDING_BASE_URL` | OpenAI-compatible embedding API URL | `http://spark-cda3.local:8001/v1` | `https://api.openai.com/v1` |
| `EMBEDDING_API_KEY` | API key for the embedding server | `EMPTY` | `sk-...` |

#### Data Storage

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCS_DIR` | Source documents directory (for batch ingestion) | `/Users/.../edu-doc/science/ch8` |
| `FAISS_DIR` | FAISS vector index directory | `./faiss_index` (resolves to `/app/faiss_index` in Docker) |
| `SQLITE_DB_PATH` | SQLite metadata database path | `./chunks_metadata.db` (resolves to `/app/chunks_metadata.db` in Docker) |

#### PostgreSQL (Future -- Not Actively Used)

> **Note**: PostgreSQL with pgvector is provisioned in `docker-compose.yml` but is **not actively used** by the application today. All metadata is stored in SQLite and all vector search uses FAISS. These variables exist in preparation for a future migration from SQLite + FAISS to PostgreSQL + pgvector. You can leave them at their defaults.

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | PostgreSQL user | `myuser` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `mypassword` |
| `POSTGRES_DB` | Database name | `quiz_vector_db` |
| `POSTGRES_HOST` | Hostname (auto-set to `postgres` in Docker) | `localhost` |
| `POSTGRES_PORT` | Port | `5432` |
| `PG_CONNECTION_STRING` | Full connection string (optional; if unset, constructed from components above) | -- |

#### Quiz Generation Tuning

| Variable | Description | Default |
|----------|-------------|---------|
| `QUIZ_CONCEPT_MIN_SIMILARITY` | Minimum cosine similarity between topic and concept embeddings for hallucination filtering (0.0-1.0) | `0.2` |

#### Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL as seen by the **browser** (must be set at Docker **build** time) | `http://spark-cda3.local:8080` |
| `INTERNAL_API_URL` | Backend API URL for server-side rendering inside Docker network | `http://spark-cda3.local:8080` |

> **Note**: `NEXT_PUBLIC_API_URL` is baked into the frontend at build time (Next.js requirement). If you change it, you must rebuild the frontend image.

#### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (only needed if using OpenAI directly instead of a self-hosted LLM) | `EMPTY` |

### Docker Compose Configuration

The `docker-compose.yml` defines four services:

- **frontend** -- Next.js app, port 3000. Receives `NEXT_PUBLIC_API_URL` as a build arg.
- **backend** -- FastAPI app on NVIDIA PyTorch base image, port 8080. Mounts `./backend` as a volume for live-reload during development. Reads `backend/.env` via `env_file`.
- **postgres** -- `pgvector/pgvector:pg16`, port 5432. Data persisted in a Docker volume `postgres_data`. **Not actively used yet** -- provisioned for a planned future migration from SQLite + FAISS to PostgreSQL + pgvector.
- **localtunnel** -- Optional service that exposes the frontend via a public URL for demo purposes.

The backend container requests all available NVIDIA GPUs via `deploy.resources.reservations`. If you do not have GPUs, remove the `deploy` block from `docker-compose.yml`.

Edit `docker-compose.yml` to customize:
- Port mappings (e.g. change `8080:8080` if port is already in use)
- `extra_hosts` entries (the default maps `spark-cda3.local` to the Docker host)
- Volume mounts
- GPU reservations

## 🛠️ Development

### Project Structure

```
quiz-app/
├── backend/              # FastAPI backend
│   ├── main.py          # API endpoints
│   ├── ingest.py        # Document ingestion pipeline
│   ├── build_graph.py   # Post-ingestion topic normalization
│   ├── generator.py     # Quiz generation logic
│   ├── database.py      # Database operations (SQLite + FAISS)
│   ├── models.py        # Pydantic models
│   ├── llm_prompts.py   # LLM prompts
│   ├── config.py        # Configuration
│   ├── .env             # Environment variables
│   ├── chunks_metadata.db  # SQLite DB (created after ingestion)
│   └── faiss_index/     # FAISS vector index (created after ingestion)
├── frontend/            # Next.js frontend
│   ├── app/            # Next.js app directory
│   ├── components/     # React components
│   └── lib/            # Utilities and API client
├── postgres/            # PostgreSQL init scripts (future migration)
├── tests/               # Test files
├── docker-compose.yml   # Docker orchestration
└── Makefile            # Build automation
```

### Running Locally (Without Docker)

**Backend**:

```bash
cd backend
pip install -r requirements.txt
# Ensure backend/.env is configured (see Environment Variables above)
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

**Frontend**:

```bash
cd frontend
# Set the backend URL for the browser
export NEXT_PUBLIC_API_URL=http://localhost:8080
npm install
npm run dev
```

### Make Commands

```bash
make help              # Show all available commands
make build             # Build all Docker images (using cache)
make build-no-cache    # Rebuild all images from scratch
make build-backend     # Build only the backend image
make build-frontend    # Build only the frontend image
make up                # Start all services (detached)
make down              # Stop and remove containers
make restart           # Restart all services (down + up)
make logs              # Tail logs from all services
make clean             # Stop containers + delete volumes (WARNING: resets DB)
make prune             # Remove dangling Docker images to free disk
make shell-backend     # Open bash shell in backend container
make shell-frontend    # Open shell in frontend container
make shell-postgres    # Open psql shell in PostgreSQL container
```

## 🧠 Technical Deep-Dive: Topic, Concept, and Quiz Pipeline

This section describes the core algorithms behind topic/concept extraction, deduplication, normalization, and quiz generation.

---

### 1. Generating Topics and Concepts from Document Chunks

**Source**: `ingest.py` -> `extract_topic_metadata()`, `llm_prompts.py` -> `TOPIC_EXTRACTION_PROMPT`

During document ingestion, the system converts uploaded documents (DOCX, PPTX, etc.) into Markdown, then splits them into chunks of ~2000 characters with 500-character overlap using `RecursiveCharacterTextSplitter`. Each chunk is sent to the LLM with the **Topic Extraction Prompt**, which instructs the model to:

1. Identify **one main topic** (`topic_name`) for the chunk.
2. Extract **3-7 key sub-concepts** (`sub_concepts`) covered in that chunk.

The LLM returns a JSON object:

```json
{
  "topic_name": "Photosynthesis in Plants",
  "sub_concepts": ["light reactions", "Calvin cycle", "chlorophyll function", "ATP synthesis"]
}
```

This is done per-chunk, so a single document may yield many (topic, concepts) pairs, one per chunk.

---

### 2. Resolving Topics and Concepts to Avoid Proliferation

**Source**: `ingest.py` -> `resolve_topic_and_concepts()`, `llm_prompts.py` -> `NODE_RESOLUTION_PROMPT`

After extracting a raw topic and concepts from a chunk, the system resolves them against all previously stored topics and concepts in the database. This prevents the accumulation of near-duplicate entries (e.g., "Plant Photosynthesis" vs. "Photosynthesis in Plants").

The resolution process works as follows:

1. **Retrieve existing topics and concepts** from the SQLite `document_chunks` table.
2. **BM25 similarity search**: Since the existing topic/concept lists can grow large, a BM25 search selects the top 50 most similar existing topics and concepts to include in the prompt. This prevents token overflow while keeping the most relevant candidates.
3. **LLM-based resolution**: The extracted topic and concepts, together with the BM25-selected existing entries, are sent to the LLM with the **Node Resolution Prompt**. The LLM is instructed to:
   - If the extracted topic is similar to an existing topic, return the **exact existing topic name**.
   - If the extracted topic is genuinely new, return it as-is.
   - Apply the same logic to each extracted concept.
4. **Incremental update**: After resolution, newly resolved topics/concepts are appended to the in-memory list so that subsequent chunks in the same ingestion batch can also match against them.

This two-phase approach (BM25 pre-filtering + LLM judgment) keeps the topic/concept vocabulary compact while avoiding expensive full-list LLM calls.

---

### 3. Assigning Normalized Names (Post-Ingestion Graph Build)

**Source**: `build_graph.py` -> `_cluster_topics_by_word_overlap()`, `build_graph()`

After all documents are ingested, a separate **graph-building step** further normalizes topics using a deterministic word-overlap algorithm (no LLM involved):

#### Clustering Algorithm

1. **Tokenize** each topic into a set of meaningful words (lowercase, length > 1, stopwords removed). A curated stopword list filters out generic terms like "system", "analysis", "process", etc.
2. **Symmetric overlap test**: Two topics are grouped together only if **both** conditions hold:
   - Both topics share > **70%** of their (non-stopword) words with the other (`OVERLAP_THRESHOLD = 0.7`).
   - They share at least **3** meaningful words in common (`MIN_SHARED_WORDS = 3`).
   
   The symmetric requirement (both sides must exceed 70%) prevents transitivity chains where "A similar to B" and "B similar to C" would incorrectly merge A and C.
3. **Union-Find clustering**: A Union-Find data structure groups topics that pass the symmetric overlap test into clusters.

#### Normalized Name Assignment

For each cluster:

- **Single-member cluster**: The topic keeps its original name.
- **Small cluster** (<=12 members): The normalized name is built from the **top 10 most common words** (appearing in >= 2 members), joined by spaces.
- **Large cluster** (>12 members): The **shortest original topic name** in the cluster is used as the normalized name (to avoid long meaningless concatenations).

#### Database Tables Created

| Table | Purpose |
|-------|---------|
| `normalized_topics` | Maps each `original_topic` -> `normalized_topic` |
| `normalized_concepts` | Identity map (concepts are not further normalized) |
| `topic_concept_edges` | Links `(normalized_topic, normalized_concept, original_topic)` for provenance tracking |

These tables are used at quiz-generation time to resolve teacher-selected topics back to the original chunk-level topic names for retrieval.

---

### 4. Quiz Generation from Selected Topics

**Source**: `generator.py` -> `generate_quiz_chain_streaming()`

When the teacher selects topics and requests a quiz, the streaming pipeline works as follows:

#### Question Distribution Across Topics

Given **N** total questions and **M** topics:
- Each topic gets **K = N // M** questions.
- The first **R = N % M** topics each get **K + 1** questions (remainder distribution).

#### Per-Topic Generation Pipeline

For each topic, the system executes:

1. **Topic resolution**: The teacher-selected topic is resolved to its `normalized_topic` via the `normalized_topics` table. Then all `original_topic` names that map to that normalized topic are retrieved (these are the actual `topic_name` values stored in `document_chunks`).

2. **Chunk retrieval** (context gathering):
   - **Vector search (FAISS)**: Fetch top chunks using similarity search, filtered by the original topic names. A round-robin strategy cycles through original topics to ensure even representation.
   - **SQLite fallback**: If vector search returns no results, fall back to querying `document_chunks` directly for `teaching_material` chunks matching the original topics.

3. **Quiz concept extraction**: The retrieved chunks are sent to the LLM with the `GENERATE_QUIZ_CONCEPTS_PROMPT`, which extracts **K distinct, testable quiz concepts** (50% more than needed to allow for validation failures). Each concept is a short phrase describing a specific testable fact or idea.

4. **Hallucination validation**: Each generated quiz concept is validated by computing the **cosine similarity** between the topic embedding and the concept embedding. If similarity falls below the threshold (`QUIZ_CONCEPT_MIN_SIMILARITY`, default 0.2), the concept is discarded as a likely hallucination. Chinese text is first translated to English for meaningful embedding comparison.

5. **Blueprint construction**: Valid concepts are assigned difficulty levels and question formats according to the requested ratios:
   - **Difficulty**: `complexity_hard`% Hard, remainder split equally between Easy and Medium.
   - **Format**: `format_mc`% Multiple Choice (MCQ), remainder as Short Answer.

6. **Question generation**: For each blueprint row, the LLM generates a single question using the `GENERATE_SINGLE_PROMPT`, with:
   - The retrieved chunks as reference context (not shown to students).
   - Historical question-bank examples for diversity awareness.
   - A diversity instruction focusing the question on its specific assigned concept.

---

### 5. Ensuring Distinct and Well-Distributed Questions

The system uses multiple strategies at different levels to avoid duplicate or overlapping questions:

#### A. Concept-Level Diversification

- Each question in a quiz is assigned a **distinct concept** from the topic's concept pool. Concepts are sampled without replacement first; repeats are only allowed when the number of questions exceeds available concepts.
- The `GENERATE_QUIZ_CONCEPTS_PROMPT` explicitly instructs the LLM to produce "distinct, unique, and testable" concepts with no overlap.

#### B. Prompt-Level Diversity Instructions

- The generation prompt includes a `diversity_instruction` that tells the LLM which concepts are already covered by other questions in the quiz, and instructs it to "focus ONLY on concept X" and "must NOT overlap" with those.
- The `other_concepts_in_quiz` parameter accumulates concepts used so far in the batch, and is included in each subsequent prompt.

#### C. Historical Question Awareness

- Up to 10 historical questions from the `question_bank` (if ingested) are included in the prompt as examples of what NOT to repeat. The prompt explicitly states: "Ensure all generated questions are DISTINCT and DIVERSE from the historical questions."

#### D. Difficulty and Format Distribution

Questions are evenly distributed across difficulty levels and formats using percentage-based allocation:

| Parameter | Controls | Example |
|-----------|----------|---------|
| `complexity_hard` | % of Hard questions | 30% Hard -> 35% Medium, 35% Easy |
| `format_mc` | % of MCQ questions | 70% MCQ -> 30% Short Answer |

These ratios are applied **per-topic** so each topic gets a proportional mix of difficulty levels and formats.

#### E. Hallucination Filtering

- Quiz concepts validated via embedding similarity (topic vs. concept) catch concepts that the LLM may have invented without grounding in the source material.
- The retry loop (up to 12 retries per topic) generates additional concepts to replace any that fail validation.

#### F. Forbidden Phrase Stripping

- Since quizzes are closed-book, any references to "according to the article/context/text" are automatically stripped from question text using regex patterns. This ensures questions are self-contained.

---

### Pipeline Summary

```
Documents
    │
    ▼
[Convert to Markdown] ──▶ [Chunk (2000 chars)] ──▶ [LLM: Extract Topic + Concepts]
                                                          │
                                                          ▼
                                                    [LLM: Resolve Against Existing]
                                                          │
                                                          ▼
                                                    [Store in SQLite + FAISS]
                                                          │
                                                          ▼
                                                    [Build Normalized Graph]
                                                     (word-overlap clustering)
                                                          │
                                                          ▼
                                             ┌─────────────────────────┐
                                             │   normalized_topics     │
                                             │   normalized_concepts   │
                                             │   topic_concept_edges   │
                                             └─────────────────────────┘
                                                          │
                                              Teacher selects topics
                                                          │
                                                          ▼
                                             [Resolve to original_topics]
                                                          │
                                                          ▼
                                             [FAISS vector search for chunks]
                                                          │
                                                          ▼
                                             [LLM: Extract quiz concepts]
                                                          │
                                                          ▼
                                             [Validate via embedding similarity]
                                                          │
                                                          ▼
                                             [Build blueprint: difficulty + format]
                                                          │
                                                          ▼
                                             [LLM: Generate question per concept]
                                                          │
                                                          ▼
                                             [Strip forbidden phrases, stream to UI]
```

---

## 🧪 Testing

Run tests:
```bash
cd backend
python -m pytest tests/
```

## 📦 Dependencies

### Backend
- FastAPI - Web framework
- LangChain - LLM orchestration
- SQLite - Metadata and chunk storage
- FAISS - Vector search index
- PostgreSQL + pgvector - Provisioned for future migration (not actively used)
- Uvicorn - ASGI server

### Frontend
- Next.js 16 - React framework
- TypeScript - Type safety
- Tailwind CSS - Styling

## 🔒 Security Notes

- In production, update CORS settings in `backend/main.py` to restrict origins
- Use environment variables for sensitive credentials
- Consider using secrets management for API keys
- Review and update `.gitignore` to exclude sensitive files

## 🐛 Troubleshooting

### LLM / Embedding Connection Issues
- The backend must be able to reach `LLM_BASE_URL` and `EMBEDDING_BASE_URL` from inside the Docker container.
- If your LLM/embedding servers run on the Docker host, use the `extra_hosts` mapping in `docker-compose.yml` (default maps `spark-cda3.local` to host-gateway). Alternatively, set the URLs to `http://host.docker.internal:<port>/v1`.
- Check connectivity: `make shell-backend` then `curl $LLM_BASE_URL/models`.

### Database / Vector Index Issues
- The app currently uses **SQLite** (`backend/chunks_metadata.db`) and **FAISS** (`backend/faiss_index/`) -- not PostgreSQL.
- If the backend cannot find chunks or topics, verify these files exist and were populated by `ingest.py` and `build_graph.py`.
- You can inspect the SQLite DB directly: `sqlite3 backend/chunks_metadata.db ".tables"` and `sqlite3 backend/chunks_metadata.db "SELECT COUNT(*) FROM document_chunks"`.
- The PostgreSQL container runs but is not actively used. It is provisioned for a future migration.

### Port Conflicts
- Change port mappings in `docker-compose.yml` if ports `3000`, `8080`, or `5432` are already in use.

### GPU / NVIDIA Issues
- The backend Dockerfile uses `nvcr.io/nvidia/pytorch` and requests GPU access. If you do not have GPUs, remove the `deploy.resources.reservations` block from `docker-compose.yml`.
- Ensure the NVIDIA Container Toolkit is installed: `nvidia-smi` should work inside Docker.

### Frontend Shows "Cannot connect to API"
- Verify `NEXT_PUBLIC_API_URL` was set correctly **before** building the frontend image. This value is baked in at build time.
- Rebuild: `docker compose build frontend` after changing it.

### Build Issues
- Try `make build-no-cache` to rebuild from scratch
- Check Docker logs: `make logs`

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]
