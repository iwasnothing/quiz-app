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
- **Vector Search**: Semantic search using PostgreSQL with pgvector

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   Next.js   │────▶│   FastAPI   │────▶│  PostgreSQL  │
│  Frontend   │     │   Backend   │     │  (pgvector)  │
└─────────────┘     └─────────────┘     └──────────────┘
                            │
                            ├──────────▶ SQLite (Metadata)
                            │
                            └──────────▶ LangChain/LLM
```

## 📋 Prerequisites

- Docker and Docker Compose
- OpenAI API key (or compatible LLM API)
- Make (optional, for convenience commands)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd quiz-app
```

### 2. Set Environment Variables

Create a `.env` file in the root directory (or set environment variables):

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

For advanced configuration, see [Configuration](#configuration) section.

### 3. Start the Application

Using Make (recommended):
```bash
make build
make up
```

Or using Docker Compose directly:
```bash
docker compose build
docker compose up -d
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432

## 📖 Usage

### Upload Documents

1. Use the `/ingest` endpoint to upload educational documents:
   ```bash
   curl -X POST "http://localhost:8000/ingest" \
     -H "accept: application/json" \
     -F "file=@your_document.pdf"
   ```

2. The system will:
   - Extract text from the document
   - Chunk the content
   - Generate embeddings
   - Store in PostgreSQL (pgvector)
   - Extract topics and store metadata in SQLite

### Generate Quizzes

1. **Via Frontend UI**:
   - Open http://localhost:3000
   - Select topics from the Quiz DNA sidebar
   - Configure question count, complexity, and format ratios
   - Click "Generate Quiz"
   - Watch questions stream in real-time

2. **Via API**:
   ```bash
   curl -X POST "http://localhost:8000/generate-quiz" \
     -H "Content-Type: application/json" \
     -d '{
       "topics": ["Photosynthesis", "Cell Biology"],
       "questionCount": 10,
       "complexityHard": 30,
       "formatMC": 70
     }'
   ```

### Available Endpoints

- `POST /ingest` - Upload and process documents
- `GET /topics` - Get all topics with sub-concepts
- `GET /topics/names` - Get all topic names
- `POST /generate-quiz` - Generate quiz (streaming)
- `POST /generate-quiz-legacy` - Legacy quiz generation
- `POST /refine-question` - Refine a question
- `POST /refine-question-advanced` - Advanced refinement with LLM feedback
- `POST /reroll-question` - Regenerate a specific question

## ⚙️ Configuration

### Environment Variables

The application can be configured via environment variables. Key settings:

**Backend (`backend/config.py`)**:
- `OPENAI_API_KEY` - OpenAI API key (required)
- `NEO4J_URI` - Neo4j connection string (optional)
- `PG_CONNECTION_STRING` - PostgreSQL connection string
- `SQLITE_DB_PATH` - Path to SQLite metadata database
- `FAISS_DIR` - Directory for FAISS indexes (if using)
- `LLM_BASE_URL` - Custom LLM API endpoint
- `EMBEDDING_BASE_URL` - Custom embedding API endpoint

**Frontend**:
- `NEXT_PUBLIC_API_URL` - Backend API URL (default: http://localhost:8000)
- `INTERNAL_API_URL` - Internal Docker network API URL

### Docker Compose Configuration

Edit `docker-compose.yml` to customize:
- Port mappings
- Database credentials
- Volume mounts
- Environment variables

## 🛠️ Development

### Project Structure

```
quiz-app/
├── backend/              # FastAPI backend
│   ├── main.py          # API endpoints
│   ├── ingest.py        # Document processing
│   ├── generator.py     # Quiz generation logic
│   ├── database.py      # Database operations
│   ├── models.py        # Pydantic models
│   ├── llm_prompts.py   # LLM prompts
│   └── config.py        # Configuration
├── frontend/            # Next.js frontend
│   ├── app/            # Next.js app directory
│   ├── components/     # React components
│   └── lib/            # Utilities and API client
├── postgres/            # PostgreSQL initialization
├── tests/               # Test files
├── docker-compose.yml   # Docker orchestration
└── Makefile            # Build automation
```

### Running Locally (Without Docker)

**Backend**:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend**:
```bash
cd frontend
npm install
npm run dev
```

### Make Commands

```bash
make help              # Show all available commands
make build             # Build all Docker images
make build-no-cache    # Rebuild without cache
make up                # Start all services
make down              # Stop all services
make restart           # Restart all services
make logs              # View logs
make clean             # Remove containers and volumes (WARNING: deletes data)
make shell-backend     # Open shell in backend container
make shell-frontend    # Open shell in frontend container
make shell-postgres    # Open PostgreSQL shell
```

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
- PostgreSQL (pgvector) - Vector database
- SQLite - Metadata storage
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

### Database Connection Issues
- Ensure PostgreSQL container is running: `docker ps`
- Check connection string in `docker-compose.yml`
- Verify database credentials

### API Key Issues
- Ensure `OPENAI_API_KEY` is set in environment
- Check if API key is valid and has sufficient credits

### Port Conflicts
- Change port mappings in `docker-compose.yml` if ports 3000, 8000, or 5432 are in use

### Build Issues
- Try `make build-no-cache` to rebuild from scratch
- Check Docker logs: `make logs`

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]
