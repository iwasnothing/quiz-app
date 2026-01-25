# ==============================================================================
# VARIABLES
# ==============================================================================
# Define image names for manual builds or registry pushing
BACKEND_IMAGE_NAME = quiz-backend
FRONTEND_IMAGE_NAME = quiz-frontend
TAG = latest

# Detect architecture (useful if you ever need cross-compilation, though mostly auto-detected)
ARCH ?= $(shell uname -m)

# Docker Compose Command (v2 is 'docker compose', v1 is 'docker-compose')
# We try to detect which one is available, defaulting to 'docker-compose'
DOCKER_COMPOSE := $(shell command -v docker-compose 2> /dev/null || echo "docker compose")

# ==============================================================================
# TARGETS
# ==============================================================================

.PHONY: help all build build-no-cache build-backend build-frontend up down restart logs clean prune shell-backend shell-frontend

# Default target: Show help
help:
	@echo "----------------------------------------------------------------------"
	@echo "  DGX Quiz App - Build & Management Automation"
	@echo "----------------------------------------------------------------------"
	@echo "  make build           Build all images (using cache)"
	@echo "  make build-no-cache  Rebuild all images from scratch"
	@echo "  make build-backend   Build only the backend image"
	@echo "  make build-frontend  Build only the frontend image"
	@echo "  make up              Start the full stack (detached mode)"
	@echo "  make down            Stop and remove containers"
	@echo "  make restart         Restart all services"
	@echo "  make logs            Follow logs for all services"
	@echo "  make clean           Stop containers and remove volumes (RESET DB)"
	@echo "  make shell-backend   Open a bash shell inside the running backend"
	@echo "  make shell-frontend  Open a shell inside the running frontend"
	@echo "----------------------------------------------------------------------"

# Build everything using docker-compose
all: build up

# Build all services defined in docker-compose.yml
build:
	@echo "Building images for $(ARCH)..."
	$(DOCKER_COMPOSE) build

# Force rebuild without using cache (useful if pip/npm packages change but aren't picked up)
build-no-cache:
	@echo "Rebuilding images from scratch..."
	$(DOCKER_COMPOSE) build --no-cache

# Build only the backend image
build-backend:
	@echo "Building backend image..."
	docker build -t $(BACKEND_IMAGE_NAME):$(TAG) ./backend

# Build only the frontend image
build-frontend:
	@echo "Building frontend image..."
	docker build -t $(FRONTEND_IMAGE_NAME):$(TAG) ./frontend

# Start the application in detached mode
up:
	@echo "Starting services..."
	$(DOCKER_COMPOSE) up -d
	@echo "Services started. Frontend: http://localhost:3000 | Neo4j: http://localhost:7474"

# Stop the application
down:
	@echo "Stopping services..."
	$(DOCKER_COMPOSE) down

# Restart the application
restart: down up

# View logs (press Ctrl+C to exit)
logs:
	$(DOCKER_COMPOSE) logs -f

# ==============================================================================
# DEBUGGING & MAINTENANCE
# ==============================================================================

# Open a terminal inside the Backend container (FastAPI)
shell-backend:
	docker exec -it quiz-backend /bin/bash

# Open a terminal inside the Frontend container (Next.js)
shell-frontend:
	docker exec -it quiz-frontend /bin/sh

# Open a terminal inside the Neo4j container
shell-neo4j:
	docker exec -it neo4j-gds-arm cypher-shell -u neo4j -p password

# Open a terminal inside the Postgres container
shell-postgres:
	docker exec -it pgvector-arm psql -U myuser -d quiz_vector_db

# WARNING: DESTRUCTIVE! Removes all data (Database & Vector Store)
clean:
	@echo "WARNING: This will delete all database volumes."
	@read -p "Are you sure? [y/N] " ans && [ $${ans:-N} = y ]
	$(DOCKER_COMPOSE) down -v
	@echo "Clean complete."

# Remove dangling images to free up space on the DGX
prune:
	docker system prune -f