# ARM64v8 Node base image (works on most ARM Linux, e.g. Pi 4 64‑bit)
FROM arm64v8/node:18-alpine

# Install localtunnel globally
RUN npm install -g localtunnel

# Expose the local port you want to tunnel (optional but conventional)
EXPOSE 3000

# Command is specified in docker-compose.yml