# ========================================
#  Filharmonia AI - Docker Image
#  Multi-stage build for production
# ========================================

# Stage 1: Backend
FROM python:3.11-slim AS backend

WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and filter out CUDA torch packages
COPY backend/requirements.txt .
RUN grep -v '^torch\|^torchaudio\|^torchvision' requirements.txt > requirements-docker.txt

# Install PyTorch CPU-only (lighter image)
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (without torch)
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy backend code
COPY backend/ .

# Export ONNX model for CPU speedup
RUN python -m scripts.export_onnx || echo "ONNX export failed (non-critical)"

# Stage 2: Frontend
FROM node:20-slim AS frontend

WORKDIR /app/frontend

# Install pnpm
RUN npm install -g pnpm

# Copy frontend package files
COPY frontend/package.json frontend/pnpm-lock.yaml ./

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy frontend code
COPY frontend/ .

# Build frontend
RUN pnpm build

# Stage 3: Production
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy backend from stage 1
COPY --from=backend /app/backend /app/backend
COPY --from=backend /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend /usr/local/bin /usr/local/bin

# Copy frontend build from stage 2
COPY --from=frontend /app/frontend/dist /app/frontend/dist

# Copy nginx config
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Copy startup script
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start services
CMD ["/app/start.sh"]
