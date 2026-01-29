#!/bin/bash
# Docker Production Startup Script

set -e

echo "Starting Filharmonia AI..."

# Start backend in background
cd /app/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start nginx in foreground
echo "Starting nginx..."
nginx -g 'daemon off;' &
NGINX_PID=$!

# Handle shutdown
cleanup() {
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $NGINX_PID 2>/dev/null
    exit 0
}

trap cleanup SIGTERM SIGINT

# Wait for processes
wait
