#!/bin/bash
# Filharmonia AI - Stop Script (Linux/Mac)
# Stops both backend and frontend servers

echo "Stopping Filharmonia AI servers..."

# Find and kill backend (uvicorn)
BACKEND_PIDS=$(pgrep -f "uvicorn app.main:app")
if [ -n "$BACKEND_PIDS" ]; then
    echo "Stopping backend (PIDs: $BACKEND_PIDS)..."
    kill $BACKEND_PIDS
else
    echo "Backend not running"
fi

# Find and kill frontend (vite)
FRONTEND_PIDS=$(pgrep -f "vite")
if [ -n "$FRONTEND_PIDS" ]; then
    echo "Stopping frontend (PIDs: $FRONTEND_PIDS)..."
    kill $FRONTEND_PIDS
else
    echo "Frontend not running"
fi

echo "Done!"
