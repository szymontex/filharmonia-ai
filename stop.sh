#!/usr/bin/env bash
# Filharmonia AI - Stop Script (Linux/Mac)
# Stops both backend and frontend servers

echo "Stopping Filharmonia AI servers..."

# Kill by process name (catches all workers/children)
UVICORN_PIDS=$(pgrep -f "uvicorn app.main:app" 2>/dev/null)
if [ -n "$UVICORN_PIDS" ]; then
    echo "Stopping backend (PIDs: $UVICORN_PIDS)..."
    echo "$UVICORN_PIDS" | xargs -r kill 2>/dev/null
else
    echo "Backend not running"
fi

VITE_PIDS=$(pgrep -f "vite" 2>/dev/null)
if [ -n "$VITE_PIDS" ]; then
    echo "Stopping frontend (PIDs: $VITE_PIDS)..."
    echo "$VITE_PIDS" | xargs -r kill 2>/dev/null
else
    echo "Frontend not running"
fi

# Safety net: kill anything on our ports
sleep 1
PORT_PIDS=$(lsof -ti:8000,5173 2>/dev/null || true)
if [ -n "$PORT_PIDS" ]; then
    echo "Cleaning up remaining processes on ports 8000/5173..."
    echo "$PORT_PIDS" | xargs -r kill -9 2>/dev/null
fi

echo "Done!"
