#!/usr/bin/env bash
# ========================================
#  Filharmonia AI - Start Application
#  macOS / Linux
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "========================================"
echo "  Filharmonia AI - Starting"
echo "========================================"
echo ""

# Check if setup was completed
if [ ! -f ".setup_complete" ]; then
    echo "[ERROR] Setup not completed!"
    echo "Please run ./setup.sh first."
    exit 1
fi

if [ ! -f "backend/venv/bin/activate" ]; then
    echo "[ERROR] Backend virtual environment not found!"
    echo "Please run ./setup.sh first."
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "[ERROR] Frontend dependencies not installed!"
    echo "Please run ./setup.sh first."
    exit 1
fi

# Track all child PIDs for cleanup
CHILD_PIDS=()

CLEANED_UP=false

cleanup() {
    # Prevent running twice (TERM + EXIT)
    if [ "$CLEANED_UP" = true ]; then return; fi
    CLEANED_UP=true

    echo ""
    echo "Stopping servers..."

    # Kill process groups spawned by this script
    for pid in "${CHILD_PIDS[@]}"; do
        # Kill entire process group (catches child workers)
        kill -- -"$pid" 2>/dev/null
        # Fallback: kill the PID itself + its children
        pkill -P "$pid" 2>/dev/null
        kill "$pid" 2>/dev/null
    done

    # Safety net: kill anything still on our ports
    sleep 1
    local port_pids
    port_pids=$(lsof -ti:8000,5173 2>/dev/null || true)
    if [ -n "$port_pids" ]; then
        echo "$port_pids" | xargs kill -9 2>/dev/null
    fi

    echo "All servers stopped."
    exit 0
}

trap cleanup INT TERM EXIT

# Kill existing servers on our ports
echo "[1/4] Stopping existing servers..."
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:5173 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# Start backend in its own process group (set -m equivalent via setsid)
echo "[2/4] Starting backend server..."
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
if command -v setsid &>/dev/null; then
    setsid python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > "$SCRIPT_DIR/backend.log" 2>&1 &
else
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > "$SCRIPT_DIR/backend.log" 2>&1 &
fi
BACKEND_PID=$!
CHILD_PIDS+=($BACKEND_PID)
cd "$SCRIPT_DIR"

sleep 3

# Start frontend in its own process group
echo "[3/4] Starting frontend server..."
cd "$SCRIPT_DIR/frontend"
if command -v setsid &>/dev/null; then
    setsid pnpm dev > "$SCRIPT_DIR/frontend.log" 2>&1 &
else
    pnpm dev > "$SCRIPT_DIR/frontend.log" 2>&1 &
fi
FRONTEND_PID=$!
CHILD_PIDS+=($FRONTEND_PID)
cd "$SCRIPT_DIR"

sleep 3

# Verify servers started
BACKEND_OK=false
FRONTEND_OK=false

if kill -0 "$BACKEND_PID" 2>/dev/null; then
    BACKEND_OK=true
fi
if kill -0 "$FRONTEND_PID" 2>/dev/null; then
    FRONTEND_OK=true
fi

# Open browser
echo "[4/4] Opening browser..."
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:5173 2>/dev/null
elif command -v open &>/dev/null; then
    open http://localhost:5173
fi

echo ""
echo "========================================"
echo "  Application Running"
echo "========================================"
echo ""
if [ "$BACKEND_OK" = true ]; then
    echo "Backend:  http://localhost:8000 (PID: $BACKEND_PID)"
else
    echo "Backend:  FAILED TO START — check backend.log"
fi
if [ "$FRONTEND_OK" = true ]; then
    echo "Frontend: http://localhost:5173 (PID: $FRONTEND_PID)"
else
    echo "Frontend: FAILED TO START — check frontend.log"
fi
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Logs:"
echo "  Backend:  backend.log"
echo "  Frontend: frontend.log"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for any child to exit
wait
