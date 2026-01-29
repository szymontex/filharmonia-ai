#!/usr/bin/env bash
# ========================================
#  Filharmonia AI - Start Application
#  macOS / Linux
# ========================================

echo ""
echo "========================================"
echo "  Filharmonia AI - Starting"
echo "========================================"
echo ""

# Check if setup was completed
if [ ! -f ".setup_complete" ]; then
    echo "[ERROR] Setup not completed!"
    echo ""
    echo "Please run ./setup.sh first to install dependencies."
    echo ""
    exit 1
fi

# Verify backend setup
if [ ! -f "backend/venv/bin/activate" ]; then
    echo "[ERROR] Backend virtual environment not found!"
    echo ""
    echo "Please run ./setup.sh to install dependencies."
    echo ""
    exit 1
fi

# Verify frontend setup
if [ ! -d "frontend/node_modules" ]; then
    echo "[ERROR] Frontend dependencies not installed!"
    echo ""
    echo "Please run ./setup.sh to install dependencies."
    echo ""
    exit 1
fi

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Kill existing servers
echo "[1/4] Stopping existing servers..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 2

# Start backend
echo "[2/4] Starting backend server..."
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

sleep 3

# Start frontend
echo "[3/4] Starting frontend server..."
cd frontend
pnpm dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

sleep 5

# Open browser
echo "[4/4] Opening browser..."
if command -v open &> /dev/null; then
    # macOS
    open http://localhost:5173
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open http://localhost:5173
fi

echo ""
echo "========================================"
echo "  Application Running"
echo "========================================"
echo ""
echo "Backend:  http://localhost:8000 (PID: $BACKEND_PID)"
echo "Frontend: http://localhost:5173 (PID: $FRONTEND_PID)"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Logs:"
echo "  Backend:  backend.log"
echo "  Frontend: frontend.log"
echo ""
echo "To stop servers:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo "Or run: ./stop.sh"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for user interrupt
wait
