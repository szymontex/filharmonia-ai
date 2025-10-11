#!/bin/bash
# Filharmonia AI - Startup Script (Linux/Mac)
# Starts both backend and frontend servers and opens browser

echo "================================"
echo "  Filharmonia AI - Starting..."
echo "================================"
echo

# Kill existing servers if running
echo "[0/4] Stopping existing servers..."

# Kill backend (port 8000)
lsof -ti:8000 | xargs kill -9 2>/dev/null

# Kill frontend (port 5173)
lsof -ti:5173 | xargs kill -9 2>/dev/null

# Also kill by process name as backup
pkill -f "uvicorn app.main:app" 2>/dev/null
pkill -f "vite" 2>/dev/null

sleep 2

# Check if backend venv exists
if [ ! -f "backend/venv/bin/activate" ]; then
    echo "[ERROR] Backend virtual environment not found!"
    echo "Please run: cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "[ERROR] Frontend dependencies not installed!"
    echo "Please run: cd frontend && pnpm install"
    exit 1
fi

echo "[1/4] Starting backend server..."
cd backend
source venv/bin/activate
nohup python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..
sleep 3

echo "[2/4] Starting frontend server..."
cd frontend
nohup pnpm dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
sleep 5

echo "[3/4] Waiting for servers to start..."
sleep 3

echo "[4/4] Opening browser..."
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:5173
elif command -v open > /dev/null; then
    open http://localhost:5173
else
    echo "Could not detect browser command. Please open http://localhost:5173 manually"
fi

echo
echo "================================"
echo "  Servers are running!"
echo "================================"
echo
echo "Backend:  http://localhost:8000 (PID: $BACKEND_PID)"
echo "Frontend: http://localhost:5173 (PID: $FRONTEND_PID)"
echo "API Docs: http://localhost:8000/docs"
echo
echo "Logs: backend.log, frontend.log"
echo
echo "To stop servers:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo
