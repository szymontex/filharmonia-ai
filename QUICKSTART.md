# 🚀 Quick Start

## Windows

1. **Install Prerequisites:**
   - Python 3.10+: https://python.org (✅ check "Add Python to PATH")
   - Node.js 18+: https://nodejs.org

2. **Setup (one time):**
   ```
   Double-click: setup-windows.bat
   ```

3. **Run:**
   ```
   Double-click: run-windows.bat
   ```

4. **Open:** http://localhost:5173

---

## macOS

1. **Install Prerequisites:**
   ```bash
   brew install python@3.10 node
   ```

2. **Setup (one time):**
   ```bash
   chmod +x setup-macos-linux.sh
   ./setup-macos-linux.sh
   ```

3. **Run:**
   ```bash
   ./run-macos-linux.sh
   ```

4. **Open:** http://localhost:5173

---

## Linux (Ubuntu)

1. **Install Prerequisites:**
   ```bash
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3-pip
   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
   sudo apt install -y nodejs
   ```

2. **Setup (one time):**
   ```bash
   chmod +x setup-macos-linux.sh
   ./setup-macos-linux.sh
   ```

3. **Run:**
   ```bash
   ./run-macos-linux.sh
   ```

4. **Open:** http://localhost:5173

---

## What You'll See

Backend starts on port 8000, Frontend on port 5173.

Startup logs show device detection:
```
INFO: Device: cpu (CPU)
INFO: Using ONNX INT8 backend (3.2x speedup vs PyTorch CPU)
```

---

## Troubleshooting

**Backend won't start?**
- Check Python installed: `python --version` (Windows) or `python3 --version` (Mac/Linux)
- Check port 8000 free: `netstat -ano | findstr :8000` (Windows) or `lsof -i :8000` (Mac/Linux)

**Frontend won't start?**
- Check Node.js installed: `node --version`
- Check port 5173 free (Vite will suggest another port if busy)

**Slow inference?**
- Check ONNX model exists: `ls backend/recognition_models/ast_active_int8.onnx`
- Re-run export: `cd backend && venv/bin/activate && python -m scripts.export_onnx`

---

Full docs: See **SETUP.md**
