#!/bin/bash
# GlobalFlood AI — One-click launcher

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VENV="$PROJECT_DIR/../.venv/bin"
NODE_BIN="$HOME/.local/node/bin"

export PATH="$NODE_BIN:$PYTHON_VENV:$PATH"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/backend:$PYTHONPATH"

echo "=================================================="
echo " Starting GlobalFlood AI Services..."
echo "=================================================="

# Check if already running
if lsof -i :8000 > /dev/null 2>&1; then
  echo "⚠️  Backend already running on port 8000."
else
  echo "🚀 Launching FastAPI Backend on http://127.0.0.1:8000 ..."
  nohup "$PYTHON_VENV/uvicorn" app.main:app --app-dir "$PROJECT_DIR/backend" --host 127.0.0.1 --port 8000 > "$PROJECT_DIR/backend.log" 2>&1 &
  echo "   Backend started (PID $!). Logs: backend.log"
fi

if lsof -i :5173 > /dev/null 2>&1; then
  echo "⚠️  Frontend already running on port 5173."
else
  echo "🚀 Launching Vite Frontend on http://127.0.0.1:5173 ..."
  cd "$PROJECT_DIR/frontend" && nohup npm run dev > "$PROJECT_DIR/frontend.log" 2>&1 &
  echo "   Frontend started (PID $!). Logs: frontend.log"
fi

echo ""
echo "✅ GlobalFlood AI is active!"
echo "   - Frontend UI: http://127.0.0.1:5173"
echo "   - Backend API: http://127.0.0.1:8000/docs"
echo "=================================================="
