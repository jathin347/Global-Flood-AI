#!/bin/bash
# GlobalFlood AI — One-click shutdown

echo "=================================================="
echo " Stopping GlobalFlood AI Services..."
echo "=================================================="

# Stop Backend on 8000
BACKEND_PID=$(lsof -t -i :8000 2>/dev/null)
if [ -n "$BACKEND_PID" ]; then
  echo "🛑 Stopping Backend (PID $BACKEND_PID)..."
  kill $BACKEND_PID 2>/dev/null || kill -9 $BACKEND_PID 2>/dev/null
  echo "   Backend stopped."
else
  echo "ℹ️  Backend is not running on port 8000."
fi

# Stop Frontend on 5173
FRONTEND_PID=$(lsof -t -i :5173 2>/dev/null)
if [ -n "$FRONTEND_PID" ]; then
  echo "🛑 Stopping Frontend (PID $FRONTEND_PID)..."
  kill $FRONTEND_PID 2>/dev/null || kill -9 $FRONTEND_PID 2>/dev/null
  echo "   Frontend stopped."
else
  echo "ℹ️  Frontend is not running on port 5173."
fi

echo ""
echo "✅ All GlobalFlood AI services are deactivated."
echo "=================================================="
