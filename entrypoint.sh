#!/bin/bash
set -eo pipefail

# If OLLAMA_BASE_URL points to localhost/127.0.0.1 inside a container,
# automatically map it to host.docker.internal if host-gateway is configured.
if [[ "$OLLAMA_BASE_URL" == "http://localhost:11434" || "$OLLAMA_BASE_URL" == "http://127.0.0.1:11434" || -z "$OLLAMA_BASE_URL" ]]; then
  if getent hosts host.docker.internal >/dev/null 2>&1; then
    export OLLAMA_BASE_URL="http://host.docker.internal:11434"
    echo "[Docker Init] Auto-mapped OLLAMA_BASE_URL to host.docker.internal:11434 for host Ollama connectivity."
  fi
fi

# Determine mode: CLI argument $1 takes priority, then APP_MODE env var, defaulting to 'both'
MODE="${1:-${APP_MODE:-both}}"

case "$MODE" in
  fastapi|api)
    FASTAPI_PORT="${PORT:-8080}"
    echo "=================================================="
    echo " Starting FastAPI backend service"
    echo " Port:      ${FASTAPI_PORT}"
    echo " Docs URL:  http://0.0.0.0:${FASTAPI_PORT}/docs"
    echo "=================================================="
    exec uvicorn app.main:app --host 0.0.0.0 --port "${FASTAPI_PORT}"
    ;;

  streamlit|ui)
    STREAMLIT_PORT="${STREAMLIT_SERVER_PORT:-${PORT:-8501}}"
    echo "=================================================="
    echo " Starting Streamlit UI service"
    echo " Port:      ${STREAMLIT_PORT}"
    echo " URL:       http://0.0.0.0:${STREAMLIT_PORT}"
    echo "=================================================="
    exec streamlit run app/streamlit_app.py --server.port "${STREAMLIT_PORT}" --server.address 0.0.0.0
    ;;

  both|all)
    FASTAPI_PORT="${FASTAPI_PORT:-${PORT:-8080}}"
    STREAMLIT_PORT="${STREAMLIT_SERVER_PORT:-8501}"
    echo "=================================================="
    echo " Starting Unified Multi-Service Container:"
    echo "   - FastAPI backend: http://0.0.0.0:${FASTAPI_PORT}"
    echo "   - Streamlit UI:    http://0.0.0.0:${STREAMLIT_PORT}"
    echo "=================================================="

    # Start FastAPI backend in background
    uvicorn app.main:app --host 0.0.0.0 --port "${FASTAPI_PORT}" &
    FASTAPI_PID=$!

    # Start Streamlit UI in background
    streamlit run app/streamlit_app.py --server.port "${STREAMLIT_PORT}" --server.address 0.0.0.0 &
    STREAMLIT_PID=$!

    # Graceful shutdown handler for SIGINT / SIGTERM
    term_handler() {
      echo ""
      echo "Signal caught! Gracefully shutting down services..."
      kill -TERM "$FASTAPI_PID" 2>/dev/null || true
      kill -TERM "$STREAMLIT_PID" 2>/dev/null || true
      wait "$FASTAPI_PID" 2>/dev/null || true
      wait "$STREAMLIT_PID" 2>/dev/null || true
      echo "All services shut down cleanly."
      exit 0
    }

    trap term_handler SIGINT SIGTERM

    # Wait for either process to exit
    wait -n "$FASTAPI_PID" "$STREAMLIT_PID"
    EXIT_STATUS=$?
    echo "A service process exited with code ${EXIT_STATUS}. Stopping container..."
    term_handler
    exit "$EXIT_STATUS"
    ;;

  *)
    # Pass-through for custom commands (e.g. bash, pytest, python)
    exec "$@"
    ;;
esac
