#!/usr/bin/env bash
# start.sh
set -euo pipefail
trap 'kill 0' SIGINT SIGTERM

if [[ "${DISABLE_OLLAMA:-0}" != "1" ]]; then
  echo "Starting Ollama..."
  # Run Ollama in the background
  ollama serve &
  # Wait for Ollama to be ready
  until curl -sSf "${OLLAMA_BASE_URL:-http://127.0.0.1:11434}/api/tags" >/dev/null; do
    echo "Waiting for Ollama..."
    sleep 1
  done
  echo "Ollama is up."
  # Pull the specified Ollama model if it's set
  if [[ -n "${OLLAMA_MODEL:-}" ]]; then
    echo "Pulling model: $OLLAMA_MODEL"
    # Use '|| true' to prevent script from exiting if model pull fails.
    # In a production setup, you might want more robust error handling.
    ollama pull "$OLLAMA_MODEL" || true
  fi
else
  echo "Ollama disabled (DISABLE_OLLAMA=1)."
fi

echo "Starting Flask with Gunicorn..."
# Execute Gunicorn, replacing the shell process (important for Docker health checks and signal handling)
# -w 2: run with 2 worker processes (adjust based on CPU cores and memory)
# -b 0.0.0.0:5000: bind to all network interfaces on port 5000
# app:app: Specifies the Flask application instance 'app' within the 'app.py' module
exec gunicorn -w 2 -b 0.0.0.0:5000 app:app