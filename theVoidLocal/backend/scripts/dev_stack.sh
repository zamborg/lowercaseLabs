#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${REPO_ROOT}/docker-compose.yml}"

# For backward compatibility, keep argv shape: dev_stack.sh [HOST] [PORT]
HOST="${1:-127.0.0.1}"
API_PORT="${2:-${API_PORT:-8080}}"
POSTGRES_PORT="${POSTGRES_PORT:-55432}"
FOLLOW_LOGS="${FOLLOW_LOGS:-1}"
CLEANED_UP=0

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Install Docker Desktop or Docker Engine."
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin not found."
  exit 1
fi

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Compose file not found at: ${COMPOSE_FILE}"
  exit 1
fi

compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

cleanup() {
  if [[ "${CLEANED_UP}" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1
  echo
  echo "Stopping dockerized local stack..."
  API_PORT="${API_PORT}" POSTGRES_PORT="${POSTGRES_PORT}" compose down
}

echo "Starting dockerized local stack..."
echo "API endpoint: http://${HOST}:${API_PORT}"
echo "Postgres host port: ${POSTGRES_PORT}"

API_PORT="${API_PORT}" POSTGRES_PORT="${POSTGRES_PORT}" compose up -d db
API_PORT="${API_PORT}" POSTGRES_PORT="${POSTGRES_PORT}" compose run --rm migrate
API_PORT="${API_PORT}" POSTGRES_PORT="${POSTGRES_PORT}" compose up -d api worker

if [[ "${FOLLOW_LOGS}" == "1" ]]; then
  trap cleanup INT TERM
  echo "Streaming logs. Press Ctrl+C to stop the stack."
  API_PORT="${API_PORT}" POSTGRES_PORT="${POSTGRES_PORT}" compose logs -f --tail=200 db api worker
  cleanup
else
  echo "Stack started in detached mode."
  echo "Logs: API_PORT=${API_PORT} POSTGRES_PORT=${POSTGRES_PORT} docker compose -f ${COMPOSE_FILE} logs -f db api worker"
  echo "Stop: API_PORT=${API_PORT} POSTGRES_PORT=${POSTGRES_PORT} docker compose -f ${COMPOSE_FILE} down"
fi
