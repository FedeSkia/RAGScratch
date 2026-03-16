#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "Starting Docker Compose: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

docker compose -f docker/docker-compose.yml up "$@"