#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "Stop Docker Compose..."

cd "$PROJECT_ROOT"
docker compose -f docker/docker-compose.yml down "$@"
