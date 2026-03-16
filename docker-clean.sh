#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"

# Stops and removes everything (containers, volumes, networks)
echo "Removing orphans..."
docker compose -f docker/docker-compose.yml down -v --remove-orphans

echo "Deleting image rag-app:latest..."
docker rmi rag-app:latest || echo "⚠️  Image not found, skipping..."

echo "Cleaned"
