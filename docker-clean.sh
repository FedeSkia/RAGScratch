#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"

# Ferma e rimuove tutto (containers, volumes, networks)
echo "Removing orphans..."
docker compose -f docker/docker-compose.yml down -v --remove-orphans

echo "Deleting immagine rag-app:latest..."
docker rmi rag-app:latest || echo "⚠️  Immagine non trovata, skipping..."

echo "Cleaned"
