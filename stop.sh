#!/usr/bin/env bash
# Arrête l'API ChampiQuest.
set -euo pipefail
cd "$(dirname "$0")"
docker compose down
echo "API arrêtée."
