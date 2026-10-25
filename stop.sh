#!/usr/bin/env bash
# Arrête l'API ChampiQuest.
set -euo pipefail
cd "$(dirname "$0")"
docker compose --env-file /dev/null down
echo "API arrêtée."
