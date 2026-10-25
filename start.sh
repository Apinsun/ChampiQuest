#!/usr/bin/env bash
# Démarre l'API ChampiQuest (construit l'image si besoin, puis lance le service).
set -euo pipefail
cd "$(dirname "$0")"
docker compose up -d --build
echo "API démarrée : http://localhost:8000"
