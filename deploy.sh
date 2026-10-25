#!/usr/bin/env bash
# Déploie l'API ChampiQuest depuis GHCR : pull la dernière image puis lance le service.
# Usage : ./deploy.sh   (prérequis : le volume ./data avec modèles + météo + cartes)
set -euo pipefail
cd "$(dirname "$0")"
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
echo "API démarrée (image à jour) : http://localhost:8000"
