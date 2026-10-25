#!/usr/bin/env bash
# Déploie l'API ChampiQuest depuis GHCR : pull la dernière image puis lance le service.
# Usage : ./deploy.sh   (prérequis : le volume ./data avec modèles + météo + cartes)
set -euo pipefail
cd "$(dirname "$0")"
# --env-file /dev/null : évite que docker compose lise le .env du projet
# (qui est un JSON de credentials Copernicus, pas un fichier KEY=VALUE)
docker compose --env-file /dev/null -f docker-compose.prod.yml pull
docker compose --env-file /dev/null -f docker-compose.prod.yml up -d
echo "API démarrée (image à jour) : http://localhost:8000"
