#!/usr/bin/env bash
# Smoke test local de l'API ChampiQuest — à lancer APRÈS deploy.sh (ou start.sh).
# Vérifie les endpoints clés avec les VRAIS modèles (pas de mock).
# Variables : SMOKE_URL (défaut http://localhost:8000), SMOKE_IMG (image de test).
set -euo pipefail

URL="${SMOKE_URL:-http://localhost:8000}"
IMG="${SMOKE_IMG:-data/raw/df20/extracted_300px/2238476543-19112.jpg}"
DATE="$(date +%F)"

echo "smoke test sur $URL"

echo -n "  /health         : "
curl -sf "$URL/health" | grep -q '"ok"' && echo "✔"

echo -n "  /species        : "
N=$(curl -sf "$URL/species" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['species']))")
echo "✔ ($N espèces)"

echo -n "  /growth/predict : "
curl -sf -X POST "$URL/growth/predict" -H "Content-Type: application/json" \
  -d "{\"lat\":48.5,\"lon\":7.5,\"date\":\"$DATE\"}" > /dev/null && echo "✔"

echo -n "  /identify       : "
if [ -f "$IMG" ]; then
  B64=$(base64 -w0 "$IMG")
  curl -sf -X POST "$URL/identify" -H "Content-Type: application/json" \
    -d "{\"images\":[\"$B64\"]}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('✔', d['decision'], '→', d['top_predictions'][0]['species'])"
else
  echo "⚠ image de test absente ($IMG) — /identify non testé"
fi

echo -n "  /map            : "
code=$(curl -s -o /dev/null -w "%{http_code}" "$URL/map?date=$DATE")
[ "$code" = "200" ] && echo "✔" || echo "⚠ (HTTP $code — pas de carte pour $DATE)"

echo "✔ smoke test terminé"
