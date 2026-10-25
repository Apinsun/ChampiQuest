# Image ChampiQuest — API FastAPI (code + dépendances runtime).
# Les modèles et les données (data/models, data/weather, data/maps) sont montés
# en VOLUME au runtime, PAS copiés dans l'image : retrainer un modèle ne demande
# donc aucun rebuild.

FROM python:3.12-slim

WORKDIR /app

# dépendances runtime uniquement (pas geopandas/rasterio/jupyter → pas de GDAL)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# code + configs (data/ est exclu par .dockerignore, monté en volume)
COPY src/ src/
COPY configs/ configs/

ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "champiquest.api:app", "--host", "0.0.0.0", "--port", "8000"]
