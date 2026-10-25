# Image ChampiQuest — API FastAPI (code + dépendances runtime).
# Les modèles et les données (data/models, data/weather, data/maps) sont montés
# en VOLUME au runtime, PAS copiés dans l'image : retrainer un modèle ne demande
# donc aucun rebuild.
# Dépendances : `poetry install --only main` = uniquement le groupe runtime
# (pas les groupes train/test → ni GDAL, ni mlflow, ni jupyter).

FROM python:3.12-slim

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY src/ src/
COPY configs/ configs/

# installe le groupe `main` + le paquet champiquest dans le Python système
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "champiquest.api:app", "--host", "0.0.0.0", "--port", "8000"]
