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

# installe le groupe `main` + le paquet champiquest dans le Python système,
# puis remplace torch CUDA (build par défaut de PyPI) par torch CPU dans le
# MÊME layer : l'inférence tourne sur CPU, on évite ainsi d'embarquer ~7 Go de
# bibliothèques CUDA inutilisées dans l'image finale.
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction \
    && pip install --no-cache-dir --force-reinstall --no-deps \
       --index-url https://download.pytorch.org/whl/cpu \
       "torch==2.11.0+cpu" "torchvision==0.26.0+cpu"

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "champiquest.api:app", "--host", "0.0.0.0", "--port", "8000"]
