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

# Installe le groupe `main` + le paquet champiquest dans le Python système,
# puis, dans le MÊME layer (pour que l'image finale ne garde aucune trace des
# bibliothèques CUDA) : remplace torch CUDA par torch CPU, retire les
# dépendances nvidia-*/triton (inutiles en inférence CPU) et purge le cache
# pip/poetry (~5 Go). Résultat : image d'~20 Go → ~2 Go.
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction \
    && pip install --no-cache-dir --force-reinstall --no-deps \
       --index-url https://download.pytorch.org/whl/cpu \
       "torch==2.11.0+cpu" "torchvision==0.26.0+cpu" \
    && pip uninstall -y $(pip list --format=freeze | cut -d= -f1 | grep -E '^(nvidia-|triton|cuda-python)') || true \
    && rm -rf /root/.cache

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "champiquest.api:app", "--host", "0.0.0.0", "--port", "8000"]
