"""Configuration MLflow partagée (tracking local + expérience unique).

Chaque script d'entraînement (train.py) appelle ``setup()`` avant de lancer des
runs : tous les runs se retrouvent dans la même expérience, traçables et
comparables dans l'UI MLflow.

Tracking local : base SQLite ``mlflow.db`` + artifacts ``mlartifacts/`` (gitignorés).
"""

import mlflow

EXPERIMENT_NAME = "champiquest"
TRACKING_URI = None  # None = tracking local par défaut (mlruns/)


def setup() -> None:
    """Configure le tracking MLflow et crée/récupère l'expérience du projet."""
    if TRACKING_URI is not None:
        mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
