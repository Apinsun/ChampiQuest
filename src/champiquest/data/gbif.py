"""Ingestion des occurrences de champignons depuis GBIF (zone Alsace).

Usage :
    poetry run python -m champiquest.data.gbif            # télécharge tout
    poetry run python -m champiquest.data.gbif --dry-run  # affiche le plan seulement

Lit la config ``configs/species.yaml`` (espèces, bbox, années) et écrit un CSV
par espèce dans ``data/raw/gbif/<key>.csv``.
"""

import argparse
import io
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml
from pygbif import occurrences as occ

# Racine du dépôt : gbif.py est dans src/champiquest/data/, donc 3 niveaux au-dessus.
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "species.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw" / "gbif"

# Champs conservés pour chaque occurrence (voir la doc de l'API GBIF).
FIELDS = [
    "key",
    "scientificName",
    "taxonKey",
    "species",
    "eventDate",
    "year",
    "month",
    "day",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "basisOfRecord",
    "occurrenceStatus",
    "countryCode",
    "stateProvince",
    "datasetName",
    "issues",
]

PAGE_SIZE = 300  # taille maximale de page autorisée par l'API GBIF

# Champs minimaux pour l'ensemble des Fungi (signal de co-fructification / effort).
FUNGI_FIELDS = ["decimalLatitude", "decimalLongitude", "year", "month", "day", "taxonKey"]


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Charge la configuration YAML des espèces et de la zone d'étude."""
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    required = ["bbox", "year_range", "species"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Clés manquantes dans {path}: {missing}")
    return config


def _extract(record: dict) -> dict:
    """Extrait les champs utiles d'un enregistrement GBIF (tolérant aux clés absentes)."""
    row = {field: record.get(field) for field in FIELDS}
    # ``issues`` est une liste de codes de qualité : on la sérialise pour le CSV.
    if isinstance(row["issues"], list):
        row["issues"] = ",".join(row["issues"])
    return row


def fetch_species(gbif_key: int, bbox: dict, year_start: int, year_end: int) -> tuple[list[dict], int]:
    """Récupère toutes les occurrences d'un taxon dans la bbox, en paginant.

    Retourne ``(liste d'enregistrements, nombre total annoncé par GBIF)``.
    """
    records: list[dict] = []
    offset = 0
    total = None
    while True:
        response = occ.search(
            taxonKey=gbif_key,
            decimalLatitude=f"{bbox['min_lat']},{bbox['max_lat']}",
            decimalLongitude=f"{bbox['min_lon']},{bbox['max_lon']}",
            hasCoordinate=True,
            occurrenceStatus="PRESENT",
            year=f"{year_start},{year_end}",
            limit=PAGE_SIZE,
            offset=offset,
        )
        if total is None:
            total = response.get("count", 0)
        page = response.get("results", [])
        records.extend(page)
        # Arrêt : fin annoncée par GBIF, page vide, ou offset au-delà du total.
        if response.get("endOfRecords", False) or not page or offset + PAGE_SIZE >= (total or 0):
            break
        offset += PAGE_SIZE
        time.sleep(0.5)  # ménage l'API entre deux pages
    return records, total or 0


def _read_env(key: str) -> str:
    """Lit une clé KEY=VALEUR depuis ``.env`` (gitignoré)."""
    with open(REPO_ROOT / ".env", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1]
    raise ValueError(f"Clé {key} absente de .env")


def download_all_fungi(bbox: dict, year_start: int, year_end: int) -> pd.DataFrame:
    """Télécharge toutes les occurrences fongiques via l'API de téléchargement GBIF.

    L'API de recherche (pagination) est throttlée pour les gros volumes ; l'API de
    téléchargement (asynchrone) est la voie prévue pour ça. Format SIMPLE_CSV (TSV
    zippé), filtré aux champs FUNGI_FIELDS.
    """
    user = _read_env("GBIF_USER")
    password = _read_env("GBIF_PASSWORD")

    predicate = {
        "type": "and",
        "predicates": [
            {"type": "equals", "key": "KINGDOM_KEY", "value": "5"},
            {"type": "greaterThanOrEquals", "key": "DECIMAL_LATITUDE", "value": str(bbox["min_lat"])},
            {"type": "lessThanOrEquals", "key": "DECIMAL_LATITUDE", "value": str(bbox["max_lat"])},
            {"type": "greaterThanOrEquals", "key": "DECIMAL_LONGITUDE", "value": str(bbox["min_lon"])},
            {"type": "lessThanOrEquals", "key": "DECIMAL_LONGITUDE", "value": str(bbox["max_lon"])},
            {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
            {"type": "greaterThanOrEquals", "key": "YEAR", "value": str(year_start)},
            {"type": "lessThanOrEquals", "key": "YEAR", "value": str(year_end)},
        ],
    }
    body = {
        "creator": user,
        "notificationAddresses": [],
        "sendNotification": False,
        "format": "SIMPLE_CSV",
        "predicate": predicate,
    }

    resp = requests.post(
        "https://api.gbif.org/v1/occurrence/download/request",
        json=body, auth=(user, password), timeout=60,
    )
    resp.raise_for_status()
    download_key = resp.text.strip()
    print(f"Téléchargement soumis : {download_key}", flush=True)

    while True:
        time.sleep(10)
        st = requests.get(
            f"https://api.gbif.org/v1/occurrence/download/{download_key}",
            auth=(user, password), timeout=60,
        ).json()
        print(f"  statut : {st.get('status')}", flush=True)
        if st.get("status") == "SUCCEEDED":
            url = st["downloadLink"]
            break
        if st.get("status") in ("FAILED", "CANCELLED", "KILLED"):
            raise RuntimeError(f"Téléchargement GBIF : {st.get('status')}")

    zresp = requests.get(url, timeout=600)
    zresp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(zresp.content)) as z:
        name = z.namelist()[0]
        df = pd.read_csv(z.open(name), sep="\t", usecols=FUNGI_FIELDS, on_bad_lines="skip")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche uniquement ce qui serait téléchargé, sans interroger GBIF.",
    )
    parser.add_argument(
        "--all-fungi",
        action="store_true",
        help="Télécharge toutes les occurrences fongiques (kingdom=Fungi) de la bbox.",
    )
    args = parser.parse_args()

    config = load_config()
    bbox = config["bbox"]
    year_start = config["year_range"]["start"]
    # ``end: null`` dans le YAML → on prend l'année courante.
    year_end = config["year_range"]["end"] or datetime.now().year

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_fungi:
        print("Téléchargement de toutes les occurrences fongiques (kingdom=Fungi) …", flush=True)
        df = download_all_fungi(bbox, year_start, year_end)
        out = RAW_DIR / "all_fungi.csv"
        df.to_csv(out, index=False)
        print(f"  → {len(df)} lignes sauvegardées dans {out}")
        return

    for species in config["species"]:
        out_path = RAW_DIR / f"{species['key']}.csv"
        if args.dry_run:
            print(
                f"[dry-run] {species['scientific']} "
                f"(taxonKey {species['gbif_key']}, {year_start}–{year_end}) → {out_path}"
            )
            continue

        print(f"Récupération de {species['scientific']} …", flush=True)
        try:
            records, total = fetch_species(species["gbif_key"], bbox, year_start, year_end)
        except Exception as exc:  # une espèce en échec ne bloque pas les autres
            print(f"  ! Échec pour {species['scientific']}: {exc}")
            continue

        if not records:
            print(f"  → aucune occurrence (total GBIF annoncé : {total})")
            continue

        df = pd.DataFrame([_extract(record) for record in records])
        df.to_csv(out_path, index=False)
        print(f"  → {len(df)} lignes / {total} annoncées, sauvegardées dans {out_path}")


if __name__ == "__main__":
    main()
