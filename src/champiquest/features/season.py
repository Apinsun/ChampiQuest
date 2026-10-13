"""Calcul des fenêtres de saison de fructification par espèce.

Méthode : percentiles 5 % et 95 % du jour-de-l'année des observations, puis
buffer de ±14 jours (pour anticiper les décalages climatiques). Les fenêtres
sont écrites dans ``configs/season_windows.yaml`` (lu plus tard par l'API pour
la règle métier « hors saison »).

Usage :
    poetry run python -m champiquest.features.season
"""

from pathlib import Path

import pandas as pd
import yaml

# Racine du dépôt : season.py est dans src/champiquest/features/, donc 3 niveaux au-dessus.
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw" / "gbif"
OUT_PATH = REPO_ROOT / "configs" / "season_windows.yaml"

BUFFER_DAYS = 14
LOW_Q = 0.05
HIGH_Q = 0.95


def load_occurrences(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Charge les CSVs bruts GBIF (un par taxon) et ajoute le jour-de-l'année.

    Le jour-de-l'année est reconstruit depuis les colonnes ``year``/``month``/``day``
    (déjà numériques et quasi complètes), plutôt que depuis ``eventDate`` dont le
    format texte est hétérogène (parsing peu fiable).
    """
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(path)
        date = pd.to_datetime(
            {"year": df["year"], "month": df["month"], "day": df["day"]},
            errors="coerce",
        )
        df["day_of_year"] = date.dt.dayofyear
        frames[path.stem] = df
    return frames


def compute_windows(
    frames: dict[str, pd.DataFrame],
    low_q: float = LOW_Q,
    high_q: float = HIGH_Q,
    buffer_days: int = BUFFER_DAYS,
) -> dict:
    """Calcule, pour chaque taxon, la fenêtre de saison (percentiles + buffer).

    Note : on ignore un éventuel passage par le 1er janvier (wrap-around) —
    acceptable pour nos espèces (aucune ne fructifie à cheval sur l'année).
    """
    windows: dict = {}
    for key, df in frames.items():
        doy = df["day_of_year"].dropna()
        p5 = int(doy.quantile(low_q))
        p95 = int(doy.quantile(high_q))
        windows[key] = {
            "start_doy": p5 - buffer_days,
            "end_doy": p95 + buffer_days,
            "n_obs": int(len(doy)),
            "p5_raw": p5,
            "p95_raw": p95,
        }
    return windows


def save_windows(windows: dict, out_path: Path = OUT_PATH) -> None:
    """Écrit les fenêtres dans un fichier YAML (source de config pour l'API)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(windows, fh, sort_keys=False, allow_unicode=True)


def main() -> None:
    frames = load_occurrences()
    windows = compute_windows(frames)
    save_windows(windows)
    print(f"Fenêtres écrites dans {OUT_PATH}\n")
    for key, win in windows.items():
        print(
            f"{key:24s} jour {win['start_doy']:3d} → {win['end_doy']:3d}  "
            f"(p5={win['p5_raw']}, p95={win['p95_raw']}, n={win['n_obs']})"
        )


if __name__ == "__main__":
    main()
