"""API ChampiQuest (FastAPI).

Endpoints :
- GET  /health         → statut du service
- POST /identify       → 1..N photos (base64) → espèce (modèle vision, partie 2)
- POST /growth/predict → point + date → proba de pousse (partie 1)
- GET  /map            → PNG heatmap (date + espèces sélectionnées)

Lancer :
    poetry run uvicorn champiquest.api:app --host 0.0.0.0 --port 8000
"""

import base64
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from champiquest.vision.predict import Predictor
from champiquest.growth import SPECIES, predict_growth

app = FastAPI(
    title="ChampiQuest",
    description="Prédiction de pousse (partie 1) + reconnaissance d'espèce (partie 2).",
    version="0.1.0",
)

MAPS_DIR = Path(__file__).resolve().parents[2] / "data" / "maps"
BBOX = (47.3, 49.3, 6.4, 8.7)  # (min_lat, max_lat, min_lon, max_lon)
STATIC_DIR = Path(__file__).resolve().parents[0] / "static"  # src/champiquest/static

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Schémas (pydantic : valide les entrées, documente l'OpenAPI) ---


class IdentifyRequest(BaseModel):
    images: list[str] = Field(..., min_length=1, description="1..N photos du même spécimen, encodées en base64")
    views: list[str] | None = Field(
        default=None, description="vues optionnelles : cap | gills | stem | whole | auto"
    )


class IdentifyResponse(BaseModel):
    status: str
    decision: str
    top_predictions: list[dict]
    confidence: float
    calibrated: bool


class GrowthRequest(BaseModel):
    lat: float = Field(..., ge=47.0, le=49.5, description="latitude (zone Vosges / Forêt-Noire)")
    lon: float = Field(..., ge=6.0, le=9.0, description="longitude")
    date: str = Field(..., description="date cible YYYY-MM-DD (aujourd'hui → J+16)")


# --- Dépendances : chargées paresseusement (le modèle torch n'est pas chargé à l'import) ---

_predictor: Predictor | None = None


def _get_predictor() -> Predictor:
    """Charge le modèle vision une seule fois, à la première requête /identify."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def _decode(im: str) -> bytes:
    """Accepte le base64 pur ou un data-URL ('data:image/jpeg;base64,…')."""
    if im.startswith("data:") and "," in im:
        im = im.split(",", 1)[1]
    return base64.b64decode(im)


def _render_map_png(date: str, species: list[str]) -> tuple[bytes, float, float]:
    """Combine les espèces (max), lisse, colorie et renvoie (PNG, vmin, vmax).

    - interpolation bicubique (×10) → heatmap lisse au lieu de gros blocs ;
    - échelle de couleur **fixe 0→1** (probabilité absolue, comparable entre
      espèces/dates) ;
    - alpha progressif : les zones de faible proba laissent voir le terrain.
    """
    npz_path = MAPS_DIR / date / "probs.npz"
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail=f"pas de carte pour {date}")
    npz = np.load(npz_path)
    arrays = [npz[sp] for sp in species if sp in npz.files]
    if not arrays:
        raise HTTPException(status_code=400, detail="aucune espèce valide")

    combined = np.nanmax(np.asarray(arrays), axis=0)
    valid = np.isfinite(combined)
    if not valid.any():
        raise HTTPException(status_code=404, detail="aucune donnée valide pour cette date")
    if np.nanmax(combined) <= 1e-6:
        raise HTTPException(
            status_code=404, detail="hors saison : aucune probabilité de pousse pour cette sélection"
        )

    # remplit les trous par la moyenne des voisins pour éviter les stries au lissage
    filled = combined.copy()
    if (~valid).any() and valid.any():
        from scipy.ndimage import distance_transform_edt

        _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
        filled = combined[iy, ix]

    smooth = gaussian_filter(filled, sigma=1.0)
    up = zoom(smooth, 10, order=3)  # ×10 → grille ~380×430 px, interpolation bicubique
    up = np.clip(up, 0.0, 1.0)

    # échelle fixe 0→1 : la couleur encode la probabilité absolue (0.25 = faible, 0.8 = fort)
    vmin, vmax = 0.0, 1.0
    norm = np.clip(up, 0.0, 1.0)

    cmap = matplotlib.colormaps["turbo"]
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    # alpha : transparent sous vmin, franchement visible en haut
    alpha = (norm**0.7 * 200 + 30).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    buf = BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue(), vmin, vmax


# --- Endpoints ---


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    """Frontend Leaflet (carte interactive)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/demo")
def demo_page() -> FileResponse:
    """Page de démo : upload d'une photo → classification (modèle vision)."""
    return FileResponse(STATIC_DIR / "identify.html")


@app.post("/identify", response_model=IdentifyResponse)
def identify(req: IdentifyRequest) -> IdentifyResponse:
    images = [_decode(im) for im in req.images]
    result = _get_predictor().identify(images, req.views)
    return IdentifyResponse(**result)


@app.post("/growth/predict")
def growth_predict(req: GrowthRequest) -> dict:
    probs = predict_growth(req.lat, req.lon, req.date)
    return {"date": req.date, "predictions": probs}


@app.get("/map")
def map_heatmap(date: str, species: str | None = None) -> Response:
    sp_list = [s.strip() for s in species.split(",")] if species else SPECIES
    png, vmin, vmax = _render_map_png(date, sp_list)
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Vmin": f"{vmin:.4f}", "X-Vmax": f"{vmax:.4f}"},
    )


@app.get("/map/meta")
def map_meta(date: str, species: str | None = None) -> dict:
    """Échelle de couleur utilisée (pour dessiner la légende côté frontend)."""
    sp_list = [s.strip() for s in species.split(",")] if species else SPECIES
    _, vmin, vmax = _render_map_png(date, sp_list)
    return {"date": date, "species": sp_list, "vmin": vmin, "vmax": vmax}


@app.get("/species")
def species_list() -> dict:
    """Liste des espèces disponibles (clé technique + nom commun)."""
    return {
        "species": [
            {"key": s["name"], "common_name": s["common_name"]} for s in _config_species()
        ]
    }


def _config_species() -> list[dict]:
    """Espèces de la partie 1 lues depuis configs/species.yaml."""
    import yaml

    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "configs" / "species.yaml"))
    by_key = {s["key"]: s for s in cfg["species"]}
    return [{"name": k, "common_name": by_key.get(k, {}).get("scientific", k)} for k in SPECIES]
