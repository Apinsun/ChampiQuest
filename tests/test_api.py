"""Tests des endpoints API avec modèles mockés (aucun chargement torch réel).

Pourquoi mocker : on teste la LOGIQUE des endpoints (validation, codes HTTP,
format de réponse) sans charger le .pt de 350 Mo ni faire tourner le GPU.
"""
import base64
from io import BytesIO

from PIL import Image
from fastapi.testclient import TestClient

import champiquest.api as api

client = TestClient(api.app)


def _fake_png() -> bytes:
    buf = BytesIO()
    Image.new("RGBA", (4, 4)).save(buf, "PNG")
    return buf.getvalue()


class _FakePredictor:
    def identify(self, images, views=None):
        return {
            "status": "ok",
            "decision": "edible",
            "top_predictions": [
                {
                    "species": "Hydnum repandum",
                    "common_name": "Pied de mouton",
                    "probability": 0.9,
                    "edible": True,
                }
            ],
            "confidence": 0.9,
            "calibrated": True,
        }


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_identify(monkeypatch):
    monkeypatch.setattr(api, "_get_predictor", lambda: _FakePredictor())
    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    r = client.post("/identify", json={"images": [b64]})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "edible"
    assert body["top_predictions"][0]["species"] == "Hydnum repandum"


def test_growth_predict(monkeypatch):
    monkeypatch.setattr(api, "predict_growth", lambda lat, lon, date: {"boletus_edulis": 0.5})
    r = client.post("/growth/predict", json={"lat": 48.5, "lon": 7.5, "date": "2024-10-01"})
    assert r.status_code == 200
    assert r.json()["predictions"]["boletus_edulis"] == 0.5


def test_growth_predict_valide_la_zone(monkeypatch):
    # latitude hors de la zone (47.0–49.5) → 422 (validation pydantic)
    r = client.post("/growth/predict", json={"lat": 50.0, "lon": 7.5, "date": "2024-10-01"})
    assert r.status_code == 422


def test_species_contient_les_nouvelles():
    r = client.get("/species")
    assert r.status_code == 200
    keys = [s["key"] for s in r.json()["species"]]
    assert "craterellus_cornucopioides" in keys
    assert "hydnum_repandum" in keys


def test_map(monkeypatch):
    monkeypatch.setattr(api, "_render_map_png", lambda date, species: (_fake_png(), 0.0, 1.0))
    r = client.get("/map", params={"date": "2024-10-01"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
