"""Tests du store météo Parquet : déduplication (era5 > forecast) + drop_source.

Pourquoi tester ça : c'est l'intégrité des données météo. Un bug ici corrompt
silencieusement les cartes (priorité de source, écrasement des jours).
"""
import pandas as pd

from champiquest import weather_store as ws


def _row(**overrides) -> pd.DataFrame:
    row = {
        "lat": 48.0,
        "lon": 7.0,
        "date": "2024-10-01",
        "source": "forecast",
        "precipitation_sum": 1.0,
        "temperature_2m_min": 5.0,
        "temperature_2m_max": 15.0,
        "temperature_2m_mean": 10.0,
        "et0_fao_evapotranspiration": 2.0,
        "soil_temperature_0_to_7cm_mean": 12.0,
        "soil_moisture_0_to_7cm_mean": 0.3,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_era5_ecrase_forecast_sur_le_meme_jour(monkeypatch, tmp_path):
    monkeypatch.setattr(ws, "STORE_PATH", tmp_path / "store.parquet")
    ws.append(_row(source="forecast", precipitation_sum=1.0))
    ws.append(_row(source="era5", precipitation_sum=9.9))
    df = ws.load()
    assert len(df) == 1
    assert df.iloc[0]["source"] == "era5"
    assert df.iloc[0]["precipitation_sum"] == 9.9


def test_append_garde_les_dates_distinctes(monkeypatch, tmp_path):
    monkeypatch.setattr(ws, "STORE_PATH", tmp_path / "store.parquet")
    ws.append(_row(date="2024-10-01"))
    ws.append(_row(date="2024-10-02"))
    assert len(ws.load()) == 2


def test_drop_source_ne_supprime_que_la_source_demandee(monkeypatch, tmp_path):
    monkeypatch.setattr(ws, "STORE_PATH", tmp_path / "store.parquet")
    ws.append(_row(source="forecast", date="2024-10-01"))
    ws.append(_row(source="era5", date="2024-10-02"))
    ws.drop_source("forecast")
    df = ws.load()
    assert len(df) == 1
    assert df.iloc[0]["source"] == "era5"
