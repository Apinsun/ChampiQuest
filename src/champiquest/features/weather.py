"""Calcul des features météo à partir de la série quotidienne d'un point.

Les features sont calculées sur la fenêtre PRÉCÉDANT l'observation (J-1 → J-window),
car on prédit la pousse à partir de la météo *passée*.

Version v1 (baseline) : cumuls de précipitations, températures et bilan hydrique.
Les features plus fines (choc thermique, classes été sec/pluvieux…) viendront ensuite.
"""

import pandas as pd


def compute_weather_features(daily: pd.DataFrame, obs_date: pd.Timestamp) -> dict:
    """Calcule les features météo de base pour une observation.

    Paramètres
    ----------
    daily : DataFrame indexé par date, colonnes = variables Open-Meteo
        (``precipitation_sum``, ``temperature_2m_min/max/mean``,
        ``et0_fao_evapotranspiration``).
    obs_date : date de l'observation. On regarde les jours **précédents**
        (jusqu'à J-1), jamais le jour J (on prédit avant d'aller sur le terrain).

    Retourne un dict de features numériques.
    """
    sub = daily.loc[: obs_date - pd.Timedelta(days=1)]

    def cumul(col: str, days: int) -> float:
        return float(sub[col].tail(days).sum())

    return {
        "precip_J7": cumul("precipitation_sum", 7),
        "precip_J14": cumul("precipitation_sum", 14),
        "precip_J30": cumul("precipitation_sum", 30),
        "temp_mean_J14": float(sub["temperature_2m_mean"].tail(14).mean()),
        "temp_min_J14": float(sub["temperature_2m_min"].tail(14).min()),
        "temp_max_J14": float(sub["temperature_2m_max"].tail(14).max()),
        # Bilan hydrique 30 j : proxy de sécheresse (précipitations − évapotranspiration)
        "bilan_hydrique_J30": float(
            (sub["precipitation_sum"].tail(30) - sub["et0_fao_evapotranspiration"].tail(30)).sum()
        ),
    }
