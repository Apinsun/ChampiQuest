"""Calcul des features météo à partir de la série quotidienne d'un point.

Les features sont calculées sur la fenêtre PRÉCÉDANT l'observation (J-1 → J-window),
car on prédit la pousse à partir de la météo *passée*.
"""

import pandas as pd


def compute_weather_features(daily: pd.DataFrame, obs_date: pd.Timestamp) -> dict:
    """Calcule les features météo pour une observation (fenêtre avant J).

    Paramètres
    ----------
    daily : DataFrame indexé par date, colonnes Open-Meteo
        (``precipitation_sum``, ``temperature_2m_min/max/mean``,
        ``et0_fao_evapotranspiration``).
    obs_date : date de l'observation. On regarde les jours précédents, jamais le jour J.

    Retourne un dict de features numériques.
    """
    sub = daily.loc[: obs_date - pd.Timedelta(days=1)]

    def cumul(col: str, days: int) -> float:
        return float(sub[col].tail(days).sum())

    # Bilan hydrique (précipitations − évapotranspiration) : proxy de sécheresse
    def bilan(days: int) -> float:
        return float(
            (sub["precipitation_sum"].tail(days) - sub["et0_fao_evapotranspiration"].tail(days)).sum()
        )

    # Choc thermique : température moyenne (J-21→J-11) − (J-10→J-1).
    # Positif = refroidissement récent (déclencheur de fructification souvent cité).
    t_early = float(sub["temperature_2m_mean"].tail(21).head(11).mean())
    t_late = float(sub["temperature_2m_mean"].tail(10).mean())

    return {
        # Cumuls de précipitations (fenêtres courtes → longues)
        "precip_J7": cumul("precipitation_sum", 7),
        "precip_J14": cumul("precipitation_sum", 14),
        "precip_J30": cumul("precipitation_sum", 30),
        "precip_J60": cumul("precipitation_sum", 60),
        "precip_J90": cumul("precipitation_sum", 90),
        # Températures
        "temp_mean_J14": float(sub["temperature_2m_mean"].tail(14).mean()),
        "temp_min_J14": float(sub["temperature_2m_min"].tail(14).min()),
        "temp_max_J14": float(sub["temperature_2m_max"].tail(14).max()),
        # Dynamiques thermiques
        "choc_thermique": t_early - t_late,
        "amplitude_thermique": float(
            (sub["temperature_2m_max"] - sub["temperature_2m_min"]).tail(14).mean()
        ),
        # Bilan hydrique (mémoire de sécheresse, fenêtres croissantes)
        "bilan_hydrique_J30": bilan(30),
        "bilan_hydrique_J60": bilan(60),
        "bilan_hydrique_J90": bilan(90),
    }
