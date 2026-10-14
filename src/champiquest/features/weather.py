"""Calcul des features météo à partir de la série quotidienne d'un point.

Les features sont calculées sur la fenêtre PRÉCÉDANT l'observation (J-1 → J-window),
car on prédit la pousse à partir de la météo *passée*.
"""

import numpy as np
import pandas as pd

SEUIL_GROSSE_PLUIE = 10.0  # mm/jour : ce qu'on considère comme un "gros" épisode de pluie


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

    # Bilan hydrique (précipitations − évapotranspiration) : proxy d'humidité du sol
    def bilan(days: int) -> float:
        return float(
            (sub["precipitation_sum"].tail(days) - sub["et0_fao_evapotranspiration"].tail(days)).sum()
        )

    # Choc thermique : température moyenne (J-21→J-11) − (J-10→J-1).
    t_early = float(sub["temperature_2m_mean"].tail(21).head(11).mean())
    t_late = float(sub["temperature_2m_mean"].tail(10).mean())

    # Tendance de température (pente sur 30 j, en °C/mois) : positive = réchauffement.
    t30 = sub["temperature_2m_mean"].tail(30).to_numpy(dtype=float)
    tendance = float(np.polyfit(np.arange(len(t30)), t30, 1)[0] * 30) if len(t30) >= 5 else 0.0

    # Dernier gros épisode de pluie (≥ seuil) : jours écoulés depuis.
    precip30 = sub["precipitation_sum"].tail(30)
    gros = precip30[precip30 >= SEUIL_GROSSE_PLUIE]
    jours_depuis = float((sub.index.max() - gros.index.max()).days) if len(gros) else 30.0

    return {
        # Cumuls de précipitations
        "precip_J7": cumul("precipitation_sum", 7),
        "precip_J14": cumul("precipitation_sum", 14),
        "precip_J30": cumul("precipitation_sum", 30),
        "precip_J60": cumul("precipitation_sum", 60),
        "precip_J90": cumul("precipitation_sum", 90),
        # Événements pluvieux
        "pluie_max_J14": float(sub["precipitation_sum"].tail(14).max()),
        "jours_depuis_grosse_pluie": jours_depuis,
        # Températures
        "temp_mean_J14": float(sub["temperature_2m_mean"].tail(14).mean()),
        "temp_min_J14": float(sub["temperature_2m_min"].tail(14).min()),
        "temp_max_J14": float(sub["temperature_2m_max"].tail(14).max()),
        "gel_J14": float((sub["temperature_2m_min"].tail(14) < 0).sum()),
        # Dynamiques thermiques
        "choc_thermique": t_early - t_late,  # positif = refroidissement (conservé pour comparaison)
        "tendance_temp": tendance,           # positif = réchauffement
        "amplitude_thermique": float(
            (sub["temperature_2m_max"] - sub["temperature_2m_min"]).tail(14).mean()
        ),
        # Bilan hydrique (mémoire de sécheresse, fenêtres croissantes)
        "bilan_hydrique_J30": bilan(30),
        "bilan_hydrique_J60": bilan(60),
        "bilan_hydrique_J90": bilan(90),
        "bilan_hydrique_J120": bilan(120),
    }


def compute_soil_features(daily: pd.DataFrame, obs_date: pd.Timestamp) -> dict:
    """Calcule les features de température de surface et humidité du sol (NASA POWER).

    Paramètres
    ----------
    daily : DataFrame indexé par date, colonnes NASA POWER (``ts``, ``gwet_top``).
    obs_date : date de l'observation (on regarde les jours précédents, jamais le jour J).

    Retourne un dict de features numériques.
    """
    sub = daily.loc[: obs_date - pd.Timedelta(days=1)]

    ts = sub["ts"]
    gwet = sub["gwet_top"]

    # Réchauffement du sol : température (J-7) − (J-30). Positif = le sol se
    # réchauffe (déclencheur printanier de la morille), négatif = il refroidit.
    ts7 = float(ts.tail(7).mean())
    ts30 = float(ts.tail(30).mean())

    return {
        "temp_sol_J14": float(ts.tail(14).mean()),
        "rechauffement_sol": ts7 - ts30,
        "humidite_sol_J30": float(gwet.tail(30).mean()),
    }
