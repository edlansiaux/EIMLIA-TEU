"""
Configuration de la simulation
==============================

Paramètres calibrés depuis les données réelles du CHU de Lille (via PM4Py).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

# Graine aléatoire globale
RANDOM_SEED = 42


@dataclass
class SimulationConfig:
    """Configuration complète de la simulation."""
    
    # Durée
    duree_jours: int = 180
    
    # Ressources (capacités)
    nb_box_triage: int = 3
    nb_box_consultation: int = 8
    nb_lits_uhcd: int = 12
    nb_scanners: int = 2
    nb_radios: int = 3
    nb_iao: int = 3
    nb_medecins: int = 8
    
    # Taux d'arrivée (patients/heure par tranche horaire)
    # Calibré depuis les données PMSI
    lambda_arrivees: List[float] = field(default_factory=lambda: [
        # 00h-23h (24 valeurs)
        2.1, 1.8, 1.5, 1.2, 1.0, 1.2,   # 00-05h (nuit)
        1.8, 3.2, 4.5, 5.8, 6.2, 6.5,   # 06-11h (matin)
        6.8, 6.5, 6.2, 5.8, 5.5, 5.2,   # 12-17h (après-midi)
        5.0, 4.5, 4.0, 3.5, 3.0, 2.5    # 18-23h (soir)
    ])
    
    # Distribution des niveaux FRENCH (%)
    distribution_french: List[float] = field(default_factory=lambda: [
        0.05,  # FRENCH 1 (très grave)
        0.15,  # FRENCH 2
        0.45,  # FRENCH 3
        0.30,  # FRENCH 4
        0.05   # FRENCH 5 (peu grave)
    ])
    
    # Distribution des parcours par niveau FRENCH
    # {niveau: {type_parcours: probabilité}}
    parcours_par_niveau: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        1: {'dechocage': 0.8, 'uhcd': 0.15, 'standard': 0.05},
        2: {'uhcd': 0.5, 'imagerie': 0.3, 'standard': 0.2},
        3: {'imagerie': 0.4, 'standard': 0.5, 'sortie_rapide': 0.1},
        4: {'standard': 0.6, 'sortie_rapide': 0.35, 'imagerie': 0.05},
        5: {'sortie_rapide': 0.8, 'standard': 0.2}
    })
    
    # Agent IA
    agent_ia: Optional[str] = None  # 'triagemaster', 'urgentiaparse', 'emerginet'
    taux_utilisation_ia: float = 1.0  # % des patients évalués par IA
    
    # Stress-tests
    facteur_charge: float = 1.0  # Multiplicateur du lambda
    
    # Seed
    random_seed: int = RANDOM_SEED


# Distributions temporelles (calibrées depuis PM4Py)
# Format: (type, params) où type = 'lognormal', 'exponential', etc.
DISTRIBUTIONS = {
    # Durées en minutes
    'triage': ('lognormal', {'mean': 1.9, 'sigma': 0.6}),  # ~8 min médiane
    'consultation': ('lognormal', {'mean': 3.1, 'sigma': 0.8}),  # ~22 min médiane
    'scanner': ('lognormal', {'mean': 2.7, 'sigma': 0.5}),  # ~15 min médiane
    'radio': ('lognormal', {'mean': 2.3, 'sigma': 0.4}),  # ~10 min médiane
    'biologie': ('lognormal', {'mean': 3.4, 'sigma': 0.6}),  # ~30 min médiane
    'avis_specialise': ('lognormal', {'mean': 3.9, 'sigma': 0.8}),  # ~50 min médiane
    'uhcd': ('lognormal', {'mean': 5.5, 'sigma': 0.9}),  # ~4h médiane
    'dechocage': ('lognormal', {'mean': 4.6, 'sigma': 1.0}),  # ~1.5h médiane
    
    # Latences IA en ms
    'latence_nlp': ('lognormal', {'mean': 4.8, 'sigma': 0.3}),  # ~120 ms
    'latence_llm': ('lognormal', {'mean': 5.9, 'sigma': 0.4}),  # ~380 ms
    'latence_jepa': ('lognormal', {'mean': 5.5, 'sigma': 0.3}),  # ~240 ms
}


# Capacités des ressources
RESSOURCES = {
    'box_triage': 3,
    'box_consultation': 8,
    'lits_uhcd': 12,
    'scanners': 2,
    'radios': 3,
    'iao': 3,
    'medecins': 8,
}


# Taux d'erreur des modèles IA (depuis les résultats d'entraînement)
TAUX_ERREUR_IA = {
    'triagemaster': 0.39,   # ~39% erreur
    'urgentiaparse': 0.25,  # ~25% erreur  
    'emerginet': 0.10,      # ~10% erreur
}


# Paramètres comportementaux des IAO
COMPORTEMENT_IAO = {
    'taux_acceptation_base': 0.85,
    'bonus_concordance_totale': 0.10,
    'malus_ecart_1_niveau': 0.05,
    'malus_ecart_2_niveaux': 0.30,
    'malus_experience_5ans': 0.05,
    'malus_charge_10patients': 0.05,
    'bonus_explication': 0.05,
    'bonus_alerte': 0.10,
    'fatigue_increment': 0.01,  # Par heure
    'fatigue_max': 0.3,
}


def sample_duration(distribution_name: str, random_state: np.random.RandomState = None) -> float:
    """
    Échantillonne une durée depuis une distribution calibrée.
    
    Args:
        distribution_name: Nom de la distribution dans DISTRIBUTIONS
        random_state: Générateur aléatoire optionnel
        
    Returns:
        Durée en minutes
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    dist_type, params = DISTRIBUTIONS[distribution_name]
    
    if dist_type == 'lognormal':
        return float(random_state.lognormal(params['mean'], params['sigma']))
    elif dist_type == 'exponential':
        return float(random_state.exponential(params['scale']))
    elif dist_type == 'normal':
        return float(max(0, random_state.normal(params['mean'], params['std'])))
    else:
        raise ValueError(f"Distribution inconnue: {dist_type}")


def get_lambda_for_hour(hour: int, config: SimulationConfig = None) -> float:
    """
    Retourne le taux d'arrivée (lambda) pour une heure donnée.
    
    Args:
        hour: Heure (0-23)
        config: Configuration optionnelle
        
    Returns:
        Lambda (patients/heure)
    """
    if config is None:
        config = SimulationConfig()
    
    return config.lambda_arrivees[hour % 24] * config.facteur_charge
