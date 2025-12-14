"""
Configuration pytest
====================
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def sample_texts():
    """Textes d'exemple pour les tests."""
    return [
        "Patient de 65 ans, douleur thoracique depuis 2h, EVA 7/10",
        "Femme 45 ans, dyspnée progressive, SpO2 94%",
        "Homme 30 ans, traumatisme cheville suite chute",
        "Patient 72 ans, confusion, désorientation",
        "Enfant 8 ans, fièvre 39.5°C depuis 24h"
    ]


@pytest.fixture(scope="session")
def sample_numerical():
    """Features numériques d'exemple."""
    return np.array([
        [65, 95, 140, 85, 96, 37.2, 7, 0.68, 0, 1],
        [45, 88, 130, 80, 94, 37.5, 3, 0.68, 0, 0],
        [30, 75, 120, 75, 98, 36.8, 5, 0.63, 0, 1],
        [72, 82, 135, 78, 95, 36.5, 2, 0.61, 0, 1],
        [8, 110, 95, 60, 97, 39.5, 4, 1.16, 0, 0]
    ])


@pytest.fixture(scope="session")
def sample_labels():
    """Labels d'exemple."""
    return np.array([1, 2, 3, 2, 2])


@pytest.fixture(scope="session")
def feature_names():
    """Noms des features."""
    return ['Age', 'FC', 'PAS', 'PAD', 'SpO2', 'Temperature',
            'EVA', 'ShockIndex', 'O2', 'Sexe_num']


@pytest.fixture
def simulation_config():
    """Configuration de simulation pour tests."""
    from src.simulation.config import SimulationConfig
    return SimulationConfig(duree_jours=1, random_seed=42)


@pytest.fixture
def patient_generator(simulation_config):
    """Générateur de patients pour tests."""
    from src.simulation.patient_generator import GenerateurPatientsVirtuels
    return GenerateurPatientsVirtuels(simulation_config, seed=42)


# Markers
def pytest_configure(config):
    """Configure les markers pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "gpu: requires GPU")
