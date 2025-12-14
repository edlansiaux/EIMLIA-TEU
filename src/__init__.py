"""
EIMLIA-3M-TEU
=============

Étude d'Impact des Modèles d'IA sur le Lean management et l'Interprétabilité
du triage des Admissions aux urgences - 3 Modèles - Triage des Entrées aux Urgences

Simulation prospective comparant 3 modèles d'IA pour le triage aux urgences (FRENCH)
sur 600 000 patients virtuels.

Modules:
    - models: Les 3 modèles d'IA (TRIAGEMASTER, URGENTIAPARSE, EMERGINET)
    - simulation: Simulations SimPy (DES) et Mesa (SMA)
    - process_mining: Pipeline PM4Py
    - utils: Utilitaires (logging, métriques, visualisation)
    - api: API FastAPI

Usage:
    >>> from src.models import EMERGINET
    >>> from src.simulation import OrchestrateurSimulation
    >>> 
    >>> model = EMERGINET(epochs=50)
    >>> orchestrateur = OrchestrateurSimulation(n_patients=100_000)
"""

__version__ = "1.0.0"
__author__ = "Équipe EIMLIA - CHU de Lille"
__email__ = "eimlia@chu-lille.fr"
__license__ = "MIT"

from src.models import TRIAGEMASTER, URGENTIAPARSE, EMERGINET
from src.simulation import OrchestrateurSimulation, SimulationHybride

__all__ = [
    "TRIAGEMASTER",
    "URGENTIAPARSE", 
    "EMERGINET",
    "OrchestrateurSimulation",
    "SimulationHybride",
    "__version__",
]
