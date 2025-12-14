"""
Module de simulation EIMLIA
===========================

Simulation hybride SimPy (DES) + Mesa (SMA) pour modéliser
le service d'urgences et l'impact des modèles d'IA.

Composants:
    - SimulationUrgencesSimPy: Simulation DES des flux patients
    - ModeleSAU: Modèle multi-agents Mesa
    - SimulationHybride: Synchronisation SimPy + Mesa
    - OrchestrateurSimulation: Gestion des scénarios
    - GenerateurPatientsVirtuels: Génération de cohortes

Usage:
    >>> from src.simulation import OrchestrateurSimulation
    >>> 
    >>> orchestrateur = OrchestrateurSimulation(
    ...     n_patients=100_000,
    ...     duree_jours=180
    ... )
    >>> resultats = orchestrateur.executer_tous_scenarios()
"""

from src.simulation.config import SimulationConfig, DISTRIBUTIONS, RESSOURCES
from src.simulation.patient_generator import GenerateurPatientsVirtuels, PatientVirtuel
from src.simulation.simpy_des import SimulationUrgencesSimPy
from src.simulation.mesa_sma import ModeleSAU, AgentPatient, AgentIAO, AgentMedecin, AgentIA
from src.simulation.hybrid import SimulationHybride
from src.simulation.orchestrator import OrchestrateurSimulation

__all__ = [
    "SimulationConfig",
    "DISTRIBUTIONS",
    "RESSOURCES",
    "GenerateurPatientsVirtuels",
    "PatientVirtuel",
    "SimulationUrgencesSimPy",
    "ModeleSAU",
    "AgentPatient",
    "AgentIAO",
    "AgentMedecin",
    "AgentIA",
    "SimulationHybride",
    "OrchestrateurSimulation",
]
