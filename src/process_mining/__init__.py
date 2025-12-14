"""
Module Process Mining avec PM4Py
================================

Remplace Celonis pour l'analyse des processus.

Fonctionnalités:
    - Découverte de processus (Alpha Miner, Inductive Miner)
    - Conformance checking
    - Analyse de performance
    - Extraction de KPIs

Usage:
    >>> from src.process_mining import ProcessMiningPipeline
    >>> 
    >>> pipeline = ProcessMiningPipeline('event_log.csv')
    >>> kpis = pipeline.compute_kpis()
"""

from src.process_mining.pipeline import ProcessMiningPipeline
from src.process_mining.discovery import discover_alpha, discover_inductive, discover_heuristic
from src.process_mining.conformance import check_conformance, compute_fitness

__all__ = [
    "ProcessMiningPipeline",
    "discover_alpha",
    "discover_inductive", 
    "discover_heuristic",
    "check_conformance",
    "compute_fitness",
]
