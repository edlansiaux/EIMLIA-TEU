"""
API FastAPI EIMLIA
==================

API REST pour les prédictions de triage et la gestion des simulations.
"""

from src.api.main import app, create_app

__all__ = ["app", "create_app"]
