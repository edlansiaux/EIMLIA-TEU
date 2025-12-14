"""
Modèles d'IA pour le triage FRENCH
==================================

Trois architectures avec explicabilité intégrée:
    - TRIAGEMASTER: Doc2Vec + MLP + SHAP KernelExplainer
    - URGENTIAPARSE: FlauBERT + XGBoost + SHAP TreeExplainer + Attention
    - EMERGINET: JEPA + VICReg + Integrated Gradients (réévaluation continue)

Usage:
    >>> from src.models import EMERGINET, load_data, evaluate_model
    >>> 
    >>> texts, numerical, labels, features = load_data('data.xlsx')
    >>> model = EMERGINET(epochs=50)
    >>> model.fit(texts, numerical, labels, features)
    >>> predictions = model.predict(texts_test, numerical_test)
"""

from src.models.base import BaseTriageModel, SHAPHelper
from src.models.data_loader import load_data, generate_synthetic_data
from src.models.triagemaster import TRIAGEMASTER
from src.models.urgentiaparse import URGENTIAPARSE
from src.models.emerginet import EMERGINET
from src.models.evaluation import evaluate_model, compare_models

__all__ = [
    "BaseTriageModel",
    "SHAPHelper",
    "TRIAGEMASTER",
    "URGENTIAPARSE",
    "EMERGINET",
    "load_data",
    "generate_synthetic_data",
    "evaluate_model",
    "compare_models",
]
