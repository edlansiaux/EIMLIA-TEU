"""
Métriques et calculs
====================
"""

import numpy as np
from typing import Dict, Any, List


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcule les métriques de classification."""
    from sklearn.metrics import (
        accuracy_score, f1_score, cohen_kappa_score,
        mean_absolute_error, mean_squared_error
    )
    from scipy.stats import spearmanr
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_micro': float(f1_score(y_true, y_pred, average='micro')),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'kappa': float(cohen_kappa_score(y_true, y_pred, weights='quadratic')),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'spearman': float(spearmanr(y_true, y_pred)[0]),
        'exact_match': float(np.mean(y_true == y_pred)),
        'near_match': float(np.mean(np.abs(y_true - y_pred) <= 1))
    }


def format_metrics(metrics: Dict[str, float], prefix: str = '') -> str:
    """Formate les métriques pour affichage."""
    lines = []
    for key, value in metrics.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, float):
            lines.append(f"{name}: {value:.4f}")
        else:
            lines.append(f"{name}: {value}")
    return '\n'.join(lines)


def aggregate_results(results_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Agrège les résultats de plusieurs runs."""
    if not results_list:
        return {}
    
    aggregated = {}
    keys = results_list[0].keys()
    
    for key in keys:
        values = [r[key] for r in results_list if key in r]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    return aggregated
