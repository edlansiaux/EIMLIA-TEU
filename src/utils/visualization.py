"""
Visualisation des résultats
===========================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path


def plot_results(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualise les résultats d'une simulation.
    
    Args:
        results: Dict des résultats
        output_path: Chemin de sauvegarde
        show: Afficher le plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distribution DMS
    if 'dms_list' in results:
        axes[0, 0].hist(results['dms_list'], bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('DMS (minutes)')
        axes[0, 0].set_ylabel('Fréquence')
        axes[0, 0].set_title('Distribution de la Durée Moyenne de Séjour')
    
    # Métriques clés
    metrics = ['dms_median', 'attente_triage_mean', 'concordance_ia']
    values = [results.get(m, 0) or 0 for m in metrics]
    axes[0, 1].bar(metrics, values)
    axes[0, 1].set_ylabel('Valeur')
    axes[0, 1].set_title('Métriques Clés')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    metrics: List[str] = None,
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Compare visuellement plusieurs scénarios.
    
    Args:
        results_dict: {scenario_name: results}
        metrics: Métriques à comparer
        output_path: Chemin de sauvegarde
        show: Afficher le plot
    """
    import matplotlib.pyplot as plt
    
    if metrics is None:
        metrics = ['dms_median', 'attente_triage_mean', 'sous_triage']
    
    scenarios = list(results_dict.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = []
        for scenario in scenarios:
            res = results_dict[scenario].get('resultats_principaux', results_dict[scenario])
            values.append(res.get(metric, 0) or 0)
        
        bars = ax.bar(scenarios, values)
        ax.set_ylabel(metric)
        ax.set_title(f'Comparaison: {metric}')
        ax.tick_params(axis='x', rotation=45)
        
        # Colorer le meilleur
        if metric in ['dms_median', 'attente_triage_mean', 'sous_triage']:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_color('green')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_summary_table(
    results_dict: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Crée un tableau récapitulatif des résultats.
    
    Args:
        results_dict: {scenario_name: results}
        
    Returns:
        DataFrame récapitulatif
    """
    rows = []
    
    for scenario, results in results_dict.items():
        res = results.get('resultats_principaux', results)
        
        row = {
            'Scénario': scenario,
            'Patients': res.get('n_patients', 0),
            'DMS médiane (min)': res.get('dms_median', 0),
            'Attente triage (min)': res.get('attente_triage_mean', 0),
            'Concordance IA': res.get('concordance_ia'),
            'Sous-triage': res.get('sous_triage', 0),
            'Sur-triage': res.get('sur_triage', 0)
        }
        rows.append(row)
    
    return pd.DataFrame(rows)
