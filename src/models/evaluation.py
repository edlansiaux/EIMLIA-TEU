"""
Évaluation des modèles de triage
================================

Métriques et comparaison des performances.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import (
    f1_score, cohen_kappa_score, mean_absolute_error,
    mean_squared_error, classification_report, confusion_matrix
)
from scipy.stats import spearmanr


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Évalue un modèle avec les métriques du protocole EIMLIA.
    
    Métriques calculées:
        - MAE: Mean Absolute Error (erreur moyenne en niveaux)
        - RMSE: Root Mean Square Error
        - Kappa: Cohen's Kappa pondéré (accord inter-annotateur)
        - Spearman: Corrélation de rang
        - F1_micro/macro: F1-scores
        - Exact: % de prédictions exactes
        - Near: % de prédictions à ±1 classe
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions
        model_name: Nom du modèle pour affichage
        verbose: Afficher les résultats
        
    Returns:
        Dict des métriques
    """
    results = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'Kappa': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'Spearman': spearmanr(y_true, y_pred)[0],
        'F1_micro': f1_score(y_true, y_pred, average='micro'),
        'F1_macro': f1_score(y_true, y_pred, average='macro'),
        'Exact': float(np.mean(y_true == y_pred)),
        'Near': float(np.mean(np.abs(y_true - y_pred) <= 1))
    }
    
    # Taux d'erreur par type
    diff = y_pred - y_true
    results['Sous_triage'] = float(np.mean(diff < 0))  # Prédit moins grave
    results['Sur_triage'] = float(np.mean(diff > 0))   # Prédit plus grave
    
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"📊 RÉSULTATS: {model_name}")
        print('=' * 50)
        for k, v in results.items():
            print(f"  {k:12}: {v:.4f}")
        
        # Classification report détaillé
        print(f"\n  Classification Report:")
        print(classification_report(y_true, y_pred, digits=3))
    
    return results


def compare_models(
    results: Dict[str, Dict[str, float]],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare plusieurs modèles et calcule un Z-score composite.
    
    Le Z-score combine:
        - MAE, RMSE (plus bas = meilleur, donc inversé)
        - Kappa, Spearman (plus haut = meilleur)
    
    Args:
        results: Dict {model_name: metrics_dict}
        verbose: Afficher la comparaison
        
    Returns:
        DataFrame avec toutes les métriques et le Z-score
    """
    df = pd.DataFrame(results).T
    
    # Calculer Z-scores
    z_scores = {}
    for metric in ['MAE', 'RMSE', 'Kappa', 'Spearman']:
        if metric in df.columns:
            values = df[metric].values
            z = (values - values.mean()) / (values.std() + 1e-6)
            # Inverser pour MAE/RMSE (plus bas = meilleur)
            if metric in ['MAE', 'RMSE']:
                z = -z
            z_scores[metric] = z
    
    df['Z_composite'] = sum(z_scores.values())
    
    if verbose:
        print("\n" + "=" * 70)
        print("📊 COMPARAISON FINALE DES MODÈLES")
        print("=" * 70)
        
        # Afficher tableau formaté
        display_cols = ['MAE', 'RMSE', 'Kappa', 'Spearman', 'Exact', 'Near', 'Z_composite']
        display_cols = [c for c in display_cols if c in df.columns]
        print(df[display_cols].round(4).to_string())
        
        # Ranking
        print("\n🏆 RANKING (Z-score composite):")
        ranking = df['Z_composite'].sort_values(ascending=False)
        for i, (model, score) in enumerate(ranking.items(), 1):
            medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
            print(f"  {medal} {model}: {score:.3f}")
    
    return df


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Calcule et formate la matrice de confusion.
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions
        class_names: Noms des classes
        normalize: Normaliser par ligne
        
    Returns:
        DataFrame de la matrice de confusion
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    if class_names is None:
        class_names = [f"Classe {i}" for i in range(cm.shape[0])]
    
    return pd.DataFrame(
        cm,
        index=[f"Réel: {c}" for c in class_names],
        columns=[f"Prédit: {c}" for c in class_names]
    )


def compute_triage_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    critical_classes: List[int] = None
) -> Dict[str, float]:
    """
    Calcule les métriques spécifiques au triage médical.
    
    Args:
        y_true: Labels réels (niveaux de gravité)
        y_pred: Prédictions
        critical_classes: Classes considérées comme critiques (défaut: [3, 4] = FRENCH 4-5)
        
    Returns:
        Dict avec métriques triage:
            - concordance: % accord exact
            - sous_triage: % de patients sous-classés
            - sur_triage: % de patients sur-classés
            - sous_triage_critique: % de patients critiques sous-classés (dangereux)
            - sensibilite_critique: sensibilité pour détecter les cas critiques
            - specificite_critique: spécificité pour les cas critiques
    """
    if critical_classes is None:
        critical_classes = [2, 3]  # Indices 0-based pour CCMU 3-4
    
    diff = y_pred - y_true
    
    # Métriques de base
    concordance = float(np.mean(y_true == y_pred))
    sous_triage = float(np.mean(diff < 0))
    sur_triage = float(np.mean(diff > 0))
    
    # Sous-triage critique (patients graves classés moins graves)
    is_critical_real = np.isin(y_true, critical_classes)
    sous_triage_critique = 0.0
    if is_critical_real.sum() > 0:
        sous_triage_critique = float(np.mean(diff[is_critical_real] < 0))
    
    # Sensibilité/Spécificité pour cas critiques
    is_critical_pred = np.isin(y_pred, critical_classes)
    
    # True positives (critiques détectés)
    tp = np.sum(is_critical_real & is_critical_pred)
    # False negatives (critiques manqués)
    fn = np.sum(is_critical_real & ~is_critical_pred)
    # False positives (non-critiques classés critiques)
    fp = np.sum(~is_critical_real & is_critical_pred)
    # True negatives
    tn = np.sum(~is_critical_real & ~is_critical_pred)
    
    sensibilite = tp / (tp + fn + 1e-6)
    specificite = tn / (tn + fp + 1e-6)
    
    return {
        'concordance': concordance,
        'sous_triage': sous_triage,
        'sur_triage': sur_triage,
        'sous_triage_critique': sous_triage_critique,
        'sensibilite_critique': float(sensibilite),
        'specificite_critique': float(specificite)
    }


def generate_report(
    results: Dict[str, Dict[str, float]],
    output_path: str = None
) -> str:
    """
    Génère un rapport textuel complet de comparaison.
    
    Args:
        results: Dict {model_name: metrics_dict}
        output_path: Chemin optionnel pour sauvegarder
        
    Returns:
        Rapport formaté en texte
    """
    lines = []
    lines.append("=" * 70)
    lines.append("RAPPORT D'ÉVALUATION - ÉTUDE EIMLIA-3M-TEU")
    lines.append("=" * 70)
    lines.append("")
    
    # Résumé par modèle
    for model_name, metrics in results.items():
        lines.append(f"\n{'─' * 50}")
        lines.append(f"📊 {model_name}")
        lines.append('─' * 50)
        
        for metric, value in metrics.items():
            lines.append(f"  {metric:20}: {value:.4f}")
    
    # Comparaison
    df = compare_models(results, verbose=False)
    
    lines.append(f"\n{'=' * 70}")
    lines.append("CLASSEMENT FINAL")
    lines.append('=' * 70)
    
    ranking = df['Z_composite'].sort_values(ascending=False)
    for i, (model, score) in enumerate(ranking.items(), 1):
        lines.append(f"  {i}. {model}: Z={score:.3f}")
    
    # Recommandation
    best_model = ranking.index[0]
    lines.append(f"\n✅ Recommandation: {best_model}")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Rapport sauvegardé: {output_path}")
    
    return report
