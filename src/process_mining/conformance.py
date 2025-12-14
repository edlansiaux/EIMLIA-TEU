"""
Conformance Checking
====================

Vérification de la conformance des traces au modèle de processus.
"""

from typing import Any, Dict


def check_conformance(
    event_log: Any,
    model: Dict[str, Any],
    algorithm: str = 'token_replay'
) -> Dict[str, Any]:
    """
    Vérifie la conformance d'un event log par rapport à un modèle.
    
    Args:
        event_log: Event log PM4Py
        model: Dict avec net, initial_marking, final_marking
        algorithm: 'token_replay' ou 'alignments'
        
    Returns:
        Dict avec métriques de conformance
    """
    import pm4py
    
    net = model['net']
    im = model['initial_marking']
    fm = model['final_marking']
    
    if algorithm == 'token_replay':
        result = pm4py.conformance_diagnostics_token_based_replay(
            event_log, net, im, fm
        )
    else:
        result = pm4py.conformance_diagnostics_alignments(
            event_log, net, im, fm
        )
    
    return result


def compute_fitness(
    event_log: Any,
    model: Dict[str, Any],
    algorithm: str = 'token_replay'
) -> Dict[str, float]:
    """
    Calcule le fitness d'un modèle par rapport à un event log.
    
    Le fitness mesure à quel point le modèle peut rejouer les traces du log.
    
    Args:
        event_log: Event log PM4Py
        model: Dict avec net, initial_marking, final_marking
        algorithm: 'token_replay' ou 'alignments'
        
    Returns:
        Dict avec average_trace_fitness, percentage_of_fitting_traces, etc.
    """
    import pm4py
    
    net = model['net']
    im = model['initial_marking']
    fm = model['final_marking']
    
    if algorithm == 'token_replay':
        fitness = pm4py.fitness_token_based_replay(event_log, net, im, fm)
    else:
        fitness = pm4py.fitness_alignments(event_log, net, im, fm)
    
    return fitness


def compute_precision(
    event_log: Any,
    model: Dict[str, Any]
) -> float:
    """
    Calcule la précision d'un modèle.
    
    La précision mesure à quel point le modèle est restrictif
    (évite de permettre trop de comportements non observés).
    
    Args:
        event_log: Event log PM4Py
        model: Dict avec net, initial_marking, final_marking
        
    Returns:
        Score de précision (0-1)
    """
    import pm4py
    
    net = model['net']
    im = model['initial_marking']
    fm = model['final_marking']
    
    return pm4py.precision_token_based_replay(event_log, net, im, fm)


def compute_generalization(
    event_log: Any,
    model: Dict[str, Any]
) -> float:
    """
    Calcule la généralisation d'un modèle.
    
    La généralisation mesure à quel point le modèle n'est pas
    sur-ajusté aux données observées.
    
    Args:
        event_log: Event log PM4Py
        model: Dict avec net, initial_marking, final_marking
        
    Returns:
        Score de généralisation (0-1)
    """
    import pm4py
    
    net = model['net']
    im = model['initial_marking']
    fm = model['final_marking']
    
    return pm4py.generalization_tbr(event_log, net, im, fm)


def compute_simplicity(model: Dict[str, Any]) -> float:
    """
    Calcule la simplicité d'un modèle.
    
    Basé sur le nombre de places et transitions.
    
    Args:
        model: Dict avec net
        
    Returns:
        Score de simplicité (0-1)
    """
    import pm4py
    
    net = model['net']
    
    return pm4py.simplicity_arc_degree(net)


def compute_all_metrics(
    event_log: Any,
    model: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calcule toutes les métriques de qualité d'un modèle.
    
    Args:
        event_log: Event log PM4Py
        model: Dict avec net, initial_marking, final_marking
        
    Returns:
        Dict avec fitness, precision, generalization, simplicity, f1
    """
    fitness_dict = compute_fitness(event_log, model)
    fitness = fitness_dict.get('average_trace_fitness', 0)
    
    precision = compute_precision(event_log, model)
    generalization = compute_generalization(event_log, model)
    simplicity = compute_simplicity(model)
    
    # F1 entre fitness et precision
    f1 = 2 * fitness * precision / (fitness + precision + 1e-6)
    
    return {
        'fitness': fitness,
        'precision': precision,
        'generalization': generalization,
        'simplicity': simplicity,
        'f1': f1
    }
