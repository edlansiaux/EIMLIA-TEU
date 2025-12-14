"""
Découverte de processus
=======================

Algorithmes de découverte de processus avec PM4Py.
"""

from typing import Any, Dict, Tuple


def discover_alpha(event_log: Any) -> Dict[str, Any]:
    """
    Découvre un processus avec l'algorithme Alpha Miner.
    
    Algorithme classique, adapté aux logs simples sans bruit.
    
    Args:
        event_log: Event log PM4Py
        
    Returns:
        Dict avec net, initial_marking, final_marking
    """
    import pm4py
    
    net, im, fm = pm4py.discover_petri_net_alpha(event_log)
    
    return {
        'net': net,
        'initial_marking': im,
        'final_marking': fm,
        'algorithm': 'alpha'
    }


def discover_inductive(
    event_log: Any,
    noise_threshold: float = 0.0
) -> Any:
    """
    Découvre un processus avec l'Inductive Miner.
    
    Algorithme robuste au bruit, garantit un modèle sound.
    
    Args:
        event_log: Event log PM4Py
        noise_threshold: Seuil de filtrage du bruit (0-1)
        
    Returns:
        Process tree
    """
    import pm4py
    
    if noise_threshold > 0:
        process_tree = pm4py.discover_process_tree_inductive(
            event_log,
            noise_threshold=noise_threshold
        )
    else:
        process_tree = pm4py.discover_process_tree_inductive(event_log)
    
    return process_tree


def discover_heuristic(
    event_log: Any,
    dependency_threshold: float = 0.5,
    and_threshold: float = 0.65,
    loop_two_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Découvre un processus avec le Heuristics Miner.
    
    Algorithme basé sur les fréquences, bon pour les logs bruités.
    
    Args:
        event_log: Event log PM4Py
        dependency_threshold: Seuil de dépendance
        and_threshold: Seuil pour les AND-splits
        loop_two_threshold: Seuil pour les boucles de longueur 2
        
    Returns:
        Dict avec net, initial_marking, final_marking
    """
    import pm4py
    
    net, im, fm = pm4py.discover_petri_net_heuristics(
        event_log,
        dependency_threshold=dependency_threshold,
        and_threshold=and_threshold,
        loop_two_threshold=loop_two_threshold
    )
    
    return {
        'net': net,
        'initial_marking': im,
        'final_marking': fm,
        'algorithm': 'heuristic'
    }


def convert_to_petri_net(process_tree: Any) -> Tuple[Any, Any, Any]:
    """
    Convertit un process tree en Petri net.
    
    Args:
        process_tree: Process tree PM4Py
        
    Returns:
        Tuple (net, initial_marking, final_marking)
    """
    import pm4py
    
    return pm4py.convert_to_petri_net(process_tree)


def convert_to_bpmn(process_tree: Any) -> Any:
    """
    Convertit un process tree en BPMN.
    
    Args:
        process_tree: Process tree PM4Py
        
    Returns:
        Modèle BPMN
    """
    import pm4py
    
    return pm4py.convert_to_bpmn(process_tree)
