#!/usr/bin/env python
"""
Script d'exécution des simulations
==================================

Usage:
    python scripts/run_simulation.py --scenario all --days 30
    python scripts/run_simulation.py --scenario jepa --days 180 --stress-test
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.config import SimulationConfig
from src.simulation.orchestrator import OrchestrateurSimulation
from src.simulation.hybrid import SimulationHybride, run_scenario
from src.utils.visualization import plot_comparison, create_summary_table


def run_single_scenario(scenario: str, args):
    """Exécute un scénario unique."""
    
    config = SimulationConfig(
        duree_jours=args.days,
        facteur_charge=args.charge,
        random_seed=args.seed
    )
    
    results = run_scenario(scenario, config, verbose=not args.quiet)
    
    return results


def run_all_scenarios(args):
    """Exécute tous les scénarios via l'orchestrateur."""
    
    orchestrateur = OrchestrateurSimulation(
        duree_jours=args.days,
        random_seed=args.seed,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    scenarios = args.scenarios if args.scenarios else None
    
    resultats = orchestrateur.executer_tous_scenarios(
        scenarios=scenarios,
        parallele=args.parallel
    )
    
    # Générer le rapport
    orchestrateur.generer_rapport_comparatif(resultats)
    
    # Générer le tableau Excel
    df = orchestrateur.generer_tableau_excel(resultats)
    
    # Visualisation
    if not args.no_plot:
        plot_comparison(
            resultats,
            output_path=str(Path(args.output) / 'comparison.png'),
            show=not args.quiet
        )
    
    return resultats


def run_stress_tests(args):
    """Exécute les stress-tests."""
    
    print("\n" + "=" * 70)
    print("  STRESS-TESTS EIMLIA")
    print("=" * 70)
    
    config = SimulationConfig(
        duree_jours=min(args.days, 7),  # Max 7 jours pour stress-test
        agent_ia='emerginet',
        random_seed=args.seed
    )
    
    results = {}
    
    # Test 1: Surcharge
    print("\n📈 Test 1: Surcharge x2")
    sim = SimulationHybride(config, verbose=not args.quiet)
    results['surge'] = sim.run_stress_test('surge', factor=2.0, duration_hours=48)
    
    # Test 2: Panne IA
    print("\n⚠️  Test 2: Panne IA 60min")
    config2 = SimulationConfig(
        duree_jours=1,
        agent_ia='emerginet',
        random_seed=args.seed
    )
    sim2 = SimulationHybride(config2, verbose=not args.quiet)
    results['failure'] = sim2.run_stress_test('failure', component='ia', duration_minutes=60)
    
    # Résumé
    print("\n" + "-" * 70)
    print("  RÉSULTATS STRESS-TESTS")
    print("-" * 70)
    
    for test_name, test_results in results.items():
        dms = test_results.get('dms_mean', 0)
        n = test_results.get('n_patients', 0)
        print(f"  {test_name}: DMS={dms:.1f}min, N={n}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Simulation EIMLIA-3M-TEU')
    
    parser.add_argument('--scenario', type=str, default='all',
                       choices=['reference', 'nlp', 'llm', 'jepa', 'crise', 'all'],
                       help='Scénario à exécuter')
    parser.add_argument('--scenarios', nargs='+', default=None,
                       help='Liste de scénarios (si --scenario all)')
    parser.add_argument('--days', type=int, default=30,
                       help='Durée en jours')
    parser.add_argument('--charge', type=float, default=1.0,
                       help='Facteur de charge')
    parser.add_argument('--seed', type=int, default=42,
                       help='Graine aléatoire')
    parser.add_argument('--output', type=str, default='results/',
                       help='Répertoire de sortie')
    parser.add_argument('--parallel', action='store_true',
                       help='Exécution parallèle')
    parser.add_argument('--stress-test', action='store_true',
                       help='Exécuter les stress-tests')
    parser.add_argument('--no-plot', action='store_true',
                       help='Désactiver les graphiques')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Mode silencieux')
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  SIMULATION EIMLIA-3M-TEU")
    print(f"  Date: {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 70)
    
    if args.stress_test:
        results = run_stress_tests(args)
    elif args.scenario == 'all':
        results = run_all_scenarios(args)
    else:
        results = run_single_scenario(args.scenario, args)
    
    print("\n" + "=" * 70)
    print("  ✅ SIMULATION TERMINÉE")
    print(f"  Résultats: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
