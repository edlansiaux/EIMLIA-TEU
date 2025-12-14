"""
Orchestrateur de simulation
===========================

Gère l'exécution des différents scénarios et la comparaison des résultats.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.simulation.config import SimulationConfig, RANDOM_SEED
from src.simulation.hybrid import SimulationHybride, run_scenario
from src.simulation.simpy_des import SimulationUrgencesSimPy


class OrchestrateurSimulation:
    """
    Orchestre l'exécution de tous les scénarios EIMLIA.
    
    Scénarios:
        1. Référence: Triage manuel (100K patients, 180 jours)
        2a. NLP: TRIAGEMASTER seul
        2b. LLM: URGENTIAPARSE seul
        2c. JEPA: EMERGINET seul
        3. Crise: Hybride LLM+JEPA, charge 200%
    
    Example:
        >>> orchestrateur = OrchestrateurSimulation(n_patients=100_000)
        >>> resultats = orchestrateur.executer_tous_scenarios()
        >>> orchestrateur.generer_rapport_comparatif(resultats)
    """
    
    SCENARIOS = {
        'reference': {
            'nom': '1. Référence (manuel)',
            'agent_ia': None,
            'facteur_charge': 1.0,
            'description': 'Triage manuel sans assistance IA'
        },
        'nlp': {
            'nom': '2a. NLP (TRIAGEMASTER)',
            'agent_ia': 'triagemaster',
            'facteur_charge': 1.0,
            'description': 'Assistance IA avec Doc2Vec + MLP'
        },
        'llm': {
            'nom': '2b. LLM (URGENTIAPARSE)',
            'agent_ia': 'urgentiaparse',
            'facteur_charge': 1.0,
            'description': 'Assistance IA avec FlauBERT + XGBoost'
        },
        'jepa': {
            'nom': '2c. JEPA (EMERGINET)',
            'agent_ia': 'emerginet',
            'facteur_charge': 1.0,
            'description': 'Assistance IA avec JEPA + VICReg'
        },
        'crise': {
            'nom': '3. Crise (hybride)',
            'agent_ia': 'emerginet',
            'facteur_charge': 2.0,
            'description': 'Scénario de crise avec charge doublée'
        }
    }
    
    def __init__(
        self,
        n_patients: int = 100_000,
        duree_jours: int = 180,
        random_seed: int = RANDOM_SEED,
        output_dir: str = 'results',
        verbose: bool = True
    ):
        """
        Args:
            n_patients: Nombre de patients par scénario (informatif)
            duree_jours: Durée de chaque simulation
            random_seed: Graine pour reproductibilité
            output_dir: Répertoire de sortie
            verbose: Afficher la progression
        """
        self.n_patients = n_patients
        self.duree_jours = duree_jours
        self.random_seed = random_seed
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.resultats: Dict[str, Dict] = {}
    
    def _creer_config(self, scenario_key: str) -> SimulationConfig:
        """Crée la configuration pour un scénario."""
        scenario = self.SCENARIOS[scenario_key]
        
        return SimulationConfig(
            duree_jours=self.duree_jours,
            agent_ia=scenario['agent_ia'],
            facteur_charge=scenario['facteur_charge'],
            random_seed=self.random_seed
        )
    
    def executer_scenario(
        self,
        scenario_key: str,
        avec_stress_tests: bool = True
    ) -> Dict[str, Any]:
        """
        Exécute un scénario unique.
        
        Args:
            scenario_key: Clé du scénario
            avec_stress_tests: Inclure les stress-tests
            
        Returns:
            Résultats du scénario
        """
        if scenario_key not in self.SCENARIOS:
            raise ValueError(f"Scénario inconnu: {scenario_key}")
        
        scenario = self.SCENARIOS[scenario_key]
        
        if self.verbose:
            print(f"\n{'#'*70}")
            print(f"# SCÉNARIO: {scenario['nom']}")
            print(f"# {scenario['description']}")
            print(f"{'#'*70}")
        
        # Configuration
        config = self._creer_config(scenario_key)
        
        # Simulation principale
        sim = SimulationHybride(config, verbose=self.verbose)
        resultats_principaux = sim.run()
        
        # Stress-tests (sauf pour référence et crise)
        resultats_stress = {}
        if avec_stress_tests and scenario_key not in ['reference', 'crise']:
            if self.verbose:
                print(f"\n  → Exécution des stress-tests...")
            
            # Test surcharge 48h
            config_stress = self._creer_config(scenario_key)
            sim_stress = SimulationHybride(config_stress, verbose=False)
            resultats_stress['surge_48h'] = sim_stress.run_stress_test(
                'surge', factor=1.5, duration_hours=48
            )
            
            # Test panne IA
            config_stress = self._creer_config(scenario_key)
            sim_stress = SimulationHybride(config_stress, verbose=False)
            resultats_stress['failure_ia'] = sim_stress.run_stress_test(
                'failure', component='ia', duration_minutes=60
            )
        
        # Assembler les résultats
        resultats = {
            'scenario': scenario_key,
            'nom': scenario['nom'],
            'description': scenario['description'],
            'config': {
                'duree_jours': self.duree_jours,
                'agent_ia': scenario['agent_ia'],
                'facteur_charge': scenario['facteur_charge']
            },
            'resultats_principaux': resultats_principaux,
            'stress_tests': resultats_stress,
            'timestamp': datetime.now().isoformat()
        }
        
        self.resultats[scenario_key] = resultats
        
        return resultats
    
    def executer_tous_scenarios(
        self,
        scenarios: List[str] = None,
        parallele: bool = False
    ) -> Dict[str, Dict]:
        """
        Exécute tous les scénarios.
        
        Args:
            scenarios: Liste des scénarios (tous si None)
            parallele: Exécution parallèle (expérimental)
            
        Returns:
            Dict des résultats par scénario
        """
        if scenarios is None:
            scenarios = list(self.SCENARIOS.keys())
        
        if self.verbose:
            print("=" * 70)
            print("  ORCHESTRATEUR EIMLIA-3M-TEU")
            print(f"  Scénarios: {len(scenarios)}")
            print(f"  Durée: {self.duree_jours} jours par scénario")
            print("=" * 70)
        
        if parallele:
            # Exécution parallèle (attention à la mémoire)
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(self.executer_scenario, s): s
                    for s in scenarios
                }
                for future in as_completed(futures):
                    scenario = futures[future]
                    try:
                        self.resultats[scenario] = future.result()
                    except Exception as e:
                        print(f"Erreur scénario {scenario}: {e}")
        else:
            # Exécution séquentielle
            for scenario in scenarios:
                self.executer_scenario(scenario)
        
        # Sauvegarder
        self._sauvegarder_resultats()
        
        return self.resultats
    
    def _sauvegarder_resultats(self) -> None:
        """Sauvegarde les résultats en JSON."""
        output_file = self.output_dir / f"resultats_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        # Convertir en sérialisable
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        resultats_json = json.loads(
            json.dumps(self.resultats, default=convert)
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(resultats_json, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"\n  ✓ Résultats sauvegardés: {output_file}")
    
    def generer_rapport_comparatif(
        self,
        resultats: Dict[str, Dict] = None
    ) -> str:
        """
        Génère un rapport comparatif des scénarios.
        
        Args:
            resultats: Résultats (self.resultats si None)
            
        Returns:
            Rapport formaté
        """
        if resultats is None:
            resultats = self.resultats
        
        lines = []
        lines.append("=" * 80)
        lines.append("  RAPPORT COMPARATIF - ÉTUDE EIMLIA-3M-TEU")
        lines.append("=" * 80)
        lines.append(f"\nDate: {datetime.now():%Y-%m-%d %H:%M}")
        lines.append(f"Durée simulation: {self.duree_jours} jours par scénario\n")
        
        # Tableau comparatif
        lines.append("-" * 80)
        lines.append(f"{'Scénario':<25} {'Patients':<10} {'DMS (min)':<12} "
                    f"{'Attente':<10} {'Concord.':<10} {'Sous-tri.':<10}")
        lines.append("-" * 80)
        
        for key, res in resultats.items():
            if 'resultats_principaux' not in res:
                continue
            
            rp = res['resultats_principaux']
            nom = res.get('nom', key)[:24]
            n_patients = rp.get('n_patients', 0)
            dms = rp.get('dms_median', 0)
            attente = rp.get('attente_triage_mean', 0)
            concordance = rp.get('concordance_ia')
            sous_triage = rp.get('sous_triage', 0)
            
            conc_str = f"{concordance:.1%}" if concordance else "N/A"
            
            lines.append(f"{nom:<25} {n_patients:<10,} {dms:<12.1f} "
                        f"{attente:<10.1f} {conc_str:<10} {sous_triage:<10.1%}")
        
        lines.append("-" * 80)
        
        # Analyse
        lines.append("\n📊 ANALYSE:")
        
        # Trouver le meilleur scénario (hors référence)
        best_dms = None
        best_scenario = None
        
        for key, res in resultats.items():
            if key == 'reference':
                continue
            if 'resultats_principaux' not in res:
                continue
            
            dms = res['resultats_principaux'].get('dms_median', float('inf'))
            if best_dms is None or dms < best_dms:
                best_dms = dms
                best_scenario = key
        
        if best_scenario:
            lines.append(f"\n  ✓ Meilleure performance DMS: {self.SCENARIOS[best_scenario]['nom']}")
        
        # Comparaison avec référence
        if 'reference' in resultats and best_scenario:
            ref_dms = resultats['reference']['resultats_principaux'].get('dms_median', 0)
            if ref_dms > 0:
                gain = (ref_dms - best_dms) / ref_dms * 100
                lines.append(f"  ✓ Gain DMS vs référence: {gain:.1f}%")
        
        # Recommandation
        lines.append("\n" + "=" * 80)
        lines.append("  RECOMMANDATION")
        lines.append("=" * 80)
        
        if best_scenario == 'jepa':
            lines.append("\n  Le modèle EMERGINET (JEPA + VICReg) offre les meilleures performances")
            lines.append("  en termes de réduction de la DMS et de qualité de triage.")
            lines.append("  La réévaluation continue est particulièrement utile en situation de crise.")
        elif best_scenario == 'llm':
            lines.append("\n  Le modèle URGENTIAPARSE (FlauBERT + XGBoost) offre un bon compromis")
            lines.append("  entre performance et explicabilité grâce à l'attention BERT.")
        else:
            lines.append(f"\n  Le scénario {best_scenario} présente les meilleurs résultats.")
        
        lines.append("\n" + "=" * 80)
        
        rapport = "\n".join(lines)
        
        # Sauvegarder
        rapport_file = self.output_dir / f"rapport_{datetime.now():%Y%m%d_%H%M%S}.txt"
        with open(rapport_file, 'w', encoding='utf-8') as f:
            f.write(rapport)
        
        if self.verbose:
            print(rapport)
            print(f"\n  ✓ Rapport sauvegardé: {rapport_file}")
        
        return rapport
    
    def generer_tableau_excel(
        self,
        resultats: Dict[str, Dict] = None,
        output_file: str = None
    ) -> pd.DataFrame:
        """
        Génère un tableau Excel des résultats.
        
        Args:
            resultats: Résultats
            output_file: Fichier de sortie
            
        Returns:
            DataFrame des résultats
        """
        if resultats is None:
            resultats = self.resultats
        
        rows = []
        for key, res in resultats.items():
            if 'resultats_principaux' not in res:
                continue
            
            rp = res['resultats_principaux']
            
            row = {
                'Scénario': res.get('nom', key),
                'Agent IA': res['config'].get('agent_ia', 'Manuel'),
                'Facteur charge': res['config'].get('facteur_charge', 1.0),
                'Patients traités': rp.get('n_patients', 0),
                'DMS moyenne (min)': rp.get('dms_mean', 0),
                'DMS médiane (min)': rp.get('dms_median', 0),
                'DMS P95 (min)': rp.get('dms_p95', 0),
                'Attente triage (min)': rp.get('attente_triage_mean', 0),
                'Attente consultation (min)': rp.get('attente_consultation_mean', 0),
                'Concordance IA': rp.get('concordance_ia'),
                'Taux acceptation': rp.get('taux_acceptation_ia'),
                'Sous-triage': rp.get('sous_triage', 0),
                'Sur-triage': rp.get('sur_triage', 0),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if output_file is None:
            output_file = self.output_dir / f"resultats_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        
        df.to_excel(output_file, index=False)
        
        if self.verbose:
            print(f"\n  ✓ Tableau Excel: {output_file}")
        
        return df


def main():
    """Point d'entrée CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Orchestrateur EIMLIA')
    parser.add_argument('--scenarios', nargs='+', default=None,
                       help='Scénarios à exécuter')
    parser.add_argument('--duree', type=int, default=30,
                       help='Durée en jours')
    parser.add_argument('--output', default='results',
                       help='Répertoire de sortie')
    
    args = parser.parse_args()
    
    orchestrateur = OrchestrateurSimulation(
        duree_jours=args.duree,
        output_dir=args.output
    )
    
    resultats = orchestrateur.executer_tous_scenarios(args.scenarios)
    orchestrateur.generer_rapport_comparatif(resultats)


if __name__ == '__main__':
    main()
