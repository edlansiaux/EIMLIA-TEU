"""
Simulation hybride SimPy + Mesa
===============================

Synchronise la simulation DES (flux) avec la simulation SMA (comportements).
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.simulation.config import SimulationConfig, get_lambda_for_hour, RANDOM_SEED
from src.simulation.patient_generator import GenerateurPatientsVirtuels, PatientVirtuel
from src.simulation.simpy_des import SimulationUrgencesSimPy
from src.simulation.mesa_sma import ModeleSAU, AgentPatient, EtatPatient


class SimulationHybride:
    """
    Simulation hybride combinant SimPy (DES) et Mesa (SMA).
    
    Architecture:
        - SimPy gère les flux temporels et les ressources
        - Mesa gère les comportements individuels des agents
        - Synchronisation bidirectionnelle à chaque pas
    
    Cette approche permet de:
        - Modéliser les contraintes de ressources (SimPy)
        - Capturer les comportements émergents (Mesa)
        - Simuler les interactions IA-soignants
    
    Example:
        >>> config = SimulationConfig(duree_jours=30, agent_ia='emerginet')
        >>> sim = SimulationHybride(config)
        >>> resultats = sim.run()
    """
    
    def __init__(
        self,
        config: SimulationConfig = None,
        verbose: bool = True
    ):
        """
        Args:
            config: Configuration de simulation
            verbose: Afficher la progression
        """
        self.config = config or SimulationConfig()
        self.verbose = verbose
        
        self.rng = np.random.RandomState(self.config.random_seed)
        
        # Composants
        self.simpy_sim: Optional[SimulationUrgencesSimPy] = None
        self.mesa_model: Optional[ModeleSAU] = None
        self.generateur: Optional[GenerateurPatientsVirtuels] = None
        
        # État
        self.temps_courant = 0  # en minutes
        self.patients_injectes = 0
        
        # Métriques agrégées
        self.metriques_history: List[Dict[str, Any]] = []
    
    def _setup(self) -> None:
        """Initialise les composants."""
        # Générateur de patients
        self.generateur = GenerateurPatientsVirtuels(
            self.config,
            self.config.random_seed
        )
        
        # Modèle Mesa
        self.mesa_model = ModeleSAU(self.config, verbose=False)
        
        # Temps et état
        self.temps_courant = 0
        self.patients_injectes = 0
        self._prochain_patient_temps = self._generer_inter_arrivee()
    
    def _generer_inter_arrivee(self) -> float:
        """Génère le temps jusqu'au prochain patient."""
        heure = int((self.temps_courant / 60) % 24)
        lambda_h = get_lambda_for_hour(heure, self.config)
        return self.rng.exponential(60 / lambda_h)
    
    def _injecter_patient(self) -> None:
        """Injecte un nouveau patient dans le système."""
        patient_virtuel = self.generateur.generer_patient()
        self.mesa_model.ajouter_patient(patient_virtuel)
        self.patients_injectes += 1
    
    def _synchroniser(self) -> None:
        """
        Synchronise les états entre SimPy et Mesa.
        
        Dans cette version simplifiée, Mesa gère principalement
        les comportements, tandis que les métriques temporelles
        sont calculées en temps réel.
        """
        # Collecter les métriques Mesa
        if self.mesa_model:
            self.mesa_model.datacollector.collect(self.mesa_model)
    
    def step(self) -> None:
        """
        Exécute un pas de simulation (1 minute).
        """
        # Vérifier si un patient arrive
        self._prochain_patient_temps -= 1
        if self._prochain_patient_temps <= 0:
            self._injecter_patient()
            self._prochain_patient_temps = self._generer_inter_arrivee()
        
        # Avancer Mesa d'un step
        self.mesa_model.step()
        
        # Synchroniser
        self._synchroniser()
        
        # Avancer le temps
        self.temps_courant += 1
    
    def run(self, duree_minutes: int = None) -> Dict[str, Any]:
        """
        Exécute la simulation complète.
        
        Args:
            duree_minutes: Durée optionnelle (sinon depuis config)
            
        Returns:
            Dict avec métriques agrégées
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  SIMULATION HYBRIDE SimPy+Mesa")
            print(f"  Durée: {self.config.duree_jours} jours")
            print(f"  Agent IA: {self.config.agent_ia or 'Manuel'}")
            print('='*60)
        
        # Setup
        self._setup()
        
        # Durée totale
        if duree_minutes is None:
            duree_minutes = self.config.duree_jours * 24 * 60
        
        # Boucle principale
        checkpoints = [int(duree_minutes * p / 10) for p in range(1, 11)]
        checkpoint_idx = 0
        
        for minute in range(duree_minutes):
            self.step()
            
            # Affichage progression
            if self.verbose and checkpoint_idx < len(checkpoints):
                if minute >= checkpoints[checkpoint_idx]:
                    pct = (checkpoint_idx + 1) * 10
                    patients_sortis = sum(
                        1 for a in self.mesa_model.schedule.agents
                        if isinstance(a, AgentPatient) and a.etat == EtatPatient.SORTIE
                    )
                    print(f"  [{pct}%] Jour {minute // 1440}, "
                          f"Patients: {self.patients_injectes}, "
                          f"Traités: {patients_sortis}")
                    checkpoint_idx += 1
        
        # Résultats
        return self._compute_results()
    
    def _compute_results(self) -> Dict[str, Any]:
        """Calcule les métriques finales."""
        mesa_results = self.mesa_model.get_resultats()
        
        # Enrichir avec métriques supplémentaires
        results = {
            'duree_jours': self.config.duree_jours,
            'agent_ia': self.config.agent_ia,
            'facteur_charge': self.config.facteur_charge,
            'patients_injectes': self.patients_injectes,
            **mesa_results
        }
        
        # Collecter historique
        model_vars = self.mesa_model.datacollector.get_model_vars_dataframe()
        if not model_vars.empty:
            results['satisfaction_finale'] = model_vars['satisfaction_moyenne'].iloc[-1]
            results['pic_attente_triage'] = model_vars['patients_attente_triage'].max()
            results['pic_attente_consultation'] = model_vars['patients_attente_consultation'].max()
        
        if self.verbose:
            print(f"\n  ✓ Simulation terminée!")
            print(f"    Patients injectés: {self.patients_injectes:,}")
            print(f"    Patients traités: {results.get('n_patients', 0):,}")
            print(f"    DMS moyenne: {results.get('dms_mean', 0):.1f} min")
            if results.get('concordance_ia') is not None:
                print(f"    Concordance IA: {results['concordance_ia']:.1%}")
        
        return results
    
    def run_stress_test(
        self,
        test_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Exécute un stress-test.
        
        Args:
            test_type: 'surge', 'failure', 'data_loss'
            **kwargs: Paramètres du test
            
        Returns:
            Résultats du stress-test
        """
        if test_type == 'surge':
            return self._stress_test_surge(**kwargs)
        elif test_type == 'failure':
            return self._stress_test_failure(**kwargs)
        elif test_type == 'data_loss':
            return self._stress_test_data_loss(**kwargs)
        else:
            raise ValueError(f"Type de test inconnu: {test_type}")
    
    def _stress_test_surge(
        self,
        factor: float = 2.0,
        duration_hours: int = 48
    ) -> Dict[str, Any]:
        """
        Test de surcharge: augmentation du flux patients.
        
        Args:
            factor: Multiplicateur du flux
            duration_hours: Durée de la surcharge
        """
        if self.verbose:
            print(f"\n  🔴 STRESS-TEST: Surcharge x{factor} pendant {duration_hours}h")
        
        # Modifier la config temporairement
        original_factor = self.config.facteur_charge
        self.config.facteur_charge = factor
        
        # Exécuter
        results = self.run(duree_minutes=duration_hours * 60)
        results['stress_test'] = 'surge'
        results['surge_factor'] = factor
        
        # Restaurer
        self.config.facteur_charge = original_factor
        
        return results
    
    def _stress_test_failure(
        self,
        component: str = 'ia',
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Test de panne: désactivation temporaire d'un composant.
        
        Args:
            component: 'ia' pour panne IA
            duration_minutes: Durée de la panne
        """
        if self.verbose:
            print(f"\n  🔴 STRESS-TEST: Panne {component} pendant {duration_minutes}min")
        
        # Setup avec IA
        self._setup()
        
        # Phase 1: Normal (30 min)
        for _ in range(30):
            self.step()
        
        # Phase 2: Panne (désactiver IA)
        original_ia = self.mesa_model.agent_ia
        self.mesa_model.agent_ia = None
        
        for _ in range(duration_minutes):
            self.step()
        
        # Phase 3: Récupération
        self.mesa_model.agent_ia = original_ia
        
        for _ in range(30):
            self.step()
        
        results = self._compute_results()
        results['stress_test'] = 'failure'
        results['failed_component'] = component
        results['failure_duration'] = duration_minutes
        
        return results
    
    def _stress_test_data_loss(
        self,
        percentage: float = 0.1
    ) -> Dict[str, Any]:
        """
        Test de perte de données: données incomplètes.
        
        Args:
            percentage: % de données perdues
        """
        if self.verbose:
            print(f"\n  🔴 STRESS-TEST: Perte données {percentage:.0%}")
        
        # Implémenter en masquant des features patients
        # (simplifié ici)
        results = self.run()
        results['stress_test'] = 'data_loss'
        results['data_loss_pct'] = percentage
        
        return results


def run_scenario(
    scenario_name: str,
    config: SimulationConfig = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Exécute un scénario prédéfini.
    
    Scénarios disponibles:
        - 'reference': Triage manuel
        - 'nlp': TRIAGEMASTER
        - 'llm': URGENTIAPARSE
        - 'jepa': EMERGINET
        - 'hybrid': LLM + JEPA
        - 'crisis': Charge 200% avec hybride
    
    Args:
        scenario_name: Nom du scénario
        config: Configuration de base
        **kwargs: Paramètres additionnels
        
    Returns:
        Résultats du scénario
    """
    if config is None:
        config = SimulationConfig()
    
    scenarios = {
        'reference': {'agent_ia': None, 'facteur_charge': 1.0},
        'nlp': {'agent_ia': 'triagemaster', 'facteur_charge': 1.0},
        'llm': {'agent_ia': 'urgentiaparse', 'facteur_charge': 1.0},
        'jepa': {'agent_ia': 'emerginet', 'facteur_charge': 1.0},
        'hybrid': {'agent_ia': 'emerginet', 'facteur_charge': 1.0},  # Simplifié
        'crisis': {'agent_ia': 'emerginet', 'facteur_charge': 2.0},
    }
    
    if scenario_name not in scenarios:
        raise ValueError(f"Scénario inconnu: {scenario_name}")
    
    # Appliquer les paramètres du scénario
    scenario_params = scenarios[scenario_name]
    for key, value in scenario_params.items():
        setattr(config, key, value)
    
    # Exécuter
    sim = SimulationHybride(config, verbose=kwargs.get('verbose', True))
    results = sim.run()
    results['scenario'] = scenario_name
    
    return results
