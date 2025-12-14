"""
Simulation DES avec SimPy
=========================

Remplace Arena V15 pour la simulation à événements discrets.
Modélise les flux patients et les ressources du SAU.
"""

import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd

from src.simulation.config import (
    SimulationConfig, DISTRIBUTIONS, TAUX_ERREUR_IA,
    sample_duration, get_lambda_for_hour, RANDOM_SEED
)
from src.simulation.patient_generator import PatientVirtuel, GenerateurPatientsVirtuels


@dataclass
class MetriquesPatient:
    """Métriques collectées pour un patient."""
    patient_id: str
    heure_arrivee: float
    heure_debut_triage: float = 0
    heure_fin_triage: float = 0
    heure_debut_consultation: float = 0
    heure_fin_consultation: float = 0
    heure_sortie: float = 0
    
    niveau_french_reel: int = 0
    niveau_french_ia: Optional[int] = None
    niveau_french_final: int = 0
    
    ia_utilisee: bool = False
    ia_acceptee: bool = False
    
    parcours: List[str] = field(default_factory=list)
    
    @property
    def attente_triage(self) -> float:
        return self.heure_debut_triage - self.heure_arrivee
    
    @property
    def duree_triage(self) -> float:
        return self.heure_fin_triage - self.heure_debut_triage
    
    @property
    def attente_consultation(self) -> float:
        return self.heure_debut_consultation - self.heure_fin_triage
    
    @property
    def dms(self) -> float:
        """Durée moyenne de séjour."""
        return self.heure_sortie - self.heure_arrivee


class SimulationUrgencesSimPy:
    """
    Simulation DES du service d'urgences avec SimPy.
    
    Remplace Arena V15 avec une approche process-based en Python.
    
    Ressources modélisées:
        - Box de triage (avec priorité)
        - Box de consultation (avec priorité FRENCH)
        - Scanner, Radio
        - Lits UHCD
        - Déchocage
    
    Example:
        >>> config = SimulationConfig(duree_jours=30, agent_ia='emerginet')
        >>> sim = SimulationUrgencesSimPy(config)
        >>> resultats = sim.run()
        >>> print(f"DMS médiane: {resultats['dms_median']:.1f} min")
    """
    
    def __init__(
        self,
        config: SimulationConfig = None,
        agent_ia_model: Any = None,
        verbose: bool = True
    ):
        """
        Args:
            config: Configuration de la simulation
            agent_ia_model: Instance du modèle IA (TRIAGEMASTER, URGENTIAPARSE, EMERGINET)
            verbose: Afficher la progression
        """
        self.config = config or SimulationConfig()
        self.agent_ia_model = agent_ia_model
        self.verbose = verbose
        
        self.rng = np.random.RandomState(self.config.random_seed)
        
        # Environnement SimPy
        self.env: Optional[simpy.Environment] = None
        
        # Ressources
        self.box_triage: Optional[simpy.PriorityResource] = None
        self.box_consultation: Optional[simpy.PriorityResource] = None
        self.scanners: Optional[simpy.Resource] = None
        self.radios: Optional[simpy.Resource] = None
        self.lits_uhcd: Optional[simpy.Resource] = None
        self.dechocage: Optional[simpy.Resource] = None
        
        # Métriques
        self.metriques_patients: List[MetriquesPatient] = []
        self.patients_en_cours: int = 0
        self.patients_traites: int = 0
        
        # Générateur de patients
        self.generateur = GenerateurPatientsVirtuels(self.config, self.config.random_seed)
    
    def _setup_resources(self) -> None:
        """Initialise les ressources SimPy."""
        self.box_triage = simpy.PriorityResource(
            self.env, 
            capacity=self.config.nb_box_triage
        )
        self.box_consultation = simpy.PriorityResource(
            self.env,
            capacity=self.config.nb_box_consultation
        )
        self.scanners = simpy.Resource(
            self.env,
            capacity=self.config.nb_scanners
        )
        self.radios = simpy.Resource(
            self.env,
            capacity=self.config.nb_radios
        )
        self.lits_uhcd = simpy.Resource(
            self.env,
            capacity=self.config.nb_lits_uhcd
        )
        self.dechocage = simpy.Resource(self.env, capacity=2)
    
    def _sample_duration(self, activity: str) -> float:
        """Échantillonne une durée depuis les distributions calibrées."""
        return sample_duration(activity, self.rng)
    
    def _appeler_ia(self, patient: PatientVirtuel) -> Optional[int]:
        """
        Appelle le modèle IA pour prédire le niveau FRENCH.
        
        Returns:
            Niveau prédit (1-5) ou None si pas d'IA
        """
        if self.agent_ia_model is None or self.config.agent_ia is None:
            return None
        
        # Vérifier si on utilise l'IA pour ce patient
        if self.rng.random() > self.config.taux_utilisation_ia:
            return None
        
        # Simuler la prédiction
        # En production, on appellerait vraiment le modèle:
        # prediction = self.agent_ia_model.predict([patient.verbatim], patient.get_numerical_features().reshape(1, -1))
        
        # Simulation basée sur taux d'erreur
        taux_erreur = TAUX_ERREUR_IA.get(self.config.agent_ia, 0.25)
        
        if self.rng.random() < taux_erreur:
            # Erreur: décaler de ±1 ou ±2 niveaux
            erreur = self.rng.choice([-2, -1, 1, 2], p=[0.1, 0.4, 0.4, 0.1])
            prediction = max(1, min(5, patient.niveau_french_reel + erreur))
        else:
            # Correct
            prediction = patient.niveau_french_reel
        
        return prediction
    
    def _decision_iao(
        self,
        patient: PatientVirtuel,
        niveau_ia: Optional[int]
    ) -> int:
        """
        Simule la décision de l'IAO face à la suggestion IA.
        
        Returns:
            Niveau FRENCH final décidé par l'IAO
        """
        niveau_reel = patient.niveau_french_reel
        
        if niveau_ia is None:
            # Pas d'IA: triage manuel avec variabilité
            variabilite = self.rng.normal(0, 0.8)
            return max(1, min(5, round(niveau_reel + variabilite)))
        
        # Modèle de décision IAO
        taux_acceptation = 0.85  # Base
        
        ecart = abs(niveau_ia - niveau_reel)
        
        if ecart == 0:
            taux_acceptation += 0.10  # Concordance parfaite
        elif ecart == 1:
            taux_acceptation -= 0.05
        else:
            taux_acceptation -= 0.30  # Écart ≥2
        
        # Facteurs additionnels (simulés)
        if patient.spo2 < 94 or patient.glasgow <= 12:
            taux_acceptation += 0.10  # Alertes
        
        taux_acceptation = max(0.3, min(0.98, taux_acceptation))
        
        if self.rng.random() < taux_acceptation:
            return niveau_ia
        else:
            # IAO override
            variabilite = self.rng.normal(0, 0.5)
            return max(1, min(5, round(niveau_reel + variabilite)))
    
    def processus_patient(self, patient: PatientVirtuel) -> None:
        """
        Processus SimPy pour un patient.
        
        Modélise le parcours complet: arrivée → triage → consultation → examens → sortie
        """
        metriques = MetriquesPatient(
            patient_id=patient.id,
            heure_arrivee=self.env.now,
            niveau_french_reel=patient.niveau_french_reel
        )
        
        self.patients_en_cours += 1
        
        # =====================================================================
        # TRIAGE
        # =====================================================================
        # Priorité initiale basée sur motif/apparence (avant triage formel)
        priorite_initiale = 3  # Moyenne par défaut
        if patient.spo2 < 90 or patient.glasgow < 10:
            priorite_initiale = 1
        
        # Attente box triage
        with self.box_triage.request(priority=priorite_initiale) as req:
            yield req
            metriques.heure_debut_triage = self.env.now
            
            # Appel IA
            niveau_ia = self._appeler_ia(patient)
            metriques.niveau_french_ia = niveau_ia
            metriques.ia_utilisee = niveau_ia is not None
            
            # Décision IAO
            niveau_final = self._decision_iao(patient, niveau_ia)
            metriques.niveau_french_final = niveau_final
            metriques.ia_acceptee = (niveau_ia == niveau_final) if niveau_ia else False
            
            # Durée du triage
            duree_triage = self._sample_duration('triage')
            yield self.env.timeout(duree_triage)
            
            metriques.heure_fin_triage = self.env.now
            metriques.parcours.append('triage')
        
        # =====================================================================
        # CONSULTATION
        # =====================================================================
        # Priorité FRENCH (1 = plus urgent)
        with self.box_consultation.request(priority=niveau_final) as req:
            yield req
            metriques.heure_debut_consultation = self.env.now
            
            duree_consultation = self._sample_duration('consultation')
            yield self.env.timeout(duree_consultation)
            
            metriques.heure_fin_consultation = self.env.now
            metriques.parcours.append('consultation')
        
        # =====================================================================
        # PARCOURS SPÉCIFIQUE
        # =====================================================================
        parcours_type = patient.parcours_type
        
        if parcours_type == 'dechocage':
            # Déchocage (urgence vitale)
            with self.dechocage.request() as req:
                yield req
                duree = self._sample_duration('dechocage')
                yield self.env.timeout(duree)
                metriques.parcours.append('dechocage')
        
        elif parcours_type == 'uhcd':
            # Unité d'Hospitalisation de Courte Durée
            with self.lits_uhcd.request() as req:
                yield req
                duree = self._sample_duration('uhcd')
                yield self.env.timeout(duree)
                metriques.parcours.append('uhcd')
        
        elif parcours_type == 'imagerie':
            # Imagerie (scanner ou radio)
            if self.rng.random() < 0.4:
                # Scanner
                with self.scanners.request() as req:
                    yield req
                    duree = self._sample_duration('scanner')
                    yield self.env.timeout(duree)
                    metriques.parcours.append('scanner')
            else:
                # Radio
                with self.radios.request() as req:
                    yield req
                    duree = self._sample_duration('radio')
                    yield self.env.timeout(duree)
                    metriques.parcours.append('radio')
        
        elif parcours_type == 'sortie_rapide':
            # Sortie rapide (pas d'examen complémentaire)
            pass
        
        else:  # standard
            # Parcours standard: possibilité d'examens
            if self.rng.random() < 0.3:
                with self.radios.request() as req:
                    yield req
                    duree = self._sample_duration('radio')
                    yield self.env.timeout(duree)
                    metriques.parcours.append('radio')
        
        # =====================================================================
        # SORTIE
        # =====================================================================
        metriques.heure_sortie = self.env.now
        metriques.parcours.append('sortie')
        
        self.metriques_patients.append(metriques)
        self.patients_en_cours -= 1
        self.patients_traites += 1
    
    def generateur_arrivees(self) -> None:
        """
        Processus générateur d'arrivées (Poisson non-homogène).
        """
        duree_minutes = self.config.duree_jours * 24 * 60
        
        while self.env.now < duree_minutes:
            # Heure courante (0-23)
            heure = int((self.env.now / 60) % 24)
            
            # Lambda pour cette heure
            lambda_h = get_lambda_for_hour(heure, self.config)
            
            # Temps inter-arrivée exponentiel
            inter_arrivee = self.rng.exponential(60 / lambda_h)
            yield self.env.timeout(inter_arrivee)
            
            # Générer et lancer le patient
            patient = self.generateur.generer_patient()
            self.env.process(self.processus_patient(patient))
    
    def run(self) -> Dict[str, Any]:
        """
        Exécute la simulation complète.
        
        Returns:
            Dict avec métriques agrégées
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  SIMULATION SimPy - {self.config.duree_jours} jours")
            print(f"  Agent IA: {self.config.agent_ia or 'Aucun (manuel)'}")
            print(f"  Facteur charge: {self.config.facteur_charge:.1f}x")
            print('='*60)
        
        # Reset
        self.env = simpy.Environment()
        self.metriques_patients = []
        self.patients_en_cours = 0
        self.patients_traites = 0
        
        # Setup ressources
        self._setup_resources()
        
        # Lancer le générateur d'arrivées
        self.env.process(self.generateur_arrivees())
        
        # Exécuter
        duree_minutes = self.config.duree_jours * 24 * 60
        
        if self.verbose:
            # Simulation avec affichage progression
            step = duree_minutes // 10
            for i in range(10):
                self.env.run(until=min((i + 1) * step, duree_minutes))
                print(f"  [{(i+1)*10}%] Jour {int(self.env.now / 1440)}, "
                      f"Patients traités: {self.patients_traites:,}")
        else:
            self.env.run(until=duree_minutes)
        
        # Calculer les résultats
        return self._compute_results()
    
    def _compute_results(self) -> Dict[str, Any]:
        """Calcule les métriques agrégées."""
        if not self.metriques_patients:
            return {}
        
        # Convertir en arrays
        dms_list = [m.dms for m in self.metriques_patients]
        attente_triage_list = [m.attente_triage for m in self.metriques_patients]
        attente_consult_list = [m.attente_consultation for m in self.metriques_patients]
        
        # Concordance IA
        ia_utilisee = [m for m in self.metriques_patients if m.ia_utilisee]
        if ia_utilisee:
            concordance = np.mean([
                m.niveau_french_ia == m.niveau_french_reel 
                for m in ia_utilisee
            ])
            taux_acceptation = np.mean([m.ia_acceptee for m in ia_utilisee])
        else:
            concordance = None
            taux_acceptation = None
        
        # Sous/sur-triage
        sous_triage = np.mean([
            m.niveau_french_final > m.niveau_french_reel
            for m in self.metriques_patients
        ])
        sur_triage = np.mean([
            m.niveau_french_final < m.niveau_french_reel
            for m in self.metriques_patients
        ])
        
        results = {
            'n_patients': len(self.metriques_patients),
            'duree_jours': self.config.duree_jours,
            'agent_ia': self.config.agent_ia,
            
            # DMS
            'dms_mean': float(np.mean(dms_list)),
            'dms_median': float(np.median(dms_list)),
            'dms_p95': float(np.percentile(dms_list, 95)),
            
            # Attentes
            'attente_triage_mean': float(np.mean(attente_triage_list)),
            'attente_triage_median': float(np.median(attente_triage_list)),
            'attente_consultation_mean': float(np.mean(attente_consult_list)),
            'attente_consultation_median': float(np.median(attente_consult_list)),
            
            # Qualité triage
            'concordance_ia': concordance,
            'taux_acceptation_ia': taux_acceptation,
            'sous_triage': float(sous_triage),
            'sur_triage': float(sur_triage),
            
            # Distribution parcours
            'parcours_distribution': self._get_parcours_distribution()
        }
        
        if self.verbose:
            print(f"\n  ✓ Simulation terminée!")
            print(f"    Patients traités: {results['n_patients']:,}")
            print(f"    DMS médiane: {results['dms_median']:.1f} min")
            print(f"    Attente triage: {results['attente_triage_median']:.1f} min")
            if concordance is not None:
                print(f"    Concordance IA: {concordance:.1%}")
                print(f"    Taux acceptation: {taux_acceptation:.1%}")
        
        return results
    
    def _get_parcours_distribution(self) -> Dict[str, int]:
        """Calcule la distribution des types de parcours."""
        parcours_count = {}
        for m in self.metriques_patients:
            key = '→'.join(m.parcours)
            parcours_count[key] = parcours_count.get(key, 0) + 1
        return parcours_count
    
    def get_dataframe(self) -> pd.DataFrame:
        """Retourne les métriques patients sous forme de DataFrame."""
        data = []
        for m in self.metriques_patients:
            data.append({
                'patient_id': m.patient_id,
                'heure_arrivee': m.heure_arrivee,
                'attente_triage': m.attente_triage,
                'duree_triage': m.duree_triage,
                'attente_consultation': m.attente_consultation,
                'dms': m.dms,
                'french_reel': m.niveau_french_reel,
                'french_ia': m.niveau_french_ia,
                'french_final': m.niveau_french_final,
                'ia_utilisee': m.ia_utilisee,
                'ia_acceptee': m.ia_acceptee,
                'parcours': '→'.join(m.parcours)
            })
        return pd.DataFrame(data)
