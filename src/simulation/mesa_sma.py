"""
Simulation SMA avec Mesa
========================

Remplace AnyLogic pour la simulation multi-agents.
Modélise les comportements individuels des agents (patients, IAO, médecins, IA).
"""

import mesa
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.simulation.config import (
    SimulationConfig, TAUX_ERREUR_IA, COMPORTEMENT_IAO,
    sample_duration, RANDOM_SEED
)
from src.simulation.patient_generator import PatientVirtuel


class EtatPatient(Enum):
    """États possibles d'un patient."""
    ARRIVEE = auto()
    ATTENTE_TRIAGE = auto()
    EN_TRIAGE = auto()
    ATTENTE_CONSULTATION = auto()
    EN_CONSULTATION = auto()
    EN_EXAMEN = auto()
    EN_UHCD = auto()
    EN_DECHOCAGE = auto()
    SORTIE = auto()


@dataclass
class ProfilPatient:
    """Profil complet d'un patient pour l'agent."""
    age: int
    sexe: str
    verbatim: str
    fc: float
    pas: float
    pad: float
    spo2: float
    temperature: float
    eva: int
    glasgow: int
    charlson: int
    niveau_french_reel: int
    parcours_type: str


class AgentPatient(mesa.Agent):
    """
    Agent représentant un patient aux urgences.
    
    Comportements:
        - Attend dans les files
        - Peut demander réévaluation si attente longue
        - Satisfaction diminue avec le temps d'attente
    """
    
    def __init__(
        self,
        unique_id: int,
        model: 'ModeleSAU',
        profil: ProfilPatient
    ):
        super().__init__(unique_id, model)
        
        self.profil = profil
        self.etat = EtatPatient.ARRIVEE
        
        # Niveaux FRENCH
        self.niveau_french_reel = profil.niveau_french_reel
        self.niveau_french_ia: Optional[int] = None
        self.niveau_french_final: Optional[int] = None
        
        # Timestamps (en steps)
        self.step_arrivee = model.schedule.steps
        self.step_debut_triage: Optional[int] = None
        self.step_fin_triage: Optional[int] = None
        self.step_debut_consultation: Optional[int] = None
        self.step_sortie: Optional[int] = None
        
        # Comportement
        self.satisfaction = 1.0  # 0 à 1
        self.demande_reevaluation = False
        
        # Assignations
        self.iao_assigne: Optional['AgentIAO'] = None
        self.medecin_assigne: Optional['AgentMedecin'] = None
    
    def step(self) -> None:
        """Exécute un pas de simulation pour le patient."""
        # Diminuer la satisfaction avec l'attente
        if self.etat in [EtatPatient.ATTENTE_TRIAGE, EtatPatient.ATTENTE_CONSULTATION]:
            self.satisfaction = max(0, self.satisfaction - 0.001)
            
            # Demander réévaluation si attente > 30 min (30 steps) et JEPA disponible
            if self.etat == EtatPatient.ATTENTE_CONSULTATION:
                attente = self.model.schedule.steps - (self.step_fin_triage or 0)
                if attente > 30 and not self.demande_reevaluation:
                    if self.model.random.random() < 0.1:
                        self.demande_reevaluation = True
    
    @property
    def attente_totale(self) -> int:
        """Temps d'attente total en steps."""
        return self.model.schedule.steps - self.step_arrivee
    
    @property
    def est_critique(self) -> bool:
        """Patient avec signes critiques."""
        return (
            self.profil.spo2 < 92 or
            self.profil.glasgow <= 12 or
            self.profil.fc > 130 or
            self.profil.pas < 80
        )


class AgentIAO(mesa.Agent):
    """
    Agent représentant un Infirmier d'Accueil et d'Orientation.
    
    Comportements:
        - Effectue le triage
        - Décide d'accepter ou non la suggestion IA
        - Fatigue augmente au fil du temps
    """
    
    def __init__(
        self,
        unique_id: int,
        model: 'ModeleSAU',
        experience_annees: int = 5
    ):
        super().__init__(unique_id, model)
        
        self.experience_annees = experience_annees
        self.fatigue = 0.0  # 0 à 1
        self.patients_triages = 0
        self.patient_actuel: Optional[AgentPatient] = None
        self.disponible = True
        
        # Paramètres comportementaux
        self.taux_acceptation_base = COMPORTEMENT_IAO['taux_acceptation_base']
    
    def step(self) -> None:
        """Exécute un pas de simulation pour l'IAO."""
        # Augmenter la fatigue
        self.fatigue = min(
            COMPORTEMENT_IAO['fatigue_max'],
            self.fatigue + COMPORTEMENT_IAO['fatigue_increment'] / 60
        )
        
        # Si en train de trier, continuer
        if self.patient_actuel is not None:
            self._continuer_triage()
        elif self.disponible:
            self._chercher_patient()
    
    def _chercher_patient(self) -> None:
        """Cherche un patient en attente de triage."""
        patients_attente = [
            a for a in self.model.schedule.agents
            if isinstance(a, AgentPatient) and a.etat == EtatPatient.ATTENTE_TRIAGE
        ]
        
        if patients_attente:
            # Priorité aux patients critiques
            patients_attente.sort(
                key=lambda p: (0 if p.est_critique else 1, p.step_arrivee)
            )
            patient = patients_attente[0]
            self._commencer_triage(patient)
    
    def _commencer_triage(self, patient: AgentPatient) -> None:
        """Commence le triage d'un patient."""
        self.patient_actuel = patient
        self.disponible = False
        patient.etat = EtatPatient.EN_TRIAGE
        patient.step_debut_triage = self.model.schedule.steps
        patient.iao_assigne = self
        
        # Durée du triage (en steps, 1 step = 1 minute)
        self._duree_triage_restante = int(sample_duration('triage', self.model.rng))
    
    def _continuer_triage(self) -> None:
        """Continue le triage en cours."""
        self._duree_triage_restante -= 1
        
        if self._duree_triage_restante <= 0:
            self._terminer_triage()
    
    def _terminer_triage(self) -> None:
        """Termine le triage et assigne le niveau FRENCH."""
        patient = self.patient_actuel
        
        # Appeler l'IA si disponible
        niveau_ia = self._consulter_ia(patient)
        patient.niveau_french_ia = niveau_ia
        
        # Décider du niveau final
        niveau_final = self._decider_niveau(patient, niveau_ia)
        patient.niveau_french_final = niveau_final
        
        # Mettre à jour les états
        patient.etat = EtatPatient.ATTENTE_CONSULTATION
        patient.step_fin_triage = self.model.schedule.steps
        
        self.patients_triages += 1
        self.patient_actuel = None
        self.disponible = True
    
    def _consulter_ia(self, patient: AgentPatient) -> Optional[int]:
        """Consulte l'agent IA pour une suggestion."""
        if self.model.agent_ia is None:
            return None
        
        return self.model.agent_ia.predire(patient)
    
    def _decider_niveau(
        self,
        patient: AgentPatient,
        niveau_ia: Optional[int]
    ) -> int:
        """
        Décide du niveau FRENCH final.
        
        Modèle de décision:
            - Base: taux d'acceptation
            - Bonus si concordance avec impression clinique
            - Malus si écart important
            - Influence de l'expérience et de la fatigue
        """
        niveau_reel = patient.niveau_french_reel
        
        if niveau_ia is None:
            # Triage manuel: impression clinique avec variabilité
            variabilite = self.model.rng.normal(0, 0.8 * (1 + self.fatigue * 0.5))
            return max(1, min(5, round(niveau_reel + variabilite)))
        
        # Calculer le taux d'acceptation
        taux = self.taux_acceptation_base
        
        # Impression clinique de l'IAO (simulée)
        impression_iao = niveau_reel + self.model.rng.normal(0, 0.5)
        ecart_ia_iao = abs(niveau_ia - impression_iao)
        
        if ecart_ia_iao < 0.5:
            taux += COMPORTEMENT_IAO['bonus_concordance_totale']
        elif ecart_ia_iao < 1.5:
            taux -= COMPORTEMENT_IAO['malus_ecart_1_niveau']
        else:
            taux -= COMPORTEMENT_IAO['malus_ecart_2_niveaux']
        
        # Facteurs IAO
        if self.experience_annees > 5:
            taux -= COMPORTEMENT_IAO['malus_experience_5ans']
        
        # Charge de travail
        patients_en_attente = sum(
            1 for a in self.model.schedule.agents
            if isinstance(a, AgentPatient) and a.etat == EtatPatient.ATTENTE_TRIAGE
        )
        if patients_en_attente > 10:
            taux -= COMPORTEMENT_IAO['malus_charge_10patients']
        
        # Alertes critiques
        if patient.est_critique:
            taux += COMPORTEMENT_IAO['bonus_alerte']
        
        # Borner
        taux = max(0.3, min(0.98, taux))
        
        # Décision
        if self.model.random.random() < taux:
            return niveau_ia
        else:
            return max(1, min(5, round(impression_iao)))


class AgentMedecin(mesa.Agent):
    """
    Agent représentant un médecin urgentiste.
    
    Comportements:
        - Consulte les patients par ordre de priorité FRENCH
        - Durée de consultation variable
    """
    
    def __init__(
        self,
        unique_id: int,
        model: 'ModeleSAU'
    ):
        super().__init__(unique_id, model)
        
        self.patient_actuel: Optional[AgentPatient] = None
        self.disponible = True
        self.patients_vus = 0
    
    def step(self) -> None:
        """Exécute un pas de simulation pour le médecin."""
        if self.patient_actuel is not None:
            self._continuer_consultation()
        elif self.disponible:
            self._chercher_patient()
    
    def _chercher_patient(self) -> None:
        """Cherche un patient en attente de consultation."""
        patients_attente = [
            a for a in self.model.schedule.agents
            if isinstance(a, AgentPatient) and a.etat == EtatPatient.ATTENTE_CONSULTATION
        ]
        
        if patients_attente:
            # Priorité FRENCH (1 = plus urgent)
            patients_attente.sort(
                key=lambda p: (p.niveau_french_final or 5, p.step_fin_triage or 0)
            )
            patient = patients_attente[0]
            self._commencer_consultation(patient)
    
    def _commencer_consultation(self, patient: AgentPatient) -> None:
        """Commence la consultation."""
        self.patient_actuel = patient
        self.disponible = False
        patient.etat = EtatPatient.EN_CONSULTATION
        patient.step_debut_consultation = self.model.schedule.steps
        patient.medecin_assigne = self
        
        self._duree_consultation_restante = int(sample_duration('consultation', self.model.rng))
    
    def _continuer_consultation(self) -> None:
        """Continue la consultation en cours."""
        self._duree_consultation_restante -= 1
        
        if self._duree_consultation_restante <= 0:
            self._terminer_consultation()
    
    def _terminer_consultation(self) -> None:
        """Termine la consultation."""
        patient = self.patient_actuel
        
        # Déterminer la suite du parcours
        parcours = patient.profil.parcours_type
        
        if parcours in ['sortie_rapide', 'standard']:
            patient.etat = EtatPatient.SORTIE
            patient.step_sortie = self.model.schedule.steps
        elif parcours == 'imagerie':
            patient.etat = EtatPatient.EN_EXAMEN
        elif parcours == 'uhcd':
            patient.etat = EtatPatient.EN_UHCD
        elif parcours == 'dechocage':
            patient.etat = EtatPatient.EN_DECHOCAGE
        else:
            patient.etat = EtatPatient.SORTIE
            patient.step_sortie = self.model.schedule.steps
        
        self.patients_vus += 1
        self.patient_actuel = None
        self.disponible = True


class AgentIA(mesa.Agent):
    """
    Agent représentant le système d'IA de triage.
    
    Wrapper pour les modèles NLP/LLM/JEPA.
    Simule les prédictions et la réévaluation.
    """
    
    def __init__(
        self,
        unique_id: int,
        model: 'ModeleSAU',
        type_ia: str = 'emerginet'
    ):
        super().__init__(unique_id, model)
        
        self.type_ia = type_ia
        self.taux_erreur = TAUX_ERREUR_IA.get(type_ia, 0.25)
        self.predictions_effectuees = 0
        self.reevaluations_effectuees = 0
        
        # Latences simulées (en ms)
        self.latences = {
            'triagemaster': 120,
            'urgentiaparse': 380,
            'emerginet': 240
        }
    
    def step(self) -> None:
        """L'agent IA n'a pas d'action par step (réactif)."""
        pass
    
    def predire(self, patient: AgentPatient) -> int:
        """
        Prédit le niveau FRENCH pour un patient.
        
        En production, appellerait le vrai modèle.
        Ici, simule basé sur le taux d'erreur.
        """
        self.predictions_effectuees += 1
        niveau_reel = patient.niveau_french_reel
        
        if self.model.random.random() < self.taux_erreur:
            # Erreur
            erreur = self.model.rng.choice([-2, -1, 1, 2], p=[0.1, 0.4, 0.4, 0.1])
            return max(1, min(5, niveau_reel + erreur))
        else:
            return niveau_reel
    
    def reevaluer(self, patient: AgentPatient) -> Optional[int]:
        """
        Réévalue un patient (fonctionnalité EMERGINET).
        
        Détecte les dégradations potentielles.
        """
        if self.type_ia != 'emerginet':
            return None
        
        self.reevaluations_effectuees += 1
        
        # Simuler détection de dégradation
        attente = patient.attente_totale
        
        if attente > 60 and self.model.random.random() < 0.1:
            # Dégradation détectée: suggérer niveau plus grave
            nouveau_niveau = max(1, (patient.niveau_french_final or 3) - 1)
            return nouveau_niveau
        
        return patient.niveau_french_final
    
    def detecter_alertes(self, patient: AgentPatient) -> List[str]:
        """Détecte les alertes critiques."""
        alertes = []
        
        if patient.profil.glasgow <= 12:
            alertes.append(f"Glasgow bas: {patient.profil.glasgow}")
        if patient.profil.spo2 < 94:
            alertes.append(f"SpO2 basse: {patient.profil.spo2}%")
        if patient.profil.fc > 120:
            alertes.append(f"Tachycardie: {patient.profil.fc} bpm")
        if patient.profil.pas < 90:
            alertes.append(f"Hypotension: {patient.profil.pas} mmHg")
        
        return alertes


class ModeleSAU(mesa.Model):
    """
    Modèle Mesa du Service d'Accueil des Urgences.
    
    Coordonne tous les agents et collecte les métriques.
    
    Example:
        >>> model = ModeleSAU(config)
        >>> for _ in range(1000):  # 1000 steps = ~16h
        ...     model.step()
        >>> resultats = model.get_resultats()
    """
    
    def __init__(
        self,
        config: SimulationConfig = None,
        verbose: bool = True
    ):
        super().__init__()
        
        self.config = config or SimulationConfig()
        self.verbose = verbose
        
        self.rng = np.random.RandomState(self.config.random_seed)
        self.schedule = mesa.time.RandomActivation(self)
        
        # Compteur d'agents
        self._agent_counter = 0
        
        # Créer les agents soignants
        self._creer_agents_soignants()
        
        # Agent IA
        self.agent_ia: Optional[AgentIA] = None
        if self.config.agent_ia:
            self._agent_counter += 1
            self.agent_ia = AgentIA(
                self._agent_counter,
                self,
                self.config.agent_ia
            )
            self.schedule.add(self.agent_ia)
        
        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                'patients_attente_triage': lambda m: sum(
                    1 for a in m.schedule.agents
                    if isinstance(a, AgentPatient) and a.etat == EtatPatient.ATTENTE_TRIAGE
                ),
                'patients_attente_consultation': lambda m: sum(
                    1 for a in m.schedule.agents
                    if isinstance(a, AgentPatient) and a.etat == EtatPatient.ATTENTE_CONSULTATION
                ),
                'patients_traites': lambda m: sum(
                    1 for a in m.schedule.agents
                    if isinstance(a, AgentPatient) and a.etat == EtatPatient.SORTIE
                ),
                'satisfaction_moyenne': lambda m: np.mean([
                    a.satisfaction for a in m.schedule.agents
                    if isinstance(a, AgentPatient)
                ] or [1.0])
            },
            agent_reporters={
                'etat': lambda a: a.etat.name if isinstance(a, AgentPatient) else None,
                'attente': lambda a: a.attente_totale if isinstance(a, AgentPatient) else None
            }
        )
    
    def _creer_agents_soignants(self) -> None:
        """Crée les agents IAO et médecins."""
        # IAO
        for i in range(self.config.nb_iao):
            self._agent_counter += 1
            experience = self.rng.randint(1, 15)
            iao = AgentIAO(self._agent_counter, self, experience)
            self.schedule.add(iao)
        
        # Médecins
        for i in range(self.config.nb_medecins):
            self._agent_counter += 1
            medecin = AgentMedecin(self._agent_counter, self)
            self.schedule.add(medecin)
    
    def ajouter_patient(self, patient_virtuel: PatientVirtuel) -> AgentPatient:
        """
        Ajoute un nouveau patient au modèle.
        
        Args:
            patient_virtuel: Patient généré
            
        Returns:
            AgentPatient créé
        """
        self._agent_counter += 1
        
        profil = ProfilPatient(
            age=patient_virtuel.age,
            sexe=patient_virtuel.sexe,
            verbatim=patient_virtuel.verbatim,
            fc=patient_virtuel.fc,
            pas=patient_virtuel.pas,
            pad=patient_virtuel.pad,
            spo2=patient_virtuel.spo2,
            temperature=patient_virtuel.temperature,
            eva=patient_virtuel.eva,
            glasgow=patient_virtuel.glasgow,
            charlson=patient_virtuel.charlson_index,
            niveau_french_reel=patient_virtuel.niveau_french_reel,
            parcours_type=patient_virtuel.parcours_type
        )
        
        agent = AgentPatient(self._agent_counter, self, profil)
        agent.etat = EtatPatient.ATTENTE_TRIAGE
        self.schedule.add(agent)
        
        return agent
    
    def step(self) -> None:
        """Exécute un pas de simulation."""
        self.datacollector.collect(self)
        self.schedule.step()
    
    def get_resultats(self) -> Dict[str, Any]:
        """Calcule et retourne les résultats de la simulation."""
        patients = [
            a for a in self.schedule.agents
            if isinstance(a, AgentPatient)
        ]
        
        patients_sortis = [p for p in patients if p.etat == EtatPatient.SORTIE]
        
        if not patients_sortis:
            return {'n_patients': 0}
        
        # Métriques
        dms_list = [
            (p.step_sortie - p.step_arrivee)
            for p in patients_sortis
            if p.step_sortie
        ]
        
        attente_triage = [
            (p.step_debut_triage - p.step_arrivee)
            for p in patients_sortis
            if p.step_debut_triage
        ]
        
        # Concordance
        avec_ia = [p for p in patients_sortis if p.niveau_french_ia is not None]
        if avec_ia:
            concordance = np.mean([
                p.niveau_french_ia == p.niveau_french_reel
                for p in avec_ia
            ])
        else:
            concordance = None
        
        return {
            'n_patients': len(patients_sortis),
            'n_patients_total': len(patients),
            'dms_mean': float(np.mean(dms_list)) if dms_list else 0,
            'dms_median': float(np.median(dms_list)) if dms_list else 0,
            'attente_triage_mean': float(np.mean(attente_triage)) if attente_triage else 0,
            'concordance_ia': concordance,
            'satisfaction_moyenne': float(np.mean([p.satisfaction for p in patients_sortis]))
        }
