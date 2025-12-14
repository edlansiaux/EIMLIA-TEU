"""
Générateur de patients virtuels
===============================

Génère des cohortes de patients synthétiques calibrés sur les données réelles.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Generator
from datetime import datetime, timedelta

from src.simulation.config import SimulationConfig, RANDOM_SEED


@dataclass
class PatientVirtuel:
    """Représentation d'un patient virtuel."""
    
    # Identifiant unique
    id: str
    
    # Démographie
    age: int
    sexe: str  # 'M' ou 'F'
    
    # Motif et contexte
    motif_consultation: str
    verbatim: str  # Texte de l'entretien
    
    # Constantes vitales
    fc: float       # Fréquence cardiaque (bpm)
    pas: float      # Pression artérielle systolique (mmHg)
    pad: float      # Pression artérielle diastolique (mmHg)
    spo2: float     # Saturation O2 (%)
    temperature: float  # Température (°C)
    eva: int        # Score douleur (0-10)
    glasgow: int    # Score Glasgow (3-15)
    
    # Scores dérivés
    shock_index: float = field(init=False)
    
    # Comorbidités
    charlson_index: int = 0
    
    # Niveau FRENCH réel (ground truth)
    niveau_french_reel: int = 3
    
    # Type de parcours attendu
    parcours_type: str = 'standard'
    
    # Timestamps
    heure_arrivee: Optional[datetime] = None
    
    def __post_init__(self):
        self.shock_index = self.fc / self.pas if self.pas > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'id': self.id,
            'age': self.age,
            'sexe': self.sexe,
            'motif': self.motif_consultation,
            'verbatim': self.verbatim,
            'fc': self.fc,
            'pas': self.pas,
            'pad': self.pad,
            'spo2': self.spo2,
            'temperature': self.temperature,
            'eva': self.eva,
            'glasgow': self.glasgow,
            'shock_index': self.shock_index,
            'charlson': self.charlson_index,
            'french_reel': self.niveau_french_reel,
            'parcours': self.parcours_type
        }
    
    def get_numerical_features(self) -> np.ndarray:
        """Retourne les features numériques pour les modèles IA."""
        return np.array([
            self.age,
            self.fc,
            self.pas,
            self.pad,
            self.spo2,
            self.temperature,
            self.eva,
            self.shock_index,
            0.0,  # O2 supplementaire (à enrichir)
            1.0 if self.sexe == 'M' else 0.0
        ])


class GenerateurPatientsVirtuels:
    """
    Génère des patients virtuels calibrés sur les distributions réelles.
    
    Distributions calibrées depuis PM4Py sur les données PMSI.
    
    Example:
        >>> gen = GenerateurPatientsVirtuels(seed=42)
        >>> patients = gen.generer_cohorte(100_000)
        >>> print(f"Généré {len(patients)} patients")
    """
    
    # Motifs de consultation et leur fréquence relative
    MOTIFS = {
        'douleur_thoracique': 0.08,
        'dyspnee': 0.07,
        'douleur_abdominale': 0.12,
        'traumatisme': 0.15,
        'cephalees': 0.06,
        'malaise': 0.08,
        'fievre': 0.10,
        'douleur_membre': 0.10,
        'nausees_vomissements': 0.05,
        'plaie': 0.08,
        'lombalgies': 0.04,
        'vertiges': 0.03,
        'palpitations': 0.02,
        'autre': 0.02
    }
    
    # Templates de verbatim par motif
    TEMPLATES_VERBATIM = {
        'douleur_thoracique': [
            "Patient de {age} ans, {sexe_txt}, consulte pour douleur thoracique {onset}. "
            "Douleur {type_douleur}, EVA {eva}/10. {irradiation}. {facteurs}.",
            
            "Arrivée pour oppression thoracique {onset}. Patient {sexe_txt} de {age} ans. "
            "Douleur cotée à {eva}/10, {type_douleur}. {antecedents}."
        ],
        'dyspnee': [
            "Patient de {age} ans présentant une dyspnée {onset}. {sexe_txt}. "
            "SpO2 à {spo2}% en air ambiant. {signes_associes}. EVA {eva}/10.",
            
            "{sexe_txt} de {age} ans, gêne respiratoire {onset}. "
            "Saturation {spo2}%. {antecedents}. Douleur {eva}/10."
        ],
        'douleur_abdominale': [
            "Douleur abdominale {localisation} {onset} chez {sexe_txt} de {age} ans. "
            "EVA {eva}/10. {signes_digestifs}. {transit}.",
            
            "Patient de {age} ans pour douleur abdominale. {localisation}, {onset}. "
            "Intensité {eva}/10. {signes_associes}."
        ],
        'traumatisme': [
            "Traumatisme {localisation} suite à {mecanisme}. Patient {sexe_txt} de {age} ans. "
            "Douleur {eva}/10. {impotence}. {plaie}.",
            
            "{sexe_txt} de {age} ans, traumatisme {localisation}. {mecanisme}. "
            "EVA {eva}/10. {examen_clinique}."
        ],
        'malaise': [
            "Malaise {type_malaise} {onset} chez {sexe_txt} de {age} ans. "
            "{circonstances}. {prodromes}. Glasgow {glasgow}.",
            
            "Patient de {age} ans amené pour malaise. {type_malaise}. "
            "{duree}. {recuperation}. TA {pas}/{pad}."
        ],
        'fievre': [
            "Fièvre à {temperature}°C {onset} chez {sexe_txt} de {age} ans. "
            "{signes_associes}. {point_appel}.",
            
            "Patient {sexe_txt} de {age} ans, hyperthermie {temperature}°C. "
            "{duree}. {signes_associes}. {traitement_domicile}."
        ],
        'default': [
            "Patient de {age} ans, {sexe_txt}, consulte pour {motif}. "
            "EVA {eva}/10. Constantes: TA {pas}/{pad}, FC {fc}, SpO2 {spo2}%.",
            
            "{sexe_txt} de {age} ans aux urgences pour {motif}. "
            "Douleur cotée {eva}/10. {signes_associes}."
        ]
    }
    
    def __init__(
        self,
        config: SimulationConfig = None,
        seed: int = RANDOM_SEED
    ):
        """
        Args:
            config: Configuration de simulation
            seed: Graine aléatoire
        """
        self.config = config or SimulationConfig()
        self.rng = np.random.RandomState(seed)
        self._patient_counter = 0
    
    def _generer_age(self, niveau_french: int = None) -> int:
        """Génère un âge avec distribution bimodale (jeunes/âgés)."""
        # Probabilité de patient âgé augmente avec gravité
        p_age = 0.3 if niveau_french is None else 0.2 + 0.15 * niveau_french
        
        if self.rng.random() < p_age:
            # Population âgée
            return int(self.rng.normal(75, 10).clip(60, 100))
        else:
            # Population jeune/adulte
            return int(self.rng.normal(40, 15).clip(18, 70))
    
    def _generer_constantes(self, niveau_french: int) -> Dict[str, float]:
        """Génère des constantes vitales cohérentes avec le niveau de gravité."""
        # Plus le niveau est bas (grave), plus les constantes sont anormales
        severity_factor = (5 - niveau_french) / 4  # 0 à 1
        
        # FC: augmente avec gravité
        fc_base = 75
        fc = self.rng.normal(
            fc_base + severity_factor * 30,
            10 + severity_factor * 10
        )
        
        # PAS: peut diminuer avec gravité (choc)
        pas_base = 130
        pas = self.rng.normal(
            pas_base - severity_factor * 20,
            15
        )
        
        # PAD
        pad = self.rng.normal(75, 10)
        
        # SpO2: diminue avec gravité
        spo2_base = 98
        spo2 = self.rng.normal(
            spo2_base - severity_factor * 5,
            2 + severity_factor * 2
        )
        
        # Température
        if self.rng.random() < 0.2 + severity_factor * 0.2:
            # Fièvre
            temp = self.rng.normal(38.5, 0.8)
        else:
            temp = self.rng.normal(37.0, 0.3)
        
        # EVA (douleur)
        eva = int(self.rng.normal(
            3 + severity_factor * 4,
            2
        ).clip(0, 10))
        
        # Glasgow
        if severity_factor > 0.7:
            glasgow = int(self.rng.choice([12, 13, 14, 15], p=[0.1, 0.2, 0.3, 0.4]))
        else:
            glasgow = 15
        
        return {
            'fc': float(fc.clip(40, 180)),
            'pas': float(pas.clip(70, 220)),
            'pad': float(pad.clip(40, 130)),
            'spo2': float(spo2.clip(70, 100)),
            'temperature': float(temp.clip(35, 41)),
            'eva': eva,
            'glasgow': glasgow
        }
    
    def _generer_verbatim(
        self,
        age: int,
        sexe: str,
        motif: str,
        constantes: Dict[str, float]
    ) -> str:
        """Génère un verbatim réaliste pour l'entretien IAO."""
        templates = self.TEMPLATES_VERBATIM.get(
            motif, 
            self.TEMPLATES_VERBATIM['default']
        )
        template = self.rng.choice(templates)
        
        # Variables de remplacement
        vars = {
            'age': age,
            'sexe_txt': 'homme' if sexe == 'M' else 'femme',
            'motif': motif.replace('_', ' '),
            'eva': constantes['eva'],
            'fc': int(constantes['fc']),
            'pas': int(constantes['pas']),
            'pad': int(constantes['pad']),
            'spo2': int(constantes['spo2']),
            'temperature': f"{constantes['temperature']:.1f}",
            'glasgow': constantes['glasgow'],
            'onset': self.rng.choice([
                'depuis ce matin', 'depuis hier', 'depuis quelques heures',
                'brutal', 'progressif depuis 2 jours'
            ]),
            'type_douleur': self.rng.choice([
                'constrictive', 'en coup de poignard', 'oppressive', 'sourde'
            ]),
            'irradiation': self.rng.choice([
                'Irradiation au bras gauche', 'Sans irradiation',
                'Irradiation dorsale', 'Irradiation à la mâchoire'
            ]),
            'facteurs': self.rng.choice([
                'Apparue au repos', 'À l\'effort', 'Post-prandiale'
            ]),
            'antecedents': self.rng.choice([
                'Pas d\'ATCD notables', 'ATCD HTA', 'Diabétique',
                'Coronarien connu', 'Tabagisme actif'
            ]),
            'signes_associes': self.rng.choice([
                'Sans signe associé', 'Nausées associées',
                'Sueurs froides', 'Pâleur cutanée'
            ]),
            'localisation': self.rng.choice([
                'épigastrique', 'en fosse iliaque droite',
                'diffuse', 'hypochondre droit', 'du membre inférieur'
            ]),
            'signes_digestifs': self.rng.choice([
                'Nausées', 'Vomissements', 'Diarrhées', 'Pas de signes digestifs'
            ]),
            'transit': self.rng.choice([
                'Transit conservé', 'Arrêt des matières', 'Constipation'
            ]),
            'mecanisme': self.rng.choice([
                'chute de sa hauteur', 'accident de la voie publique',
                'chute dans les escaliers', 'accident sportif'
            ]),
            'impotence': self.rng.choice([
                'Impotence fonctionnelle totale', 'Appui possible',
                'Mobilisation douloureuse'
            ]),
            'plaie': self.rng.choice([
                'Pas de plaie', 'Plaie superficielle', 'Dermabrasion'
            ]),
            'examen_clinique': self.rng.choice([
                'Pas de déformation', 'Œdème local', 'Hématome'
            ]),
            'type_malaise': self.rng.choice([
                'avec perte de connaissance', 'lipothymique',
                'sans perte de connaissance'
            ]),
            'circonstances': self.rng.choice([
                'en position debout', 'au lever', 'sans facteur déclenchant'
            ]),
            'prodromes': self.rng.choice([
                'Prodromes rapportés', 'Sans prodrome', 'Flou visuel avant'
            ]),
            'duree': self.rng.choice([
                'Depuis 24h', 'Depuis 48h', 'Depuis ce matin'
            ]),
            'recuperation': self.rng.choice([
                'Récupération complète', 'Confusion persistante'
            ]),
            'point_appel': self.rng.choice([
                'Syndrome grippal', 'Point d\'appel urinaire',
                'Toux productive', 'Pas de point d\'appel évident'
            ]),
            'traitement_domicile': self.rng.choice([
                'Paracétamol inefficace', 'Pas de traitement',
                'AINS à domicile'
            ])
        }
        
        # Remplacer les variables dans le template
        try:
            return template.format(**vars)
        except KeyError:
            # Fallback si variable manquante
            return f"Patient de {age} ans, {vars['sexe_txt']}, consulte pour {motif.replace('_', ' ')}. EVA {constantes['eva']}/10."
    
    def _determiner_parcours(self, niveau_french: int) -> str:
        """Détermine le type de parcours attendu."""
        parcours = self.config.parcours_par_niveau.get(niveau_french, {'standard': 1.0})
        types = list(parcours.keys())
        probs = list(parcours.values())
        return self.rng.choice(types, p=probs)
    
    def generer_patient(self, heure: datetime = None) -> PatientVirtuel:
        """
        Génère un patient virtuel unique.
        
        Args:
            heure: Heure d'arrivée optionnelle
            
        Returns:
            PatientVirtuel
        """
        self._patient_counter += 1
        patient_id = f"P{self._patient_counter:07d}"
        
        # Niveau FRENCH (ground truth)
        niveau_french = self.rng.choice(
            [1, 2, 3, 4, 5],
            p=self.config.distribution_french
        )
        
        # Démographie
        age = self._generer_age(niveau_french)
        sexe = 'M' if self.rng.random() < 0.52 else 'F'  # Légère majorité hommes
        
        # Motif de consultation
        motifs = list(self.MOTIFS.keys())
        probs = list(self.MOTIFS.values())
        motif = self.rng.choice(motifs, p=probs)
        
        # Constantes vitales
        constantes = self._generer_constantes(niveau_french)
        
        # Verbatim
        verbatim = self._generer_verbatim(age, sexe, motif, constantes)
        
        # Comorbidités (Charlson index)
        if age > 65:
            charlson = int(self.rng.poisson(2))
        else:
            charlson = int(self.rng.poisson(0.5))
        
        # Parcours
        parcours = self._determiner_parcours(niveau_french)
        
        return PatientVirtuel(
            id=patient_id,
            age=age,
            sexe=sexe,
            motif_consultation=motif,
            verbatim=verbatim,
            fc=constantes['fc'],
            pas=constantes['pas'],
            pad=constantes['pad'],
            spo2=constantes['spo2'],
            temperature=constantes['temperature'],
            eva=constantes['eva'],
            glasgow=constantes['glasgow'],
            charlson_index=charlson,
            niveau_french_reel=niveau_french,
            parcours_type=parcours,
            heure_arrivee=heure
        )
    
    def generer_cohorte(
        self,
        n_patients: int,
        date_debut: datetime = None,
        verbose: bool = True
    ) -> List[PatientVirtuel]:
        """
        Génère une cohorte complète de patients.
        
        Args:
            n_patients: Nombre de patients à générer
            date_debut: Date de début optionnelle
            verbose: Afficher la progression
            
        Returns:
            Liste de PatientVirtuel
        """
        if verbose:
            print(f"  Génération de {n_patients:,} patients virtuels...")
        
        patients = []
        
        for i in range(n_patients):
            patient = self.generer_patient()
            patients.append(patient)
            
            if verbose and (i + 1) % 10000 == 0:
                print(f"    {i+1:,} / {n_patients:,} ({100*(i+1)/n_patients:.1f}%)")
        
        if verbose:
            # Statistiques
            niveaux = [p.niveau_french_reel for p in patients]
            ages = [p.age for p in patients]
            
            print(f"\n  ✓ Cohorte générée:")
            print(f"    - Total: {len(patients):,} patients")
            print(f"    - Âge moyen: {np.mean(ages):.1f} ans")
            print(f"    - Distribution FRENCH: {dict(zip(*np.unique(niveaux, return_counts=True)))}")
        
        return patients
    
    def generer_flux(
        self,
        duree_heures: int = 24,
        date_debut: datetime = None
    ) -> Generator[PatientVirtuel, None, None]:
        """
        Génère un flux continu de patients (générateur).
        
        Utilise un processus de Poisson non-homogène pour les arrivées.
        
        Args:
            duree_heures: Durée du flux en heures
            date_debut: Timestamp de début
            
        Yields:
            PatientVirtuel avec heure d'arrivée
        """
        if date_debut is None:
            date_debut = datetime.now()
        
        temps_courant = 0  # minutes
        duree_minutes = duree_heures * 60
        
        while temps_courant < duree_minutes:
            # Heure courante
            heure = int((temps_courant / 60) % 24)
            
            # Lambda pour cette heure
            lambda_h = self.config.lambda_arrivees[heure] * self.config.facteur_charge
            
            # Temps inter-arrivée (exponentiel)
            inter_arrivee = self.rng.exponential(60 / lambda_h)  # en minutes
            temps_courant += inter_arrivee
            
            if temps_courant < duree_minutes:
                heure_arrivee = date_debut + timedelta(minutes=temps_courant)
                patient = self.generer_patient(heure_arrivee)
                yield patient
    
    def to_dataframe(self, patients: List[PatientVirtuel]) -> pd.DataFrame:
        """Convertit une liste de patients en DataFrame."""
        return pd.DataFrame([p.to_dict() for p in patients])
