#!/usr/bin/env python
"""
Génération de données synthétiques
==================================

Usage:
    python scripts/generate_synthetic_data.py --n-samples 10000 --output data/synthetic/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.data_loader import generate_synthetic_data
from src.simulation.patient_generator import GenerateurPatientsVirtuels
from src.simulation.config import SimulationConfig


def generate_ml_data(n_samples: int, output_dir: str):
    """Génère des données pour l'entraînement ML."""
    print(f"\n📊 Génération de {n_samples} échantillons ML...")
    
    texts, numerical, labels, feature_names = generate_synthetic_data(n_samples)
    
    # Créer le DataFrame
    df = pd.DataFrame(numerical, columns=feature_names)
    df['Entretien'] = texts
    df['CCMU'] = labels + 1  # Labels 1-indexed
    
    # Sauvegarder
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f'synthetic_{n_samples}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Données sauvegardées: {output_file}")
    print(f"  - Colonnes: {list(df.columns)}")
    print(f"  - Distribution CCMU: {df['CCMU'].value_counts().sort_index().to_dict()}")
    
    return df


def generate_simulation_data(n_patients: int, output_dir: str):
    """Génère des patients pour la simulation."""
    print(f"\n🏥 Génération de {n_patients} patients virtuels...")
    
    config = SimulationConfig()
    gen = GenerateurPatientsVirtuels(config, seed=42)
    
    patients = gen.generer_cohorte(n_patients, verbose=True)
    df = gen.to_dataframe(patients)
    
    # Sauvegarder
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f'patients_{n_patients}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Patients sauvegardés: {output_file}")
    
    return df


def generate_event_log(n_traces: int, output_dir: str):
    """Génère un event log pour Process Mining."""
    print(f"\n📋 Génération d'event log avec {n_traces} traces...")
    
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    
    activities = [
        'Arrivée', 'Triage', 'Consultation', 'Radio', 'Scanner',
        'Biologie', 'Avis spécialiste', 'Sortie'
    ]
    
    parcours_types = [
        ['Arrivée', 'Triage', 'Consultation', 'Sortie'],
        ['Arrivée', 'Triage', 'Consultation', 'Radio', 'Sortie'],
        ['Arrivée', 'Triage', 'Consultation', 'Biologie', 'Sortie'],
        ['Arrivée', 'Triage', 'Consultation', 'Scanner', 'Avis spécialiste', 'Sortie'],
        ['Arrivée', 'Triage', 'Consultation', 'Radio', 'Biologie', 'Sortie'],
    ]
    
    data = []
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    
    for i in range(n_traces):
        case_id = f'patient_{i:06d}'
        parcours = parcours_types[np.random.randint(len(parcours_types))]
        
        current_time = base_time + timedelta(
            days=np.random.randint(0, 180),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        for activity in parcours:
            data.append({
                'case:concept:name': case_id,
                'concept:name': activity,
                'time:timestamp': current_time.isoformat(),
                'org:resource': f'resource_{np.random.randint(1, 20)}'
            })
            
            # Ajouter du temps entre les activités
            duration = np.random.lognormal(2.5, 0.8)  # ~12 min médiane
            current_time += timedelta(minutes=duration)
    
    df = pd.DataFrame(data)
    
    # Sauvegarder en CSV (compatible PM4Py)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f'event_log_{n_traces}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Event log sauvegardé: {output_file}")
    print(f"  - Traces: {n_traces}")
    print(f"  - Événements: {len(df)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Génération de données synthétiques')
    
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Nombre d\'échantillons à générer')
    parser.add_argument('--output', type=str, default='data/synthetic/',
                       help='Répertoire de sortie')
    parser.add_argument('--type', type=str, default='all',
                       choices=['ml', 'simulation', 'eventlog', 'all'],
                       help='Type de données à générer')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  GÉNÉRATION DE DONNÉES SYNTHÉTIQUES EIMLIA")
    print("=" * 70)
    
    if args.type in ['ml', 'all']:
        generate_ml_data(args.n_samples, args.output)
    
    if args.type in ['simulation', 'all']:
        generate_simulation_data(args.n_samples, args.output)
    
    if args.type in ['eventlog', 'all']:
        generate_event_log(args.n_samples, args.output)
    
    print("\n" + "=" * 70)
    print("  ✅ GÉNÉRATION TERMINÉE")
    print("=" * 70)


if __name__ == '__main__':
    main()
