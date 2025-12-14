#!/usr/bin/env python
"""
Script d'évaluation des modèles
===============================

Usage:
    python scripts/evaluate_models.py --data data/test.csv --models models/
    python scripts/evaluate_models.py --synthetic 500 --compare
"""

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split

from src.models import (
    TRIAGEMASTER, URGENTIAPARSE, EMERGINET,
    load_data, generate_synthetic_data,
    evaluate_model, compare_models
)


def load_model(model_type: str, model_path: str = None):
    """Charge ou crée un modèle."""
    model_classes = {
        'triagemaster': TRIAGEMASTER,
        'urgentiaparse': URGENTIAPARSE,
        'emerginet': EMERGINET
    }
    
    model = model_classes[model_type]()
    
    if model_path and Path(model_path).exists():
        model.load(model_path)
        print(f"  ✓ Modèle chargé: {model_path}")
    
    return model


def evaluate_single_model(model, val_t, val_n, val_l, model_name: str):
    """Évalue un modèle unique."""
    print(f"\n{'='*60}")
    print(f"  ÉVALUATION: {model_name}")
    print('='*60)
    
    # Prédictions
    predictions = model.predict(val_t, val_n)
    
    # Métriques
    results = evaluate_model(val_l, predictions, model_name)
    
    # Explicabilité
    try:
        print("\n  📊 Test explicabilité SHAP...")
        shap_values, importance = model.explain_shap(val_t[:5], val_n[:5])
        print("  ✓ SHAP fonctionnel")
        
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]
        print("  Top 5 features:")
        for name, imp in top_features:
            print(f"    - {name}: {imp:.3f}")
    except Exception as e:
        print(f"  ⚠️ SHAP indisponible: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Évaluation des modèles EIMLIA')
    
    parser.add_argument('--data', type=str, default=None,
                       help='Chemin vers les données de test')
    parser.add_argument('--models', type=str, default='models/',
                       help='Répertoire des modèles')
    parser.add_argument('--model', type=str, default='all',
                       choices=['triagemaster', 'urgentiaparse', 'emerginet', 'all'],
                       help='Modèle à évaluer')
    parser.add_argument('--synthetic', type=int, default=None,
                       help='Utiliser N échantillons synthétiques')
    parser.add_argument('--compare', action='store_true',
                       help='Afficher la comparaison des modèles')
    parser.add_argument('--output', type=str, default=None,
                       help='Fichier de sortie JSON')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ÉVALUATION DES MODÈLES EIMLIA-3M-TEU")
    print("=" * 70)
    
    # Charger les données
    if args.data and Path(args.data).exists():
        print(f"\n📂 Chargement: {args.data}")
        texts, numerical, labels, feature_names = load_data(args.data)
    elif args.synthetic:
        print(f"\n⚗️ Génération de {args.synthetic} échantillons synthétiques")
        texts, numerical, labels, feature_names = generate_synthetic_data(args.synthetic)
    else:
        print("\n⚗️ Génération de données synthétiques par défaut (500)")
        texts, numerical, labels, feature_names = generate_synthetic_data(500)
    
    # Split pour avoir des données de test
    train_t, val_t, train_n, val_n, train_l, val_l = train_test_split(
        texts, numerical, labels,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )
    
    print(f"\n📊 Données de test: {len(val_t)} échantillons")
    
    # Modèles à évaluer
    models_to_eval = ['triagemaster', 'urgentiaparse', 'emerginet'] if args.model == 'all' else [args.model]
    
    all_results = {}
    
    for model_name in models_to_eval:
        try:
            # Charger ou créer le modèle
            model_path = Path(args.models) / f"{model_name}.pkl"
            model = load_model(model_name, str(model_path) if model_path.exists() else None)
            
            # Entraîner si pas déjà fait
            if not model.is_fitted:
                print(f"\n  🏋️ Entraînement {model_name}...")
                model.fit(train_t, train_n, train_l, feature_names)
            
            # Évaluer
            results = evaluate_single_model(model, val_t, val_n, val_l, model_name.upper())
            all_results[model_name.upper()] = results
            
        except Exception as e:
            print(f"\n❌ Erreur {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparaison
    if args.compare and len(all_results) > 1:
        compare_models(all_results)
    
    # Sauvegarder
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Résultats sauvegardés: {output_path}")
    
    print("\n" + "=" * 70)
    print("  ✅ ÉVALUATION TERMINÉE")
    print("=" * 70)
    
    return all_results


if __name__ == '__main__':
    main()
