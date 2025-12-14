#!/usr/bin/env python
"""
Script d'entraînement des modèles
=================================

Usage:
    python scripts/train_models.py --data data/data3.xlsx --model all
    python scripts/train_models.py --data data/data3.xlsx --model emerginet --epochs 50
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    TRIAGEMASTER, URGENTIAPARSE, EMERGINET,
    load_data, generate_synthetic_data, evaluate_model, compare_models
)
from sklearn.model_selection import train_test_split


def train_model(model_name: str, args):
    """Entraîne un modèle spécifique."""
    
    # Charger les données
    if args.data and Path(args.data).exists():
        print(f"\n📂 Chargement: {args.data}")
        texts, numerical, labels, feature_names = load_data(args.data)
    else:
        print("\n⚠️  Utilisation de données synthétiques")
        texts, numerical, labels, feature_names = generate_synthetic_data(args.n_samples)
    
    # Split
    train_t, val_t, train_n, val_n, train_l, val_l = train_test_split(
        texts, numerical, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    print(f"\n📊 Train: {len(train_t)} | Validation: {len(val_t)}")
    
    # Créer le modèle
    if model_name == 'triagemaster':
        model = TRIAGEMASTER(
            doc2vec_dim=args.doc2vec_dim,
            epochs=args.epochs,
            patience=args.patience
        )
    elif model_name == 'urgentiaparse':
        model = URGENTIAPARSE(
            fine_tune_epochs=min(args.epochs, 5),
            batch_size=args.batch_size
        )
    elif model_name == 'emerginet':
        model = EMERGINET(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience
        )
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")
    
    # Entraîner
    model.fit(train_t, train_n, train_l, feature_names, val_data=(val_t, val_n, val_l))
    
    # Évaluer
    predictions = model.predict(val_t, val_n)
    results = evaluate_model(val_l, predictions, model_name.upper())
    
    # Sauvegarder
    if args.output:
        output_path = Path(args.output) / f"{model_name}.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_path))
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description='Entraînement des modèles EIMLIA')
    
    parser.add_argument('--data', type=str, default=None,
                       help='Chemin vers les données (Excel ou CSV)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['triagemaster', 'urgentiaparse', 'emerginet', 'all'],
                       help='Modèle à entraîner')
    parser.add_argument('--output', type=str, default='models/',
                       help='Répertoire de sortie')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Nombre d\'époques')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Taille des batches')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--doc2vec-dim', type=int, default=100,
                       help='Dimension Doc2Vec (TRIAGEMASTER)')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Nombre d\'échantillons synthétiques')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ENTRAÎNEMENT DES MODÈLES EIMLIA-3M-TEU")
    print("=" * 70)
    
    models_to_train = ['triagemaster', 'urgentiaparse', 'emerginet'] if args.model == 'all' else [args.model]
    
    all_results = {}
    
    for model_name in models_to_train:
        try:
            _, results = train_model(model_name, args)
            all_results[model_name.upper()] = results
        except Exception as e:
            print(f"\n❌ Erreur {model_name}: {e}")
    
    # Comparaison
    if len(all_results) > 1:
        compare_models(all_results)
    
    print("\n" + "=" * 70)
    print("  ✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)


if __name__ == '__main__':
    main()
