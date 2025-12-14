"""
Chargement et préparation des données
=====================================

Fonctions pour charger les données du dataset EIMLIA.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

RANDOM_SEED = 42


def load_data(
    filepath: str,
    target_col: str = 'CCMU',
    text_col: str = 'Entretien',
    ignore_cols: Optional[List[str]] = None
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    """
    Charge et prépare les données depuis un fichier Excel.
    
    Variables utilisées:
        - Numériques: Age, FC, PAS, PAD, SpO2, Temperature, EVA, ShockIndex, O2, Sexe
        - Texte: Entretien
        - Cible: CCMU (1-4, excluant 'P' pour psychiatrie)
    
    Args:
        filepath: Chemin vers le fichier Excel (data3.xlsx)
        target_col: Nom de la colonne cible
        text_col: Nom de la colonne texte
        ignore_cols: Liste des colonnes à ignorer
        
    Returns:
        Tuple (texts, numerical, labels, feature_names):
            - texts: Liste des textes
            - numerical: Matrice numpy des features numériques
            - labels: Array des labels (0-indexed)
            - feature_names: Liste des noms de features
    
    Example:
        >>> texts, numerical, labels, features = load_data('data3.xlsx')
        >>> print(f"Samples: {len(texts)}, Features: {len(features)}")
    """
    print("=" * 60)
    print("CHARGEMENT DES DONNÉES")
    print("=" * 60)
    
    # Charger le fichier
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Format non supporté: {filepath}")
    
    print(f"  Fichier: {filepath}")
    print(f"  Shape initial: {df.shape}")
    
    # Colonnes à ignorer
    ignore_cols = ignore_cols or ['FRENCH inf']
    
    # Variables numériques (excluant GCS vide, Dextro/Duree trop de NaN)
    numerical_cols = [
        'Age', 'FC', 'PAS', 'PAD', 'SpO2', 
        'Temperature', 'EVA', 'ShockIndex', 'O2'
    ]
    numerical_cols = [
        c for c in numerical_cols 
        if c in df.columns and c not in ignore_cols
    ]
    
    # Encoder le sexe si présent
    if 'Sexe' in df.columns:
        df['Sexe_num'] = (df['Sexe'] == 'M').astype(float)
        numerical_cols.append('Sexe_num')
    
    print(f"  Features numériques: {numerical_cols}")
    
    # Filtrer: Entretien non vide + CCMU numérique (exclure 'P')
    df_work = df[df[text_col].notna()].copy()
    df_work = df_work[pd.to_numeric(df_work[target_col], errors='coerce').notna()]
    df_work[target_col] = pd.to_numeric(df_work[target_col]).astype(int)
    
    print(f"  Après filtrage: {len(df_work)} échantillons")
    print(f"  Distribution {target_col}: {df_work[target_col].value_counts().sort_index().to_dict()}")
    
    # Imputer les valeurs manquantes par la médiane
    for col in numerical_cols:
        if col in df_work.columns:
            median_val = df_work[col].median()
            df_work[col] = df_work[col].fillna(median_val)
    
    # Extraire les données
    texts = df_work[text_col].tolist()
    numerical = df_work[numerical_cols].values.astype(float)
    labels = df_work[target_col].values
    
    # Convertir labels en 0-indexed (CCMU 1->0, 2->1, 3->2, 4->3)
    labels = labels - labels.min()
    num_classes = len(np.unique(labels))
    
    print(f"  Classes: {num_classes} (labels 0 à {num_classes-1})")
    print("=" * 60)
    
    return texts, numerical, labels, numerical_cols


def generate_synthetic_data(
    n_samples: int = 400,
    random_seed: int = RANDOM_SEED
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    """
    Génère des données synthétiques pour tests et démonstrations.
    
    Args:
        n_samples: Nombre d'échantillons à générer
        random_seed: Graine aléatoire
        
    Returns:
        Tuple (texts, numerical, labels, feature_names)
        
    Example:
        >>> texts, numerical, labels, features = generate_synthetic_data(1000)
        >>> print(f"Generated {len(texts)} samples")
    """
    np.random.seed(random_seed)
    
    print("=" * 60)
    print("GÉNÉRATION DE DONNÉES SYNTHÉTIQUES")
    print("=" * 60)
    
    # Variables physiologiques
    age = np.random.normal(42, 20, n_samples).clip(18, 100)
    sex = np.random.binomial(1, 0.5, n_samples).astype(float)
    fc = np.random.normal(89, 18, n_samples).clip(40, 180)
    pas = np.random.normal(143, 24, n_samples).clip(70, 220)
    pad = np.random.normal(83, 16, n_samples).clip(40, 130)
    spo2 = np.random.normal(97, 2, n_samples).clip(70, 100)
    temp = np.random.normal(37.5, 0.6, n_samples).clip(35, 41)
    eva = np.random.uniform(0, 10, n_samples)
    shock = fc / pas
    o2 = np.random.exponential(0.1, n_samples).clip(0, 5)
    
    # Assembler les features
    numerical = np.column_stack([
        age, fc, pas, pad, spo2, temp, eva, shock, o2, sex
    ])
    feature_names = [
        'Age', 'FC', 'PAS', 'PAD', 'SpO2', 
        'Temperature', 'EVA', 'ShockIndex', 'O2', 'Sexe_num'
    ]
    
    # Générer les labels basés sur sévérité (4 classes comme CCMU)
    severity = (
        (pas < 90) * 2 + 
        (fc > 120) + 
        (spo2 < 92) * 2 + 
        (eva > 7) * 0.5
    )
    labels = np.digitize(severity, [0.5, 1.5, 2.5]).clip(0, 3)
    
    # Générer les textes
    complaints = [
        "douleur thoracique", "dyspnée", "malaise général",
        "céphalées", "douleur abdominale", "traumatisme",
        "fièvre", "vomissements", "vertiges", "palpitations"
    ]
    
    contexts = [
        "depuis quelques heures", "depuis ce matin",
        "depuis hier", "brutal", "progressif",
        "au repos", "à l'effort"
    ]
    
    texts = []
    for i in range(n_samples):
        complaint = np.random.choice(complaints)
        context = np.random.choice(contexts)
        
        text = (
            f"Patient de {int(age[i])} ans, {'homme' if sex[i] else 'femme'}, "
            f"consulte pour {complaint} {context}. "
            f"EVA {int(eva[i])}/10. SpO2 {int(spo2[i])}%. "
            f"FC {int(fc[i])} bpm, TA {int(pas[i])}/{int(pad[i])} mmHg. "
            f"Température {temp[i]:.1f}°C."
        )
        texts.append(text)
    
    # Statistiques
    print(f"  Échantillons générés: {n_samples}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Distribution classes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print("=" * 60)
    
    return texts, numerical, labels, feature_names


def split_data(
    texts: List[str],
    numerical: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    random_seed: int = RANDOM_SEED
) -> dict:
    """
    Divise les données en train/validation/test.
    
    Args:
        texts: Liste des textes
        numerical: Features numériques
        labels: Labels
        test_size: Proportion du test set
        val_size: Proportion du validation set (sur train)
        stratify: Stratifier par classe
        random_seed: Graine aléatoire
        
    Returns:
        Dict avec clés 'train', 'val', 'test', chacune contenant
        (texts, numerical, labels)
    """
    from sklearn.model_selection import train_test_split
    
    strat = labels if stratify else None
    
    # Split train+val / test
    train_t, test_t, train_n, test_n, train_l, test_l = train_test_split(
        texts, numerical, labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=strat
    )
    
    # Split train / val
    if val_size > 0:
        strat_train = train_l if stratify else None
        train_t, val_t, train_n, val_n, train_l, val_l = train_test_split(
            train_t, train_n, train_l,
            test_size=val_size / (1 - test_size),
            random_state=random_seed,
            stratify=strat_train
        )
    else:
        val_t, val_n, val_l = [], np.array([]), np.array([])
    
    return {
        'train': (train_t, train_n, train_l),
        'val': (val_t, val_n, val_l),
        'test': (test_t, test_n, test_l)
    }
