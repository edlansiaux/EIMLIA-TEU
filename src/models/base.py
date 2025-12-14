"""
Classe de base pour les modèles de triage
=========================================

Fournit l'interface commune et les utilitaires SHAP.
"""

import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Configuration globale
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class SHAPHelper:
    """Utilitaires pour l'explicabilité SHAP."""
    
    @staticmethod
    def compute_feature_importance(
        shap_values: Any,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Calcule l'importance moyenne des features depuis les valeurs SHAP.
        
        Args:
            shap_values: Valeurs SHAP (array ou liste pour multi-class)
            feature_names: Noms des features
            
        Returns:
            DataFrame avec feature et importance, trié par importance décroissante
        """
        if isinstance(shap_values, list):
            # Multi-class: moyenne sur toutes les classes
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    @staticmethod
    def print_importance(
        importance_df: pd.DataFrame,
        title: str = "Feature Importance",
        top_n: int = 15
    ) -> None:
        """
        Affiche l'importance des features avec barres visuelles.
        
        Args:
            importance_df: DataFrame avec colonnes 'feature' et 'importance'
            title: Titre à afficher
            top_n: Nombre de features à afficher
        """
        print(f"\n  📊 {title}:")
        print("  " + "-" * 40)
        max_imp = importance_df['importance'].max()
        for _, row in importance_df.head(top_n).iterrows():
            bar = "█" * int(row['importance'] * 50 / max_imp)
            print(f"  {row['feature']:20} {row['importance']:.4f} {bar}")


class BaseTriageModel(ABC):
    """
    Classe de base abstraite pour tous les modèles de triage.
    
    Définit l'interface commune que tous les modèles doivent implémenter.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        device: str = 'auto',
        random_seed: int = RANDOM_SEED
    ):
        """
        Initialise le modèle de base.
        
        Args:
            num_classes: Nombre de classes (niveaux FRENCH/CCMU)
            device: 'auto', 'cuda', 'mps', ou 'cpu'
            random_seed: Graine pour reproductibilité
        """
        self.num_classes = num_classes
        self.random_seed = random_seed
        self.device = self._get_device(device)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []
        
        # Fixer les graines
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    
    def _get_device(self, device: str) -> torch.device:
        """Détermine le device optimal."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    @abstractmethod
    def fit(
        self,
        texts: List[str],
        numerical: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        val_data: Optional[Tuple] = None
    ) -> 'BaseTriageModel':
        """
        Entraîne le modèle.
        
        Args:
            texts: Liste des textes (entretiens patient)
            numerical: Matrice des features numériques
            labels: Labels (0 à num_classes-1)
            feature_names: Noms des features numériques
            val_data: Tuple optionnel (texts_val, numerical_val, labels_val)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        texts: List[str],
        numerical: np.ndarray
    ) -> np.ndarray:
        """
        Prédit les classes.
        
        Args:
            texts: Liste des textes
            numerical: Matrice des features numériques
            
        Returns:
            Array des prédictions
        """
        pass
    
    @abstractmethod
    def predict_proba(
        self,
        texts: List[str],
        numerical: np.ndarray
    ) -> np.ndarray:
        """
        Prédit les probabilités par classe.
        
        Args:
            texts: Liste des textes
            numerical: Matrice des features numériques
            
        Returns:
            Matrice (n_samples, num_classes) de probabilités
        """
        pass
    
    @abstractmethod
    def explain_shap(
        self,
        texts: List[str],
        numerical: np.ndarray,
        n_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Calcule les explications SHAP.
        
        Args:
            texts: Liste des textes
            numerical: Matrice des features numériques
            n_samples: Nombre d'échantillons de background
            
        Returns:
            Dict avec:
                - 'shap_values': Valeurs SHAP
                - 'feature_names': Noms des features
                - 'numerical_importance': DataFrame importance features numériques
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            path: Chemin de sauvegarde
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  ✓ Modèle sauvegardé: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseTriageModel':
        """
        Charge un modèle sauvegardé.
        
        Args:
            path: Chemin du fichier
            
        Returns:
            Instance du modèle
        """
        import pickle
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"  ✓ Modèle chargé: {path}")
        return model
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_classes={self.num_classes}, device={self.device})"
