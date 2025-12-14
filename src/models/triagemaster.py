"""
TRIAGEMASTER - Doc2Vec + MLP
============================

Architecture NLP classique avec explicabilité SHAP.

Pipeline:
    1. Texte → Doc2Vec (100 dimensions)
    2. Concat avec features numériques normalisées
    3. MLP (128 → 64 → num_classes)
    4. Explicabilité: SHAP KernelExplainer
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseTriageModel, SHAPHelper, RANDOM_SEED


class Doc2VecEncoder:
    """Encodeur Doc2Vec pour les transcriptions."""
    
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 40
    ):
        """
        Args:
            vector_size: Dimension des vecteurs
            window: Taille de la fenêtre contextuelle
            min_count: Fréquence minimum des mots
            epochs: Époques d'entraînement Doc2Vec
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def fit(self, texts: List[str]) -> 'Doc2VecEncoder':
        """Entraîne le modèle Doc2Vec."""
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        
        # Tokenisation simple (peut être améliorée avec spaCy)
        tagged = [
            TaggedDocument(words=t.lower().split(), tags=[str(i)]) 
            for i, t in enumerate(texts)
        ]
        
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=1,  # Distributed Memory
            workers=4
        )
        self.model.build_vocab(tagged)
        self.model.train(
            tagged, 
            total_examples=self.model.corpus_count,
            epochs=self.epochs
        )
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforme les textes en vecteurs."""
        return np.array([
            self.model.infer_vector(t.lower().split()) 
            for t in texts
        ])
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit puis transform."""
        self.fit(texts)
        return self.transform(texts)


class TriageMasterMLP(nn.Module):
    """Réseau MLP pour TRIAGEMASTER."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int] = (128, 64),
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


class TRIAGEMASTER(BaseTriageModel):
    """
    Modèle TRIAGEMASTER: Doc2Vec + MLP avec SHAP.
    
    Architecture:
        - Encodeur texte: Doc2Vec (gensim)
        - Classifieur: MLP 2 couches avec BatchNorm et Dropout
        - Explicabilité: SHAP KernelExplainer
    
    Performances attendues:
        - Taux d'erreur: ~39%
        - Latence: ~120ms
    
    Example:
        >>> model = TRIAGEMASTER(doc2vec_dim=100, epochs=100)
        >>> model.fit(texts, numerical, labels, feature_names)
        >>> predictions = model.predict(texts_test, numerical_test)
        >>> shap_results = model.explain_shap(texts_test, numerical_test)
    """
    
    def __init__(
        self,
        doc2vec_dim: int = 100,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.3,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        num_classes: int = 4,
        device: str = 'auto'
    ):
        """
        Args:
            doc2vec_dim: Dimension des embeddings Doc2Vec
            hidden_dims: Dimensions des couches cachées MLP
            dropout: Taux de dropout
            lr: Learning rate
            weight_decay: Régularisation L2
            batch_size: Taille des batches
            epochs: Nombre max d'époques
            patience: Early stopping patience
            num_classes: Nombre de classes
            device: 'auto', 'cuda', 'mps', ou 'cpu'
        """
        super().__init__(num_classes=num_classes, device=device)
        
        self.doc2vec_dim = doc2vec_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        self.doc2vec_encoder = Doc2VecEncoder(vector_size=doc2vec_dim)
        self.model: Optional[TriageMasterMLP] = None
    
    def fit(
        self,
        texts: List[str],
        numerical: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        val_data: Optional[Tuple] = None
    ) -> 'TRIAGEMASTER':
        """
        Entraîne le modèle TRIAGEMASTER.
        
        Args:
            texts: Textes des entretiens
            numerical: Features numériques (n_samples, n_features)
            labels: Labels (0 à num_classes-1)
            feature_names: Noms des features numériques
            val_data: Optionnel (texts_val, numerical_val, labels_val)
        """
        print("\n" + "=" * 60)
        print("  ENTRAÎNEMENT TRIAGEMASTER (Doc2Vec + MLP)")
        print("=" * 60)
        
        self.feature_names = feature_names
        n_numerical = numerical.shape[1]
        
        # 1. Entraîner Doc2Vec
        print("\n  → Entraînement Doc2Vec...")
        text_embeddings = self.doc2vec_encoder.fit_transform(texts)
        print(f"    ✓ Doc2Vec: {text_embeddings.shape}")
        
        # 2. Normaliser les features numériques
        numerical_scaled = self.scaler.fit_transform(numerical)
        
        # 3. Concaténer
        X = np.hstack([text_embeddings, numerical_scaled])
        input_dim = X.shape[1]
        print(f"    ✓ Input dim: {input_dim} (Doc2Vec: {self.doc2vec_dim}, Num: {n_numerical})")
        
        # 4. Préparer validation si fournie
        if val_data:
            val_texts, val_num, val_labels = val_data
            val_emb = self.doc2vec_encoder.transform(val_texts)
            val_num_scaled = self.scaler.transform(val_num)
            X_val = np.hstack([val_emb, val_num_scaled])
        
        # 5. Créer le modèle MLP
        self.model = TriageMasterMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)
        
        # 6. Calculer les poids de classe pour le déséquilibre
        class_counts = np.bincount(labels, minlength=self.num_classes)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * self.num_classes
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # 7. Loss et optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 8. DataLoader
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(labels)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 9. Entraînement
        print("\n  → Entraînement MLP...")
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(loader)
            
            # Validation
            val_loss = epoch_loss
            if val_data:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.FloatTensor(X_val).to(self.device)
                    y_val_t = torch.LongTensor(val_labels).to(self.device)
                    val_out = self.model(X_val_t)
                    val_loss = criterion(val_out, y_val_t).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f} - Val: {val_loss:.4f}")
            
            if patience_counter >= self.patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        # Charger le meilleur modèle
        if best_state:
            self.model.load_state_dict(best_state)
        
        self.is_fitted = True
        print("  ✓ Entraînement terminé!")
        
        return self
    
    def predict(self, texts: List[str], numerical: np.ndarray) -> np.ndarray:
        """Prédit les classes."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné. Appelez fit() d'abord.")
        
        self.model.eval()
        
        text_emb = self.doc2vec_encoder.transform(texts)
        num_scaled = self.scaler.transform(numerical)
        X = np.hstack([text_emb, num_scaled])
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
        
        return predictions
    
    def predict_proba(self, texts: List[str], numerical: np.ndarray) -> np.ndarray:
        """Prédit les probabilités par classe."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné. Appelez fit() d'abord.")
        
        self.model.eval()
        
        text_emb = self.doc2vec_encoder.transform(texts)
        num_scaled = self.scaler.transform(numerical)
        X = np.hstack([text_emb, num_scaled])
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            proba = F.softmax(outputs, dim=1).cpu().numpy()
        
        return proba
    
    def explain_shap(
        self,
        texts: List[str],
        numerical: np.ndarray,
        n_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Calcule les explications SHAP avec KernelExplainer.
        
        Note: KernelExplainer est model-agnostic mais plus lent que TreeExplainer.
        
        Args:
            texts: Textes à expliquer
            numerical: Features numériques
            n_samples: Échantillons de background
            
        Returns:
            Dict avec shap_values, feature_names, numerical_importance
        """
        import shap
        
        print("\n  → Calcul SHAP (KernelExplainer)...")
        
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné.")
        
        # Préparer les données
        text_emb = self.doc2vec_encoder.transform(texts)
        num_scaled = self.scaler.transform(numerical)
        X = np.hstack([text_emb, num_scaled])
        
        # Background data (échantillon)
        n_bg = min(n_samples, len(X))
        bg_indices = np.random.choice(len(X), n_bg, replace=False)
        background = X[bg_indices]
        
        # Fonction de prédiction pour SHAP
        def predict_fn(x):
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                return F.softmax(self.model(x_tensor), dim=1).cpu().numpy()
        
        # KernelExplainer
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X[:n_samples])
        
        # Noms des features
        doc2vec_names = [f"doc2vec_{i}" for i in range(self.doc2vec_dim)]
        all_feature_names = doc2vec_names + self.feature_names
        
        # Importance des features numériques uniquement
        num_start = self.doc2vec_dim
        if isinstance(shap_values, list):
            num_shap = [sv[:, num_start:] for sv in shap_values]
        else:
            num_shap = shap_values[:, num_start:]
        
        num_importance = SHAPHelper.compute_feature_importance(
            num_shap, self.feature_names
        )
        
        SHAPHelper.print_importance(
            num_importance, 
            "Features numériques (TRIAGEMASTER)"
        )
        
        print("  ✓ SHAP calculé!")
        
        return {
            'shap_values': shap_values,
            'feature_names': all_feature_names,
            'numerical_importance': num_importance,
            'explainer': explainer
        }
