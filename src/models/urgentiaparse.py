"""
URGENTIAPARSE - FlauBERT + XGBoost
==================================

Architecture LLM avec Gradient Boosting et double explicabilité.

Pipeline:
    1. Texte → FlauBERT (fine-tuné) → CLS embedding (768 dim)
    2. Concat avec features numériques normalisées
    3. XGBoost classifier
    4. Explicabilité: SHAP TreeExplainer (rapide) + Attention visualization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
import warnings

from src.models.base import BaseTriageModel, SHAPHelper, RANDOM_SEED

warnings.filterwarnings('ignore')


class URGENTIAPARSE(BaseTriageModel):
    """
    Modèle URGENTIAPARSE: FlauBERT + XGBoost avec double explicabilité.
    
    Architecture:
        - Encodeur texte: FlauBERT (fine-tunable)
        - Classifieur: XGBoost Gradient Boosting
        - Explicabilité: SHAP TreeExplainer + Attention
    
    Performances attendues:
        - Taux d'erreur: ~25%
        - Latence: ~380ms
    
    Example:
        >>> model = URGENTIAPARSE(fine_tune_epochs=3)
        >>> model.fit(texts, numerical, labels, feature_names)
        >>> predictions = model.predict(texts_test, numerical_test)
        >>> shap_results = model.explain_shap(texts_test, numerical_test)
        >>> model.explain_attention(texts_test[:5])
    """
    
    def __init__(
        self,
        bert_model: str = 'flaubert/flaubert_base_cased',
        fine_tune_epochs: int = 3,
        max_length: int = 128,
        batch_size: int = 16,
        lr: float = 2e-5,
        xgb_params: Optional[Dict] = None,
        num_classes: int = 4,
        device: str = 'auto'
    ):
        """
        Args:
            bert_model: Nom du modèle HuggingFace
            fine_tune_epochs: Époques de fine-tuning BERT
            max_length: Longueur max des tokens
            batch_size: Taille des batches pour BERT
            lr: Learning rate pour fine-tuning
            xgb_params: Paramètres XGBoost personnalisés
            num_classes: Nombre de classes
            device: 'auto', 'cuda', 'mps', ou 'cpu'
        """
        super().__init__(num_classes=num_classes, device=device)
        
        self.bert_model_name = bert_model
        self.fine_tune_epochs = fine_tune_epochs
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        
        # Paramètres XGBoost par défaut
        self.xgb_params = xgb_params or {
            'max_depth': 6,
            'n_estimators': 200,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }
        
        self.tokenizer = None
        self.bert_model = None
        self.xgb_model = None
        self._attention_weights = None
    
    def _load_bert(self) -> None:
        """Charge le modèle FlauBERT."""
        from transformers import AutoTokenizer, AutoModel
        
        print(f"    Chargement {self.bert_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = AutoModel.from_pretrained(
            self.bert_model_name,
            output_attentions=True
        ).to(self.device)
    
    def _get_bert_embeddings(
        self,
        texts: List[str],
        store_attention: bool = False
    ) -> np.ndarray:
        """Extrait les embeddings CLS de BERT."""
        self.bert_model.eval()
        
        # Tokenisation
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        embeddings = []
        all_attentions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_ids = encoded['input_ids'][i:i+self.batch_size].to(self.device)
                batch_mask = encoded['attention_mask'][i:i+self.batch_size].to(self.device)
                
                outputs = self.bert_model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask
                )
                
                # CLS token embedding
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)
                
                if store_attention:
                    # Moyenne des attentions sur les heads et layers
                    attn = torch.stack(outputs.attentions).mean(dim=(0, 2))
                    all_attentions.append(attn.cpu().numpy())
        
        if store_attention and all_attentions:
            self._attention_weights = np.vstack(all_attentions)
            self._last_tokens = encoded['input_ids']
        
        return np.vstack(embeddings)
    
    def fit(
        self,
        texts: List[str],
        numerical: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        val_data: Optional[Tuple] = None
    ) -> 'URGENTIAPARSE':
        """
        Entraîne le modèle URGENTIAPARSE.
        
        Processus:
            1. Fine-tune FlauBERT (optionnel)
            2. Extraire embeddings CLS
            3. Entraîner XGBoost sur embeddings + features numériques
        """
        print("\n" + "=" * 60)
        print("  ENTRAÎNEMENT URGENTIAPARSE (FlauBERT + XGBoost)")
        print("=" * 60)
        
        self.feature_names = feature_names
        
        # 1. Charger BERT
        print("\n  → Chargement FlauBERT...")
        self._load_bert()
        
        # 2. Fine-tuning optionnel
        if self.fine_tune_epochs > 0:
            print(f"\n  → Fine-tuning ({self.fine_tune_epochs} époques)...")
            self._fine_tune_bert(texts, labels)
        
        # 3. Extraire embeddings
        print("\n  → Extraction des embeddings...")
        text_embeddings = self._get_bert_embeddings(texts)
        print(f"    ✓ BERT embeddings: {text_embeddings.shape}")
        
        # 4. Normaliser features numériques
        numerical_scaled = self.scaler.fit_transform(numerical)
        
        # 5. Concaténer
        X = np.hstack([text_embeddings, numerical_scaled])
        print(f"    ✓ Input dim: {X.shape[1]} (BERT: {text_embeddings.shape[1]}, Num: {numerical.shape[1]})")
        
        # 6. Entraîner XGBoost
        print("\n  → Entraînement XGBoost...")
        import xgboost as xgb
        
        self.xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=self.num_classes,
            **self.xgb_params
        )
        
        # Validation si fournie
        eval_set = None
        if val_data:
            val_texts, val_num, val_labels = val_data
            val_emb = self._get_bert_embeddings(val_texts)
            val_num_scaled = self.scaler.transform(val_num)
            X_val = np.hstack([val_emb, val_num_scaled])
            eval_set = [(X_val, val_labels)]
        
        self.xgb_model.fit(
            X, labels,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_fitted = True
        print("  ✓ Entraînement terminé!")
        
        return self
    
    def _fine_tune_bert(self, texts: List[str], labels: np.ndarray) -> None:
        """Fine-tune BERT pour la classification."""
        from torch.optim import AdamW
        
        # Ajouter une tête de classification
        classifier = nn.Linear(
            self.bert_model.config.hidden_size,
            self.num_classes
        ).to(self.device)
        
        # Tokenisation
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        dataset = TensorDataset(
            encoded['input_ids'],
            encoded['attention_mask'],
            torch.LongTensor(labels)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = AdamW([
            {'params': self.bert_model.parameters(), 'lr': self.lr},
            {'params': classifier.parameters(), 'lr': self.lr * 10}
        ])
        
        criterion = nn.CrossEntropyLoss()
        
        self.bert_model.train()
        for epoch in range(self.fine_tune_epochs):
            total_loss = 0
            for batch in loader:
                ids, mask, y = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.bert_model(input_ids=ids, attention_mask=mask)
                cls_out = outputs.last_hidden_state[:, 0, :]
                logits = classifier(cls_out)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"    Epoch {epoch+1}/{self.fine_tune_epochs} - Loss: {total_loss/len(loader):.4f}")
        
        self.bert_model.eval()
    
    def predict(self, texts: List[str], numerical: np.ndarray) -> np.ndarray:
        """Prédit les classes."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné.")
        
        text_emb = self._get_bert_embeddings(texts)
        num_scaled = self.scaler.transform(numerical)
        X = np.hstack([text_emb, num_scaled])
        
        return self.xgb_model.predict(X)
    
    def predict_proba(self, texts: List[str], numerical: np.ndarray) -> np.ndarray:
        """Prédit les probabilités par classe."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné.")
        
        text_emb = self._get_bert_embeddings(texts)
        num_scaled = self.scaler.transform(numerical)
        X = np.hstack([text_emb, num_scaled])
        
        return self.xgb_model.predict_proba(X)
    
    def explain_shap(
        self,
        texts: List[str],
        numerical: np.ndarray,
        n_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Calcule les explications SHAP avec TreeExplainer.
        
        TreeExplainer est ~100x plus rapide que KernelExplainer car
        il exploite la structure de l'arbre XGBoost.
        """
        import shap
        
        print("\n  → Calcul SHAP (TreeExplainer - rapide)...")
        
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné.")
        
        # Préparer les données
        text_emb = self._get_bert_embeddings(texts[:n_samples])
        num_scaled = self.scaler.transform(numerical[:n_samples])
        X = np.hstack([text_emb, num_scaled])
        
        # TreeExplainer (beaucoup plus rapide)
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X)
        
        # Noms des features
        bert_dim = text_emb.shape[1]
        bert_names = [f"bert_{i}" for i in range(bert_dim)]
        all_feature_names = bert_names + self.feature_names
        
        # Importance des features numériques uniquement
        if isinstance(shap_values, list):
            num_shap = [sv[:, bert_dim:] for sv in shap_values]
        else:
            num_shap = shap_values[:, bert_dim:]
        
        num_importance = SHAPHelper.compute_feature_importance(
            num_shap, self.feature_names
        )
        
        SHAPHelper.print_importance(
            num_importance,
            "Features numériques (URGENTIAPARSE)"
        )
        
        print("  ✓ SHAP calculé!")
        
        return {
            'shap_values': shap_values,
            'feature_names': all_feature_names,
            'numerical_importance': num_importance,
            'explainer': explainer
        }
    
    def explain_attention(
        self,
        texts: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Visualise les mots importants via l'attention BERT.
        
        Args:
            texts: Textes à analyser
            top_k: Nombre de tokens à afficher
            
        Returns:
            Liste de dicts avec tokens et scores d'attention
        """
        print("\n  → Analyse attention BERT...")
        
        if self.bert_model is None:
            raise ValueError("Le modèle n'est pas entraîné.")
        
        # Extraire embeddings avec attention
        _ = self._get_bert_embeddings(texts, store_attention=True)
        
        results = []
        
        for i, text in enumerate(texts):
            # Récupérer les tokens
            tokens = self.tokenizer.tokenize(text)[:self.max_length-2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            
            # Scores d'attention (moyenne sur la séquence)
            attn = self._attention_weights[i]
            # Attention du CLS vers les autres tokens
            cls_attention = attn[0, :len(tokens)]
            
            # Top-k tokens
            top_indices = np.argsort(cls_attention)[-top_k:][::-1]
            
            token_scores = [
                (tokens[j], float(cls_attention[j]))
                for j in top_indices
                if j < len(tokens)
            ]
            
            results.append({
                'text': text,
                'tokens': tokens,
                'attention_scores': cls_attention,
                'top_tokens': token_scores
            })
            
            # Affichage
            print(f"\n    Texte {i+1}: {text[:80]}...")
            print("    Mots importants (attention):")
            for token, score in token_scores[:5]:
                bar = "█" * int(score * 50)
                print(f"      {token:15} {score:.4f} {bar}")
        
        print("\n  ✓ Analyse attention terminée!")
        
        return results
