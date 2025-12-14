"""
EMERGINET - JEPA + VICReg
=========================

Architecture auto-supervisée avec réévaluation continue.

Pipeline:
    1. Texte → FlauBERT → encoder texte
    2. Numerical → MLP → encoder numérique
    3. JEPA: prédiction latente cross-modale
    4. VICReg: régularisation variance-invariance-covariance
    5. Classification finale
    6. Réévaluation continue basée sur évolution temporelle

Explicabilité: Integrated Gradients (Captum)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
import warnings

from src.models.base import BaseTriageModel, SHAPHelper, RANDOM_SEED

warnings.filterwarnings('ignore')


class JEPAEncoder(nn.Module):
    """
    Encodeur JEPA (Joint Embedding Predictive Architecture).
    
    Combine les représentations texte et numérique dans un espace latent commun.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        num_dim: int = 10,
        hidden_dim: int = 256,
        latent_dim: int = 128
    ):
        super().__init__()
        
        # Encodeur texte
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Encodeur numérique
        self.num_encoder = nn.Sequential(
            nn.Linear(num_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Prédicteur cross-modal (JEPA)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
    
    def forward(
        self,
        text_emb: torch.Tensor,
        num_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_emb: Embeddings texte (batch, text_dim)
            num_features: Features numériques (batch, num_dim)
            
        Returns:
            Dict avec z_text, z_num, z_pred, z_combined
        """
        # Encoder
        z_text = self.text_encoder(text_emb)
        z_num = self.num_encoder(num_features)
        
        # JEPA: prédire z_num depuis z_text
        z_pred = self.predictor(z_text)
        
        # Fusion
        z_combined = self.fusion(torch.cat([z_text, z_num], dim=1))
        
        return {
            'z_text': z_text,
            'z_num': z_num,
            'z_pred': z_pred,
            'z_combined': z_combined
        }


class VICRegLoss(nn.Module):
    """
    Perte VICReg (Variance-Invariance-Covariance Regularization).
    
    Encourage:
        - Invariance: représentations similaires pour même patient
        - Variance: éviter l'effondrement (collapse)
        - Covariance: décorrélation des dimensions
    """
    
    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z1, z2: Deux représentations à aligner
            
        Returns:
            (loss_total, dict des composantes)
        """
        # Invariance (MSE)
        sim_loss = F.mse_loss(z1, z2)
        
        # Variance (éviter collapse)
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        
        # Covariance (décorrélation)
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        
        cov_z1 = (z1_centered.T @ z1_centered) / (z1.size(0) - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (z2.size(0) - 1)
        
        # Off-diagonal elements
        cov_loss = (
            self._off_diagonal(cov_z1).pow(2).sum() / z1.size(1) +
            self._off_diagonal(cov_z2).pow(2).sum() / z2.size(1)
        )
        
        total = (
            self.sim_weight * sim_loss +
            self.var_weight * var_loss +
            self.cov_weight * cov_loss
        )
        
        return total, {
            'sim': sim_loss.item(),
            'var': var_loss.item(),
            'cov': cov_loss.item()
        }
    
    def _off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les éléments hors diagonale."""
        n = x.size(0)
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class EMERGINETModel(nn.Module):
    """Modèle complet EMERGINET."""
    
    def __init__(
        self,
        text_dim: int = 768,
        num_dim: int = 10,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_classes: int = 4
    ):
        super().__init__()
        
        self.jepa = JEPAEncoder(text_dim, num_dim, hidden_dim, latent_dim)
        
        # Classifieur
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim // 2, num_classes)
        )
    
    def forward(
        self,
        text_emb: torch.Tensor,
        num_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass complet."""
        jepa_out = self.jepa(text_emb, num_features)
        logits = self.classifier(jepa_out['z_combined'])
        
        return {
            **jepa_out,
            'logits': logits
        }


class EMERGINET(BaseTriageModel):
    """
    Modèle EMERGINET: JEPA + VICReg avec réévaluation continue.
    
    Architecture:
        - Encodeur texte: FlauBERT
        - Encodeur numérique: MLP
        - JEPA: prédiction cross-modale
        - VICReg: régularisation auto-supervisée
        - Explicabilité: Integrated Gradients
    
    Fonctionnalité unique:
        - Réévaluation continue basée sur évolution temporelle
        - Détection automatique de dégradation
    
    Performances attendues:
        - Taux d'erreur: ~10%
        - Latence: ~240ms
    
    Example:
        >>> model = EMERGINET(epochs=50)
        >>> model.fit(texts, numerical, labels, feature_names)
        >>> predictions = model.predict(texts_test, numerical_test)
        >>> 
        >>> # Réévaluation après 30 minutes
        >>> new_pred = model.reevaluate(patient_state, delta_time=30)
    """
    
    def __init__(
        self,
        bert_model: str = 'flaubert/flaubert_base_cased',
        max_length: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        vicreg_weight: float = 0.1,
        batch_size: int = 16,
        epochs: int = 50,
        lr: float = 1e-4,
        patience: int = 10,
        num_classes: int = 4,
        device: str = 'auto'
    ):
        """
        Args:
            bert_model: Modèle HuggingFace pour le texte
            max_length: Longueur max tokens
            hidden_dim: Dimension couche cachée JEPA
            latent_dim: Dimension espace latent
            vicreg_weight: Poids de la perte VICReg
            batch_size: Taille batch
            epochs: Nombre max époques
            lr: Learning rate
            patience: Early stopping
            num_classes: Nombre de classes
            device: Device PyTorch
        """
        super().__init__(num_classes=num_classes, device=device)
        
        self.bert_model_name = bert_model
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vicreg_weight = vicreg_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        
        self.tokenizer = None
        self.bert_model = None
        self.model: Optional[EMERGINETModel] = None
        self.vicreg_loss = VICRegLoss()
    
    def _load_bert(self) -> None:
        """Charge FlauBERT."""
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = AutoModel.from_pretrained(self.bert_model_name).to(self.device)
        self.bert_model.eval()
        
        # Freeze BERT (on utilise juste les embeddings)
        for param in self.bert_model.parameters():
            param.requires_grad = False
    
    def _get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extrait les embeddings CLS."""
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                ids = encoded['input_ids'][i:i+self.batch_size].to(self.device)
                mask = encoded['attention_mask'][i:i+self.batch_size].to(self.device)
                
                outputs = self.bert_model(input_ids=ids, attention_mask=mask)
                cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls)
        
        return np.vstack(embeddings)
    
    def fit(
        self,
        texts: List[str],
        numerical: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        val_data: Optional[Tuple] = None
    ) -> 'EMERGINET':
        """
        Entraîne EMERGINET avec JEPA + VICReg.
        
        La perte totale combine:
            - Classification (CrossEntropy)
            - JEPA (prédiction cross-modale)
            - VICReg (régularisation)
        """
        print("\n" + "=" * 60)
        print("  ENTRAÎNEMENT EMERGINET (JEPA + VICReg)")
        print("=" * 60)
        
        self.feature_names = feature_names
        num_dim = numerical.shape[1]
        
        # 1. Charger BERT
        print("\n  → Chargement FlauBERT...")
        self._load_bert()
        
        # 2. Extraire embeddings
        print("\n  → Extraction embeddings texte...")
        text_embeddings = self._get_bert_embeddings(texts)
        text_dim = text_embeddings.shape[1]
        print(f"    ✓ Text dim: {text_dim}")
        
        # 3. Normaliser features numériques
        numerical_scaled = self.scaler.fit_transform(numerical)
        
        # 4. Créer le modèle
        self.model = EMERGINETModel(
            text_dim=text_dim,
            num_dim=num_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_classes=self.num_classes
        ).to(self.device)
        
        # 5. Poids de classe
        class_counts = np.bincount(labels, minlength=self.num_classes)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * self.num_classes
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # 6. Losses et optimizer
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        
        # 7. DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(text_embeddings),
            torch.FloatTensor(numerical_scaled),
            torch.LongTensor(labels)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation
        if val_data:
            val_texts, val_num, val_labels = val_data
            val_text_emb = self._get_bert_embeddings(val_texts)
            val_num_scaled = self.scaler.transform(val_num)
        
        # 8. Entraînement
        print(f"\n  → Entraînement ({self.epochs} époques max)...")
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            epoch_ce = 0
            epoch_jepa = 0
            epoch_vic = 0
            
            for text_emb, num_feat, y in loader:
                text_emb = text_emb.to(self.device)
                num_feat = num_feat.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                outputs = self.model(text_emb, num_feat)
                
                # Classification loss
                loss_ce = ce_loss(outputs['logits'], y)
                
                # JEPA loss (prédire z_num depuis z_text)
                loss_jepa = F.mse_loss(outputs['z_pred'], outputs['z_num'].detach())
                
                # VICReg loss
                loss_vic, _ = self.vicreg_loss(outputs['z_text'], outputs['z_num'])
                
                # Total
                total_loss = loss_ce + loss_jepa + self.vicreg_weight * loss_vic
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_ce += loss_ce.item()
                epoch_jepa += loss_jepa.item()
                epoch_vic += loss_vic.item()
            
            scheduler.step()
            
            n_batches = len(loader)
            epoch_loss /= n_batches
            
            # Validation
            val_loss = epoch_loss
            if val_data:
                self.model.eval()
                with torch.no_grad():
                    val_t = torch.FloatTensor(val_text_emb).to(self.device)
                    val_n = torch.FloatTensor(val_num_scaled).to(self.device)
                    val_y = torch.LongTensor(val_labels).to(self.device)
                    val_out = self.model(val_t, val_n)
                    val_loss = ce_loss(val_out['logits'], val_y).item()
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} - "
                      f"Loss: {epoch_loss:.4f} (CE: {epoch_ce/n_batches:.4f}, "
                      f"JEPA: {epoch_jepa/n_batches:.4f}, VIC: {epoch_vic/n_batches:.4f}) - "
                      f"Val: {val_loss:.4f}")
            
            if patience_counter >= self.patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        self.is_fitted = True
        print("  ✓ Entraînement terminé!")
        
        return self
    
    def predict(self, texts: List[str], numerical: np.ndarray) -> np.ndarray:
        """Prédit les classes."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné.")
        
        self.model.eval()
        text_emb = self._get_bert_embeddings(texts)
        num_scaled = self.scaler.transform(numerical)
        
        with torch.no_grad():
            t = torch.FloatTensor(text_emb).to(self.device)
            n = torch.FloatTensor(num_scaled).to(self.device)
            outputs = self.model(t, n)
            predictions = outputs['logits'].argmax(dim=1).cpu().numpy()
        
        return predictions
    
    def predict_proba(self, texts: List[str], numerical: np.ndarray) -> np.ndarray:
        """Prédit les probabilités."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné.")
        
        self.model.eval()
        text_emb = self._get_bert_embeddings(texts)
        num_scaled = self.scaler.transform(numerical)
        
        with torch.no_grad():
            t = torch.FloatTensor(text_emb).to(self.device)
            n = torch.FloatTensor(num_scaled).to(self.device)
            outputs = self.model(t, n)
            proba = F.softmax(outputs['logits'], dim=1).cpu().numpy()
        
        return proba
    
    def reevaluate(
        self,
        patient_state: Dict[str, Any],
        delta_time: float = 30.0,
        threshold_degradation: float = 0.15
    ) -> Dict[str, Any]:
        """
        Réévalue un patient basé sur l'évolution temporelle.
        
        Fonctionnalité unique d'EMERGINET: détecte les dégradations
        et propose une reclassification.
        
        Args:
            patient_state: Dict avec 'text', 'numerical', 'initial_prediction',
                          et optionnellement 'initial_proba'
            delta_time: Temps écoulé en minutes depuis le triage initial
            threshold_degradation: Seuil pour déclencher une alerte
            
        Returns:
            Dict avec:
                - new_prediction: nouvelle prédiction
                - new_proba: nouvelles probabilités
                - degradation_score: score de dégradation détecté
                - alert: bool si dégradation significative
                - recommendation: texte de recommandation
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné.")
        
        text = patient_state['text']
        numerical = patient_state['numerical']
        initial_pred = patient_state.get('initial_prediction', None)
        initial_proba = patient_state.get('initial_proba', None)
        
        # Nouvelle prédiction
        new_proba = self.predict_proba([text], numerical.reshape(1, -1))[0]
        new_pred = new_proba.argmax()
        
        # Calculer le score de dégradation
        degradation_score = 0.0
        
        if initial_proba is not None:
            # Shift vers classes plus graves (indices plus hauts)
            prob_shift = 0
            for i in range(self.num_classes):
                prob_shift += i * (new_proba[i] - initial_proba[i])
            degradation_score = max(0, prob_shift)
        
        # Facteur temporel (dégradation plus probable avec le temps)
        time_factor = min(1.0, delta_time / 60.0)  # Normaliser sur 1h
        degradation_score *= (1 + 0.5 * time_factor)
        
        # Déterminer s'il y a alerte
        alert = degradation_score > threshold_degradation
        
        # Vérifier aussi les constantes critiques
        # (indices typiques: SpO2=4, FC=1, PAS=2)
        critical_alert = False
        if numerical.shape[0] >= 5:
            spo2_idx = 4  # SpO2
            fc_idx = 1    # FC
            if numerical[spo2_idx] < 92 or numerical[fc_idx] > 120:
                critical_alert = True
        
        # Générer recommandation
        if critical_alert:
            recommendation = "⚠️ ALERTE CRITIQUE: Constantes anormales détectées. Réévaluation médicale immédiate recommandée."
        elif alert:
            recommendation = f"⚠️ Dégradation détectée (score: {degradation_score:.2f}). Considérer reclassification de niveau {initial_pred} → {new_pred}."
        elif new_pred > (initial_pred or 0):
            recommendation = f"ℹ️ Légère évolution observée. Surveillance renforcée conseillée."
        else:
            recommendation = "✓ État stable. Pas de reclassification nécessaire."
        
        return {
            'new_prediction': int(new_pred),
            'new_proba': new_proba.tolist(),
            'degradation_score': float(degradation_score),
            'alert': alert or critical_alert,
            'critical_alert': critical_alert,
            'recommendation': recommendation,
            'delta_time': delta_time
        }
    
    def explain_shap(
        self,
        texts: List[str],
        numerical: np.ndarray,
        n_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Explicabilité via SHAP (fallback, préférer Integrated Gradients).
        """
        import shap
        
        print("\n  → Calcul SHAP (KernelExplainer)...")
        
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné.")
        
        text_emb = self._get_bert_embeddings(texts[:n_samples])
        num_scaled = self.scaler.transform(numerical[:n_samples])
        X = np.hstack([text_emb, num_scaled])
        
        def predict_fn(x):
            self.model.eval()
            # Split back
            text_part = x[:, :text_emb.shape[1]]
            num_part = x[:, text_emb.shape[1]:]
            with torch.no_grad():
                t = torch.FloatTensor(text_part).to(self.device)
                n = torch.FloatTensor(num_part).to(self.device)
                out = self.model(t, n)
                return F.softmax(out['logits'], dim=1).cpu().numpy()
        
        bg_idx = np.random.choice(len(X), min(50, len(X)), replace=False)
        explainer = shap.KernelExplainer(predict_fn, X[bg_idx])
        shap_values = explainer.shap_values(X)
        
        # Importance numérique
        text_dim = text_emb.shape[1]
        if isinstance(shap_values, list):
            num_shap = [sv[:, text_dim:] for sv in shap_values]
        else:
            num_shap = shap_values[:, text_dim:]
        
        num_importance = SHAPHelper.compute_feature_importance(
            num_shap, self.feature_names
        )
        SHAPHelper.print_importance(num_importance, "Features numériques (EMERGINET)")
        
        return {
            'shap_values': shap_values,
            'numerical_importance': num_importance
        }
    
    def explain_integrated_gradients(
        self,
        texts: List[str],
        numerical: np.ndarray,
        n_samples: int = 30,
        n_steps: int = 50
    ) -> Dict[str, Any]:
        """
        Explicabilité via Integrated Gradients (Captum).
        
        Plus adapté aux réseaux de neurones que SHAP.
        
        Args:
            texts: Textes à expliquer
            numerical: Features numériques
            n_samples: Nombre d'échantillons
            n_steps: Pas d'intégration
            
        Returns:
            Dict avec attributions par feature
        """
        from captum.attr import IntegratedGradients
        
        print("\n  → Calcul Integrated Gradients...")
        
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné.")
        
        self.model.eval()
        n = min(n_samples, len(texts))
        
        text_emb = self._get_bert_embeddings(texts[:n])
        num_scaled = self.scaler.transform(numerical[:n])
        
        # Wrapper pour IG
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, text_emb, num_feat):
                out = self.model(text_emb, num_feat)
                return out['logits']
        
        wrapper = ModelWrapper(self.model)
        ig = IntegratedGradients(wrapper)
        
        # Baselines (zéros)
        text_baseline = torch.zeros_like(torch.FloatTensor(text_emb)).to(self.device)
        num_baseline = torch.zeros_like(torch.FloatTensor(num_scaled)).to(self.device)
        
        text_tensor = torch.FloatTensor(text_emb).to(self.device)
        num_tensor = torch.FloatTensor(num_scaled).to(self.device)
        
        # Target = classe prédite
        with torch.no_grad():
            out = self.model(text_tensor, num_tensor)
            targets = out['logits'].argmax(dim=1)
        
        # Calcul IG
        attr_text, attr_num = ig.attribute(
            (text_tensor, num_tensor),
            baselines=(text_baseline, num_baseline),
            target=targets,
            n_steps=n_steps,
            return_convergence_delta=False
        )
        
        # Importance des features numériques
        num_importance = attr_num.abs().mean(dim=0).cpu().numpy()
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': num_importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        SHAPHelper.print_importance(
            importance_df,
            "Features numériques - Integrated Gradients (EMERGINET)"
        )
        
        print("  ✓ Integrated Gradients calculé!")
        
        return {
            'text_attributions': attr_text.cpu().numpy(),
            'numerical_attributions': attr_num.cpu().numpy(),
            'numerical_importance': importance_df,
            'targets': targets.cpu().numpy()
        }
    
    def visualize_latent_space(
        self,
        texts: List[str],
        numerical: np.ndarray,
        labels: np.ndarray,
        n_samples: int = 500
    ) -> Dict[str, np.ndarray]:
        """
        Visualise l'espace latent JEPA avec t-SNE.
        
        Utile pour vérifier la qualité des représentations apprises.
        """
        from sklearn.manifold import TSNE
        
        print("\n  → Analyse espace latent (t-SNE)...")
        
        self.model.eval()
        n = min(n_samples, len(texts))
        
        text_emb = self._get_bert_embeddings(texts[:n])
        num_scaled = self.scaler.transform(numerical[:n])
        
        with torch.no_grad():
            t = torch.FloatTensor(text_emb).to(self.device)
            n_t = torch.FloatTensor(num_scaled).to(self.device)
            out = self.model(t, n_t)
            embeddings = out['z_combined'].cpu().numpy()
        
        # t-SNE
        perplexity = min(30, n - 1)
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        print("  ✓ t-SNE calculé!")
        
        return {
            'embeddings': embeddings,
            'embeddings_2d': embeddings_2d,
            'labels': labels[:n]
        }
