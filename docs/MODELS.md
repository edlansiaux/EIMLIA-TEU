# Guide des Modèles EIMLIA-TEU

## Vue d'ensemble

EIMLIA propose trois modèles d'IA pour l'assistance au triage FRENCH:

| Modèle | Architecture | Performance | Latence | Cas d'usage |
|--------|-------------|-------------|---------|-------------|
| TRIAGEMASTER | Doc2Vec + MLP | ~39% erreur | 120ms | Baseline, rapide |
| URGENTIAPARSE | FlauBERT + XGBoost | ~25% erreur | 380ms | Production, explicable |
| EMERGINET | JEPA + VICReg | ~10% erreur | 240ms | Optimal, réévaluation |

## 1. TRIAGEMASTER

### Architecture

```
Texte ──► Doc2Vec (100d) ──┐
                           ├──► Concat ──► MLP (128→64→5) ──► Prédiction
Features numériques (10d) ─┘
```

### Entraînement

```python
from src.models import TRIAGEMASTER, load_data

# Charger les données
texts, numerical, labels, features = load_data('data.xlsx')

# Créer le modèle
model = TRIAGEMASTER(
    doc2vec_dim=100,      # Dimension embeddings
    hidden_dims=[128, 64], # Couches cachées MLP
    dropout=0.3,          # Dropout
    lr=0.001,             # Learning rate
    epochs=100,           # Époques max
    patience=10           # Early stopping
)

# Entraîner
model.fit(texts, numerical, labels, features)
```

### Explicabilité

```python
# SHAP KernelExplainer (model-agnostic)
shap_values, importance = model.explain_shap(text, numerical)

# Afficher l'importance
for feature, imp in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"{feature}: {imp:.3f}")
```

**Note**: KernelExplainer est lent (~1-2s par exemple) car model-agnostic.

### Hyperparamètres recommandés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| doc2vec_dim | 100 | Dimension des embeddings |
| epochs | 100 | Époques max |
| patience | 10 | Early stopping |
| batch_size | 32 | Taille batch |
| lr | 0.001 | Learning rate |

---

## 2. URGENTIAPARSE

### Architecture

```
Texte ──► FlauBERT ──► [CLS] (768d) ──┐
                                       ├──► Concat ──► XGBoost ──► Prédiction
Features numériques (10d) ────────────┘
```

### Entraînement

```python
from src.models import URGENTIAPARSE

model = URGENTIAPARSE(
    bert_model='flaubert/flaubert_base_cased',
    fine_tune_epochs=3,   # Fine-tuning BERT
    max_length=128,       # Longueur max texte
    batch_size=16,        # Batch size BERT
    xgb_n_estimators=200  # Arbres XGBoost
)

model.fit(texts, numerical, labels, features)
```

### Fine-tuning BERT

Le fine-tuning est optionnel mais améliore les performances:

```python
# Sans fine-tuning (plus rapide)
model = URGENTIAPARSE(fine_tune_epochs=0)

# Avec fine-tuning (meilleur)
model = URGENTIAPARSE(fine_tune_epochs=3, lr_bert=2e-5)
```

### Explicabilité

```python
# SHAP TreeExplainer (rapide, ~100x plus rapide que Kernel)
shap_values, importance = model.explain_shap(text, numerical)

# Attention BERT (mots importants)
attention_weights = model.explain_attention(text)
for word, weight in attention_weights[:10]:
    print(f"{word}: {weight:.3f}")
```

### Hyperparamètres recommandés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| fine_tune_epochs | 3 | Époques fine-tuning BERT |
| max_length | 128 | Longueur max tokens |
| batch_size | 16 | Batch size |
| xgb_max_depth | 6 | Profondeur arbres |
| xgb_n_estimators | 200 | Nombre d'arbres |

---

## 3. EMERGINET

### Architecture

```
                    ┌──────────────────────────────────────┐
                    │           JEPA Encoder                │
Texte ──► FlauBERT ─┤  text_encoder: 768 → 256 → 128       │
                    │  num_encoder: 10 → 128 → 128         │──► z_combined
Features ───────────┤  predictor: cross-modal prediction   │      (256d)
                    │  fusion: concat                       │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │  Classifier  │──► Prédiction
                              │   256 → 5    │
                              └──────────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │   VICReg     │
                              │  Régulariser │
                              └──────────────┘
```

### JEPA (Joint Embedding Predictive Architecture)

JEPA apprend des représentations en prédisant les embeddings d'une modalité à partir d'une autre:

```
z_text ──► predictor ──► z_num_pred ≈ z_num
```

### VICReg Loss

Régularisation auto-supervisée:
- **Invariance**: z1 ≈ z2 (MSE)
- **Variance**: éviter le collapse (std ≥ 1)
- **Covariance**: décorréler les dimensions

```python
# Poids VICReg
vicreg_weight = 0.1  # Loss = CE + JEPA + 0.1 * VICReg
```

### Entraînement

```python
from src.models import EMERGINET

model = EMERGINET(
    bert_model='flaubert/flaubert_base_cased',
    hidden_dim=256,       # Dimension cachée
    latent_dim=128,       # Dimension latente
    vicreg_weight=0.1,    # Poids VICReg
    epochs=50,            # Époques
    batch_size=16,        # Batch size
    patience=10           # Early stopping
)

model.fit(texts, numerical, labels, features)
```

### Réévaluation continue

Fonctionnalité unique d'EMERGINET: réévaluer un patient après un délai.

```python
# Prédiction initiale
initial_pred = model.predict([text], numerical)

# Après 30 minutes
result = model.reevaluate(
    numerical_features=patient_features,
    current_prediction=initial_pred[0] + 1,  # 1-indexed
    delta_time=30  # minutes
)

print(f"Nouveau niveau suggéré: {result['nouveau_niveau']}")
print(f"Score dégradation: {result['degradation_score']:.2f}")
print(f"Alertes: {result['alertes']}")
```

### Explicabilité

```python
# Integrated Gradients (Captum)
ig_attributions = model.explain_integrated_gradients(text, numerical)

# Visualisation espace latent (t-SNE)
fig = model.visualize_latent_space(texts, numerical, labels)
```

### Hyperparamètres recommandés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| hidden_dim | 256 | Dimension cachée |
| latent_dim | 128 | Dimension latente |
| vicreg_weight | 0.1 | Poids VICReg loss |
| epochs | 50 | Époques |
| batch_size | 16 | Batch size |
| lr | 1e-4 | Learning rate |

---

## Comparaison des modèles

### Métriques

```python
from src.models import compare_models, evaluate_model

results = {}
for name, model in [('TRIAGEMASTER', tm), ('URGENTIAPARSE', up), ('EMERGINET', em)]:
    preds = model.predict(test_texts, test_numerical)
    results[name] = evaluate_model(test_labels, preds, name)

compare_models(results)
```

### Résultats typiques

| Métrique | TRIAGEMASTER | URGENTIAPARSE | EMERGINET |
|----------|--------------|---------------|-----------|
| MAE | 0.39 | 0.25 | 0.10 |
| RMSE | 0.62 | 0.45 | 0.28 |
| Kappa | 0.72 | 0.81 | 0.91 |
| Exact | 0.61 | 0.75 | 0.90 |
| Near (±1) | 0.95 | 0.98 | 0.99 |
| Sous-triage | 0.18 | 0.10 | 0.04 |

---

## Sauvegarde et chargement

```python
# Sauvegarder
model.save('models/emerginet.pkl')

# Charger
from src.models import EMERGINET
model = EMERGINET()
model.load('models/emerginet.pkl')
```

---

## GPU

Les modèles utilisent automatiquement le GPU si disponible:

```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"Device utilisé: {model.device}")
```

Pour forcer le CPU:
```python
model = EMERGINET()
model.device = 'cpu'
```

---

## Tips d'optimisation

1. **Batch size**: Augmenter si GPU a assez de mémoire
2. **Early stopping**: Utiliser patience=10-15
3. **Learning rate**: 1e-4 pour EMERGINET, 2e-5 pour fine-tuning BERT
4. **VICReg weight**: 0.1 est un bon défaut, augmenter si collapse
5. **Données**: Plus de données → meilleures performances
