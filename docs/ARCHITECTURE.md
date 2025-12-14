# Architecture EIMLIA-TEU

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EIMLIA-TEU                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ TRIAGEMASTER │    │URGENTIAPARSE │    │  EMERGINET   │               │
│  │  Doc2Vec+MLP │    │FlauBERT+XGB  │    │ JEPA+VICReg  │               │
│  │   ~39% err   │    │   ~25% err   │    │   ~10% err   │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                    │                   │                       │
│         └────────────────────┴───────────────────┘                       │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │                    API FastAPI                                 │      │
│  │  POST /predict    GET /explain    POST /simulation            │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│         ┌────────────────────┼────────────────────┐                     │
│         ▼                    ▼                    ▼                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │    SimPy     │◄──►│    Mesa      │◄──►│   PM4Py     │               │
│  │     DES      │    │     SMA      │    │   Mining    │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Composants principaux

### 1. Modèles d'IA (`src/models/`)

#### TRIAGEMASTER
- **Architecture**: Doc2Vec (100 dim) → MLP (128→64→5)
- **Entrée**: Texte verbatim + features numériques
- **Explicabilité**: SHAP KernelExplainer
- **Latence**: ~120ms
- **Performance**: MAE ~0.39

```python
from src.models import TRIAGEMASTER

model = TRIAGEMASTER(doc2vec_dim=100, epochs=50)
model.fit(texts, numerical, labels, feature_names)
predictions = model.predict(test_texts, test_numerical)
shap_values, importance = model.explain_shap(text, numerical)
```

#### URGENTIAPARSE
- **Architecture**: FlauBERT (fine-tuned) → XGBoost
- **Entrée**: Texte + features numériques
- **Explicabilité**: SHAP TreeExplainer + Attention visualization
- **Latence**: ~380ms
- **Performance**: MAE ~0.25

```python
from src.models import URGENTIAPARSE

model = URGENTIAPARSE(fine_tune_epochs=3)
model.fit(texts, numerical, labels, feature_names)
attention_weights = model.explain_attention(text)
```

#### EMERGINET
- **Architecture**: FlauBERT → JEPA Encoder → VICReg → Classifier
- **Entrée**: Texte + features numériques + temps
- **Explicabilité**: Integrated Gradients (Captum)
- **Fonctionnalité**: Réévaluation continue
- **Latence**: ~240ms
- **Performance**: MAE ~0.10

```python
from src.models import EMERGINET

model = EMERGINET(latent_dim=128, vicreg_weight=0.1)
model.fit(texts, numerical, labels, feature_names)

# Réévaluation
result = model.reevaluate(patient_features, current_prediction, delta_time=30)
```

### 2. Simulation (`src/simulation/`)

#### Architecture hybride

```
┌─────────────────────────────────────────────────────────────┐
│                  SimulationHybride                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────┐         ┌─────────────────┐           │
│   │    SimPy DES    │◄───────►│   Mesa SMA      │           │
│   │                 │  sync   │                 │           │
│   │  - Ressources   │         │  - AgentPatient │           │
│   │  - Files        │         │  - AgentIAO     │           │
│   │  - Temps        │         │  - AgentMedecin │           │
│   │  - Flux         │         │  - AgentIA      │           │
│   └─────────────────┘         └─────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Scénarios

| Scénario | Agent IA | Charge | Description |
|----------|----------|--------|-------------|
| reference | Aucun | 1.0x | Triage manuel |
| nlp | TRIAGEMASTER | 1.0x | Doc2Vec + MLP |
| llm | URGENTIAPARSE | 1.0x | FlauBERT + XGBoost |
| jepa | EMERGINET | 1.0x | JEPA + réévaluation |
| crise | EMERGINET | 2.0x | Charge doublée |

### 3. Process Mining (`src/process_mining/`)

```python
from src.process_mining import ProcessMiningPipeline

pipeline = ProcessMiningPipeline('traces.csv')

# Découverte de processus
process_model = pipeline.discover_process('inductive')

# Conformance checking
metrics = pipeline.check_conformance(process_model)

# KPIs
kpis = pipeline.compute_kpis()
```

### 4. API (`src/api/`)

#### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Prédiction triage |
| POST | `/explain` | Explicabilité SHAP |
| GET | `/models` | Liste des modèles |
| POST | `/simulation/start` | Démarrer simulation |
| GET | `/simulation/status` | État simulation |
| GET | `/metrics` | Métriques Prometheus |

## Flux de données

```
                    ┌─────────────┐
                    │   Patient   │
                    │   Données   │
                    └──────┬──────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                      Preprocessing                            │
│  - Tokenization (texte)                                      │
│  - Normalisation (numériques)                                │
│  - Imputation (valeurs manquantes)                           │
└──────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ TRIAGEMASTER│  │URGENTIAPARSE│  │ EMERGINET  │
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │               │               │
          └───────────────┴───────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │    Prédiction       │
              │  FRENCH (1-5)       │
              │  + Probabilités     │
              │  + Alertes          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Explicabilité     │
              │  - SHAP values      │
              │  - Feature import.  │
              │  - Attention        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Interface IAO    │
              │  - Suggestion IA    │
              │  - Décision finale  │
              └─────────────────────┘
```

## Déploiement

### Docker

```bash
# Build
docker build -f docker/Dockerfile -t eimlia:latest .

# Run
docker-compose -f docker/docker-compose.yml up -d
```

### Kubernetes (optionnel)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eimlia-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: eimlia
  template:
    spec:
      containers:
      - name: api
        image: eimlia:latest
        ports:
        - containerPort: 8000
```

## Performance

### Benchmarks modèles

| Modèle | MAE | Kappa | Latence | GPU Mem |
|--------|-----|-------|---------|---------|
| TRIAGEMASTER | 0.39 | 0.72 | 120ms | 500 MB |
| URGENTIAPARSE | 0.25 | 0.81 | 380ms | 2 GB |
| EMERGINET | 0.10 | 0.91 | 240ms | 1.5 GB |

### Benchmarks simulation

- SimPy: ~50K patients/seconde
- Mesa: ~10K agents/seconde
- Hybride: ~5K patients/seconde

## Sécurité

- Données anonymisées (RGPD)
- API authentifiée (JWT optionnel)
- Logs audit
- Pas de données patient en clair dans les logs
