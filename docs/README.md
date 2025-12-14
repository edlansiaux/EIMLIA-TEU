# Documentation EIMLIA-TEU

## Table des matières

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Guide d'utilisation](#guide-dutilisation)
5. [API Reference](#api-reference)
6. [Modèles IA](#modèles-ia)
7. [Simulation](#simulation)
8. [Process Mining](#process-mining)

---

## Introduction

### Contexte

L'étude EIMLIA-TEU vise à évaluer l'impact de l'assistance IA sur le triage aux urgences.

### Objectifs

1. **Comparer 3 architectures IA** pour l'aide au triage:
   - TRIAGEMASTER (NLP classique)
   - URGENTIAPARSE (LLM fine-tuné)
   - EMERGINET (JEPA + VICReg)

2. **Simuler l'impact** sur un flux de 100 000 patients virtuels

3. **Mesurer les KPIs clés**:
   - Durée Moyenne de Séjour (DMS)
   - Concordance du triage
   - Taux de sous/sur-triage

### Stack technique

| Composant | Technologie | Remplace |
|-----------|-------------|----------|
| Process Mining | PM4Py | Celonis |
| Simulation DES | SimPy | Arena |
| Simulation SMA | Mesa | AnyLogic |
| Deep Learning | PyTorch | - |
| NLP | FlauBERT, gensim | - |
| API | FastAPI | - |

**Économie estimée**: ~220-430K€/an vs stack propriétaire

---

## Architecture

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                        EIMLIA-TEU                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ TRIAGEMASTER│  │URGENTIAPARSE│  │  EMERGINET  │   Modèles IA │
│  │ Doc2Vec+MLP │  │FlauBERT+XGB │  │ JEPA+VICReg │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Simulation Hybride                          │    │
│  │  ┌───────────────┐        ┌───────────────┐             │    │
│  │  │    SimPy      │ <───> │     Mesa       │             │    │
│  │  │  (DES/Flux)   │  sync  │ (SMA/Agents)  │             │    │
│  │  └───────────────┘        └───────────────┘             │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Process Mining (PM4Py)                      │    │
│  │  Découverte │ Conformance │ Performance │ KPIs          │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    API FastAPI                           │    │
│  │  /predict │ /explain │ /simulation │ /metrics           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Structure du code

```
eimlia-teu/
├── src/
│   ├── models/           # Modèles IA
│   │   ├── triagemaster.py
│   │   ├── urgentiaparse.py
│   │   └── emerginet.py
│   ├── simulation/       # Simulation hybride
│   │   ├── simpy_des.py
│   │   ├── mesa_sma.py
│   │   └── orchestrator.py
│   ├── process_mining/   # Analyse PM4Py
│   ├── api/              # API REST
│   └── utils/            # Utilitaires
├── tests/                # Tests unitaires
├── config/               # Configuration
├── scripts/              # Scripts CLI
└── docker/               # Conteneurisation
```

---

## Installation

### Prérequis

- Python 3.10+
- pip ou conda
- 8 GB RAM minimum (16 GB recommandé pour EMERGINET)
- GPU optionnel (CUDA 11.8+ pour accélération)

### Installation standard

```bash
# Cloner le repository
git clone https://github.com/chu-lille/eimlia-teu.git
cd eimlia-teu

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer le package
pip install -e .
```

### Installation Docker

```bash
# Construire l'image
docker build -t eimlia -f docker/Dockerfile .

# Lancer le conteneur
docker run -p 8000:8000 eimlia
```

### Vérification

```python
from src.models import TRIAGEMASTER, URGENTIAPARSE, EMERGINET
from src.simulation import OrchestrateurSimulation

print("✓ Installation réussie!")
```

---

## Guide d'utilisation

### Entraînement des modèles

```bash
# Entraîner tous les modèles
python scripts/train_models.py --data data/data3.xlsx --model all

# Entraîner un modèle spécifique
python scripts/train_models.py --model emerginet --epochs 50
```

### Exécution des simulations

```bash
# Tous les scénarios
python scripts/run_simulation.py --scenario all --days 30

# Scénario JEPA uniquement
python scripts/run_simulation.py --scenario jepa --days 180

# Avec stress-tests
python scripts/run_simulation.py --scenario jepa --stress-test
```

### Utilisation de l'API

```bash
# Démarrer l'API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Documentation: http://localhost:8000/docs
```

Exemple de requête:

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "text": "Patient 65 ans, douleur thoracique depuis 2h",
    "numerical": {"age": 65, "fc": 95, "pas": 140, "spo2": 96},
    "model": "emerginet"
})

print(response.json())
# {"niveau_french": 2, "probabilites": [...], "confiance": 0.87, ...}
```

---

## API Reference

### Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Info API |
| `/health` | GET | Health check |
| `/predict` | POST | Prédiction triage |
| `/explain` | POST | Explications SHAP |
| `/simulation/start` | POST | Démarrer simulation |
| `/simulation/status` | GET | État simulation |
| `/models` | GET | Liste des modèles |
| `/metrics` | GET | Métriques Prometheus |

### Schémas

#### PatientInput

```json
{
  "text": "string",
  "numerical": {
    "age": 65,
    "fc": 95,
    "pas": 140,
    "pad": 85,
    "spo2": 96,
    "temperature": 37.2,
    "eva": 7
  },
  "model": "emerginet"
}
```

#### PredictionOutput

```json
{
  "niveau_french": 2,
  "probabilites": [0.05, 0.60, 0.25, 0.08, 0.02],
  "confiance": 0.60,
  "model_used": "emerginet",
  "timestamp": "2025-01-15T10:30:00",
  "alertes": ["SpO2 limite: 96%"]
}
```

---

## Modèles IA

### TRIAGEMASTER

**Architecture**: Doc2Vec (100 dim) + MLP (128→64→5)

- **Explicabilité**: SHAP KernelExplainer
- **Latence**: ~120ms
- **Erreur**: ~39%
- **Avantages**: Simple, rapide, interprétable

```python
from src.models import TRIAGEMASTER

model = TRIAGEMASTER(doc2vec_dim=100, epochs=50)
model.fit(texts, numerical, labels, feature_names)

prediction = model.predict(["Douleur thoracique..."], numerical_features)
importance = model.explain_shap(text, numerical, feature_names)
```

### URGENTIAPARSE

**Architecture**: FlauBERT (fine-tuné) + XGBoost

- **Explicabilité**: SHAP TreeExplainer + Attention BERT
- **Latence**: ~380ms
- **Erreur**: ~25%
- **Avantages**: Compréhension contextuelle, attention visualisable

```python
from src.models import URGENTIAPARSE

model = URGENTIAPARSE(fine_tune_epochs=3)
model.fit(texts, numerical, labels, feature_names)

# Visualiser l'attention
attention_words = model.explain_attention("Douleur thoracique...")
```

### EMERGINET

**Architecture**: FlauBERT (frozen) + JEPA encoder + VICReg loss

- **Explicabilité**: Integrated Gradients (Captum)
- **Latence**: ~240ms
- **Erreur**: ~10%
- **Avantages**: Réévaluation continue, représentations robustes

```python
from src.models import EMERGINET

model = EMERGINET(latent_dim=128, vicreg_weight=0.1)
model.fit(texts, numerical, labels, feature_names)

# Réévaluation continue
result = model.reevaluate(
    text="Douleur thoracique...",
    numerical=features,
    initial_prediction=3,
    delta_time=30  # minutes
)
```

---

## Simulation

### Architecture hybride

La simulation combine:
- **SimPy** pour les flux (DES)
- **Mesa** pour les comportements (SMA)

### Scénarios

| Scénario | Agent IA | Charge | Description |
|----------|----------|--------|-------------|
| reference | Aucun | 100% | Triage manuel |
| nlp | TRIAGEMASTER | 100% | NLP classique |
| llm | URGENTIAPARSE | 100% | LLM fine-tuné |
| jepa | EMERGINET | 100% | JEPA + VICReg |
| crise | EMERGINET | 200% | Stress-test |

### Utilisation

```python
from src.simulation import OrchestrateurSimulation

orch = OrchestrateurSimulation(duree_jours=180)
resultats = orch.executer_tous_scenarios()
orch.generer_rapport_comparatif(resultats)
```

---

## Process Mining

### Pipeline PM4Py

```python
from src.process_mining import ProcessMiningPipeline

pipeline = ProcessMiningPipeline('traces.csv')

# Découverte de processus
model = pipeline.discover_process(algorithm='inductive')

# Conformance checking
metrics = pipeline.check_conformance(model)

# KPIs
kpis = pipeline.compute_kpis()
print(f"DMS médiane: {kpis['dms_median']:.1f} min")
```

### Export des distributions

```python
# Exporter pour calibrer la simulation
distributions = pipeline.export_distributions('distributions.json')
```

---

## Support

- **Issues**: https://github.com/edlansiaux/eimlia-teu/issues
- **Email**: edouard1.lansiaux@chu-lille.fr
- **Documentation**: https://eimlia.readthedocs.io

## Licence

MIT License - Voir [LICENSE](../LICENSE)
