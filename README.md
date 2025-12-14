# EIMLIA-TEU 🏥

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SimPy](https://img.shields.io/badge/SimPy-4.0+-green.svg)](https://simpy.readthedocs.io/)
[![Mesa](https://img.shields.io/badge/Mesa-2.0+-orange.svg)](https://mesa.readthedocs.io/)

> **Étude des Impacts Médico-Financiers, Logistiques des systèmes d’Intelligence Artificielle pour le Triage à l’Entrée des Urgences**

Simulation prospective comparant 3 modèles d'IA pour le triage aux urgences (FRENCH) sur 600 000 patients virtuels. Stack 100% open source Python.

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Installation](#-installation)
- [Utilisation rapide](#-utilisation-rapide)
- [Les 3 modèles d'IA](#-les-3-modèles-dia)
- [Simulation](#-simulation)
- [Process Mining](#-process-mining)
- [API](#-api)
- [Tests](#-tests)
- [Licence](#-licence)

## 🎯 Vue d'ensemble

### Objectif

Comparer l'impact de 3 architectures d'IA de triage sur :
- La **qualité du triage** (concordance, sous/sur-triage)
- Les **flux patients** (DMS, temps d'attente)
- L'**acceptabilité soignante** (taux d'adhésion IAO)
- La **résilience** face aux crises (surcharge, pannes)

### Les 3 modèles comparés

| Modèle | Architecture | Explicabilité | Taux d'erreur simulé |
|--------|-------------|---------------|---------------------|
| **TRIAGEMASTER** | Doc2Vec + MLP | SHAP Kernel | ~39% |
| **URGENTIAPARSE** | FlauBERT + XGBoost | SHAP Tree + Attention | ~25% |
| **EMERGINET** | JEPA + VICReg | Integrated Gradients | ~10% |

### Stack technique (100% Open Source)

| Composant | Outil | Licence | Remplace |
|-----------|-------|---------|----------|
| Process Mining | PM4Py | AGPL-3.0 | Celonis (~150-300K€/an) |
| Simulation DES | SimPy | MIT | Arena (~30-50K€/an) |
| Simulation SMA | Mesa | Apache 2.0 | AnyLogic (~40-80K€/an) |
| Deep Learning | PyTorch | BSD-3 | - |
| NLP | FlauBERT/Transformers | Apache 2.0 | - |
| API | FastAPI | MIT | - |

**💰 Économie totale : ~220-430K€/an de licences**

## 🚀 Installation

### Prérequis

- Python 3.10+
- CUDA 11.8+ (optionnel, pour GPU)
- 16 GB RAM minimum (32 GB recommandé)

### Installation rapide

```bash
# Cloner le repository
git clone https://github.com/votre-org/eimlia-teu.git
cd eimlia-teu

# Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
python -c "from src import __version__; print(f'EIMLIA v{__version__}')"
```

### Installation avec Docker

```bash
docker-compose up -d
```

## ⚡ Utilisation rapide

### 1. Entraîner les modèles

```bash
python scripts/train_models.py --data data/data3.xlsx --output models/
```

### 2. Lancer la simulation complète

```bash
python scripts/run_simulation.py --scenario all --patients 100000 --days 180
```

### 3. En Python

```python
from src.models import EMERGINET
from src.simulation import OrchestrateurSimulation

# Entraîner le modèle JEPA
model = EMERGINET(epochs=50)
model.fit(texts_train, numerical_train, labels_train, feature_names)

# Lancer la simulation
orchestrateur = OrchestrateurSimulation(n_patients=100_000, duree_jours=180)
resultats = orchestrateur.executer_tous_scenarios()
```

## 📁 Structure du projet

```
eimlia-teu/
├── src/
│   ├── models/           # 3 modèles IA (TRIAGEMASTER, URGENTIAPARSE, EMERGINET)
│   ├── simulation/       # SimPy (DES) + Mesa (SMA)
│   ├── process_mining/   # PM4Py pipeline
│   ├── utils/            # Utilitaires
│   └── api/              # FastAPI
├── tests/                # Tests unitaires
├── config/               # Configuration YAML
├── scripts/              # Scripts CLI
├── docker/               # Docker/Kubernetes
└── docs/                 # Documentation
```

## 🤖 Les 3 modèles d'IA

Voir [`docs/models.md`](docs/models.md) pour la documentation complète.

```python
from src.models import TRIAGEMASTER, URGENTIAPARSE, EMERGINET

# TRIAGEMASTER - NLP classique
model_nlp = TRIAGEMASTER(doc2vec_dim=100, epochs=100)

# URGENTIAPARSE - LLM + Gradient Boosting  
model_llm = URGENTIAPARSE(bert_model='flaubert/flaubert_base_cased')

# EMERGINET - JEPA + VICReg (le plus performant)
model_jepa = EMERGINET(jepa_dim=256, vicreg_weight=0.1)
```

## 🔬 Simulation

Voir [`docs/simulation.md`](docs/simulation.md) pour la documentation complète.

### Scénarios

| # | Scénario | IA | Patients | Charge |
|---|----------|-----|----------|--------|
| 1 | Référence | Manuel | 100K | 100% |
| 2a | NLP | TRIAGEMASTER | 100K | 100% |
| 2b | LLM | URGENTIAPARSE | 100K | 100% |
| 2c | JEPA | EMERGINET | 100K | 100% |
| 3 | Crise | LLM+JEPA | 200K | 200% |

## 📊 Process Mining

```python
from src.process_mining import ProcessMiningPipeline

pipeline = ProcessMiningPipeline('data/event_log.csv')
kpis = pipeline.compute_kpis()
```

## 🌐 API

```bash
uvicorn src.api.main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient 65 ans, douleur thoracique", "model": "emerginet"}'
```

## 🧪 Tests

```bash
pytest tests/ -v --cov=src
```

## 📄 Licence

MIT License - voir [LICENSE](LICENSE)

## 📚 Références

- Lansiaux et al. (2024). "AI Models for Emergency Triage Prediction"
- Berti et al. (2023). "PM4Py: Process Mining for Python"

---

<p align="center">
  <b>Made with ❤️ at CHU de Lille</b><br>
  <i>Stack 100% Open Source</i>
</p>
