# Données EIMLIA-3M-TEU

Ce répertoire contient les données pour l'étude EIMLIA-3M-TEU.

## Structure attendue

```
data/
├── raw/                      # Données brutes
│   └── data3.xlsx           # Fichier source Excel
├── processed/               # Données pré-traitées
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── synthetic/               # Données synthétiques générées
│   └── synthetic_1000.csv
└── event_logs/              # Logs pour Process Mining
    └── traces_urgences.xes
```

## Format des données

### Fichier principal (data3.xlsx)

| Colonne | Type | Description |
|---------|------|-------------|
| Entretien | str | Verbatim de l'entretien IAO |
| Age | int | Âge du patient |
| Sexe | str | M/F |
| FC | float | Fréquence cardiaque (bpm) |
| PAS | float | Pression artérielle systolique (mmHg) |
| PAD | float | Pression artérielle diastolique (mmHg) |
| SpO2 | float | Saturation en oxygène (%) |
| Temperature | float | Température (°C) |
| EVA | int | Échelle visuelle analogique douleur (0-10) |
| Glasgow | int | Score de Glasgow (3-15) |
| CCMU | int | Classification CCMU (1-5) |
| FRENCH inf | int | Niveau FRENCH estimé par l'infirmier |

### Niveaux FRENCH

| Niveau | Description | Couleur |
|--------|-------------|---------|
| 1 | Détresse vitale | Rouge |
| 2 | Atteinte instable | Orange |
| 3 | Atteinte stable urgente | Jaune |
| 4 | Atteinte stable non urgente | Vert |
| 5 | Consultation non urgente | Bleu |

## Génération de données synthétiques

Si vous n'avez pas accès aux données réelles, vous pouvez générer des données synthétiques :

```python
from src.models.data_loader import generate_synthetic_data

# Générer 1000 patients synthétiques
texts, numerical, labels, feature_names = generate_synthetic_data(1000)
```

Ou via le script CLI :

```bash
python scripts/generate_synthetic_data.py --n-samples 10000 --output data/synthetic/
```

## Confidentialité

⚠️ **IMPORTANT** : Les données réelles de patients sont confidentielles et ne doivent pas être partagées. Les données uploadées dans ce repository sont exclusivement synthétiques.

Les données réelles doivent être :
- Anonymisées conformément au RGPD
- Stockées sur un serveur sécurisé
- Accessibles uniquement aux membres autorisés de l'étude

## Préparation des données

```python
from src.models.data_loader import load_data, split_data

# Charger les données
texts, numerical, labels, feature_names = load_data('data/raw/data3.xlsx')

# Split train/val/test
train_data, val_data, test_data = split_data(
    texts, numerical, labels,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True
)
```

## Event Logs pour Process Mining

Les event logs pour PM4Py doivent être au format XES ou CSV avec les colonnes :
- `case:concept:name` : ID unique du patient
- `concept:name` : Nom de l'activité
- `time:timestamp` : Horodatage
- `org:resource` : Ressource (optionnel)
