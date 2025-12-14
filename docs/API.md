# Documentation API EIMLIA-3M-TEU

## Base URL

```
http://localhost:8000
```

## Authentification

L'API est actuellement ouverte. Pour la production, ajouter un header d'authentification:

```
Authorization: Bearer <token>
```

## Endpoints

### Health Check

Vérifie l'état de l'API.

**Request**
```http
GET /health
```

**Response**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### Prédiction de triage

Prédit le niveau de triage FRENCH pour un patient.

**Request**
```http
POST /predict
Content-Type: application/json

{
  "text": "Patient de 65 ans, douleur thoracique depuis 2h, EVA 7/10",
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

**Paramètres**

| Champ | Type | Requis | Description |
|-------|------|--------|-------------|
| text | string | ✓ | Verbatim de l'entretien IAO |
| numerical | object | - | Features numériques |
| model | string | - | Modèle: `triagemaster`, `urgentiaparse`, `emerginet` |

**Features numériques supportées**

| Feature | Type | Unité | Range |
|---------|------|-------|-------|
| age | int | années | 0-120 |
| fc | float | bpm | 30-250 |
| pas | float | mmHg | 50-250 |
| pad | float | mmHg | 30-150 |
| spo2 | float | % | 50-100 |
| temperature | float | °C | 34-42 |
| eva | int | - | 0-10 |
| glasgow | int | - | 3-15 |

**Response**
```json
{
  "niveau_french": 2,
  "probabilites": [0.05, 0.65, 0.20, 0.08, 0.02],
  "confiance": 0.65,
  "model_used": "emerginet",
  "timestamp": "2024-01-15T10:30:00Z",
  "alertes": ["Tachycardie: 95 bpm"]
}
```

**Codes d'erreur**

| Code | Description |
|------|-------------|
| 400 | Données invalides |
| 422 | Validation error |
| 500 | Erreur interne |

---

### Explicabilité

Retourne les explications SHAP pour une prédiction.

**Request**
```http
POST /explain
Content-Type: application/json

{
  "text": "Patient de 65 ans, douleur thoracique",
  "numerical": {"age": 65, "fc": 95},
  "model": "emerginet"
}
```

**Response**
```json
{
  "feature_importance": {
    "spo2": 0.25,
    "fc": 0.20,
    "eva": 0.15,
    "pas": 0.12,
    "age": 0.10,
    "text_embedding": 0.18
  },
  "method": "IntegratedGradients",
  "model": "emerginet"
}
```

---

### Liste des modèles

Retourne les modèles disponibles.

**Request**
```http
GET /models
```

**Response**
```json
{
  "models": [
    {
      "name": "triagemaster",
      "architecture": "Doc2Vec + MLP",
      "error_rate": "~39%",
      "latency_ms": 120
    },
    {
      "name": "urgentiaparse",
      "architecture": "FlauBERT + XGBoost",
      "error_rate": "~25%",
      "latency_ms": 380
    },
    {
      "name": "emerginet",
      "architecture": "JEPA + VICReg",
      "error_rate": "~10%",
      "latency_ms": 240,
      "features": ["reevaluation_continue", "integrated_gradients"]
    }
  ]
}
```

---

### Démarrer une simulation

Lance une simulation en arrière-plan.

**Request**
```http
POST /simulation/start
Content-Type: application/json

{
  "scenario": "jepa",
  "duree_jours": 30,
  "facteur_charge": 1.0
}
```

**Paramètres**

| Champ | Type | Default | Description |
|-------|------|---------|-------------|
| scenario | string | "jepa" | Scénario: `reference`, `nlp`, `llm`, `jepa`, `crise` |
| duree_jours | int | 30 | Durée de la simulation (1-365) |
| facteur_charge | float | 1.0 | Facteur de charge (0.1-5.0) |

**Response**
```json
{
  "message": "Simulation démarrée",
  "scenario": "jepa"
}
```

---

### État de la simulation

Retourne l'état de la simulation en cours.

**Request**
```http
GET /simulation/status
```

**Response**
```json
{
  "running": true,
  "progress": 45.5,
  "scenario": "jepa",
  "start_time": "2024-01-15T10:00:00Z",
  "results": null
}
```

Quand terminée:
```json
{
  "running": false,
  "progress": 100,
  "scenario": "jepa",
  "start_time": "2024-01-15T10:00:00Z",
  "results": {
    "dms_median": 185.5,
    "concordance": 0.85,
    "n_patients": 15234
  }
}
```

---

### Métriques Prometheus

Retourne les métriques au format Prometheus.

**Request**
```http
GET /metrics
```

**Response**
```
# HELP eimlia_predictions_total Total predictions
# TYPE eimlia_predictions_total counter
eimlia_predictions_total 1234

# HELP eimlia_simulation_progress Simulation progress
# TYPE eimlia_simulation_progress gauge
eimlia_simulation_progress 45.5

# HELP eimlia_prediction_latency_seconds Prediction latency
# TYPE eimlia_prediction_latency_seconds histogram
eimlia_prediction_latency_seconds_bucket{le="0.1"} 500
eimlia_prediction_latency_seconds_bucket{le="0.5"} 1200
eimlia_prediction_latency_seconds_bucket{le="+Inf"} 1234
```

---

## Exemples d'utilisation

### Python (requests)

```python
import requests

# Prédiction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "Douleur thoracique intense",
        "numerical": {"age": 65, "fc": 110, "spo2": 94},
        "model": "emerginet"
    }
)
prediction = response.json()
print(f"Niveau FRENCH: {prediction['niveau_french']}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Douleur thoracique", "model": "emerginet"}'
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: "Douleur thoracique intense",
    numerical: { age: 65, fc: 110 },
    model: "emerginet"
  })
});

const prediction = await response.json();
console.log(`Niveau FRENCH: ${prediction.niveau_french}`);
```

---

## Codes de statut HTTP

| Code | Description |
|------|-------------|
| 200 | Succès |
| 400 | Requête invalide |
| 404 | Ressource non trouvée |
| 409 | Conflit (ex: simulation déjà en cours) |
| 422 | Erreur de validation |
| 500 | Erreur serveur interne |

---

## Rate Limiting

| Endpoint | Limite |
|----------|--------|
| /predict | 100 req/min |
| /simulation/start | 1 req/min |
| autres | 1000 req/min |

---

## WebSocket (futur)

Endpoint planifié pour le streaming des résultats de simulation:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/simulation');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress}%`);
};
```
