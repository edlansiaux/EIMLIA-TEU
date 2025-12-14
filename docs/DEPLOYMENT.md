# Guide de Déploiement EIMLIA-TEU

## Prérequis

- Python 3.10+
- Docker (optionnel)
- GPU NVIDIA avec CUDA 11.8+ (recommandé)
- 8 GB RAM minimum (16 GB recommandé)

## Installation locale

### 1. Cloner le repository

```bash
git clone https://github.com/votre-org/eimlia-teu.git
cd eimlia-teu
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Vérifier l'installation

```bash
python -c "from src.models import EMERGINET; print('OK')"
pytest tests/ -v --ignore=tests/test_integration.py
```

---

## Déploiement Docker

### Build de l'image

```bash
docker build -f docker/Dockerfile -t eimlia:latest .
```

### Exécution simple

```bash
docker run -p 8000:8000 eimlia:latest
```

### Docker Compose (recommandé)

```bash
cd docker
docker-compose up -d
```

Cela lance:
- API sur le port 8000
- Prometheus sur le port 9090 (profil monitoring)
- Grafana sur le port 3000 (profil monitoring)

Pour inclure le monitoring:
```bash
docker-compose --profile monitoring up -d
```

### Variables d'environnement

Créer un fichier `.env`:

```env
# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO

# Modèles
DEFAULT_MODEL=emerginet
MODELS_PATH=/app/models

# Base de données (optionnel)
# DATABASE_URL=postgresql://user:pass@host:5432/eimlia

# Monitoring
PROMETHEUS_ENABLED=true
```

---

## Déploiement Kubernetes

### 1. Créer le namespace

```bash
kubectl create namespace eimlia
```

### 2. Déployer les secrets

```bash
kubectl create secret generic eimlia-secrets \
  --from-literal=api-key=your-api-key \
  -n eimlia
```

### 3. Déployer l'application

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eimlia-api
  namespace: eimlia
spec:
  replicas: 3
  selector:
    matchLabels:
      app: eimlia-api
  template:
    metadata:
      labels:
        app: eimlia-api
    spec:
      containers:
      - name: api
        image: eimlia:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: LOG_LEVEL
          value: "INFO"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: eimlia-api
  namespace: eimlia
spec:
  selector:
    app: eimlia-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

```bash
kubectl apply -f k8s/deployment.yaml
```

### 4. Autoscaling (optionnel)

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: eimlia-api-hpa
  namespace: eimlia
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: eimlia-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Configuration GPU

### Docker avec GPU

```bash
docker run --gpus all -p 8000:8000 eimlia:latest
```

### Kubernetes avec GPU

```yaml
spec:
  containers:
  - name: api
    resources:
      limits:
        nvidia.com/gpu: 1
```

---

## Monitoring

### Prometheus

L'API expose des métriques sur `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'eimlia'
    static_configs:
      - targets: ['eimlia-api:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Importer le dashboard depuis `docker/grafana-dashboard.json` (à créer).

Métriques disponibles:
- `eimlia_predictions_total` - Total des prédictions
- `eimlia_prediction_latency_seconds` - Latence
- `eimlia_simulation_progress` - Progression simulation
- `eimlia_model_errors_total` - Erreurs

### Alertes

```yaml
# alertmanager/rules.yml
groups:
- name: eimlia
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, eimlia_prediction_latency_seconds) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Latence élevée sur EIMLIA"

  - alert: APIDown
    expr: up{job="eimlia"} == 0
    for: 1m
    labels:
      severity: critical
```

---

## Logging

### Configuration

```python
# src/utils/logging.py
from loguru import logger

logger.add(
    "logs/eimlia.log",
    rotation="100 MB",
    retention="30 days",
    level="INFO"
)
```

### Niveaux de log

| Niveau | Usage |
|--------|-------|
| DEBUG | Développement |
| INFO | Production normale |
| WARNING | Alertes non-critiques |
| ERROR | Erreurs |

### Logs structurés (JSON)

```python
import json_logging
json_logging.init_fastapi(enable_json=True)
```

---

## Sécurité

### HTTPS

Utiliser un reverse proxy (nginx, traefik) pour TLS:

```nginx
server {
    listen 443 ssl;
    server_name eimlia.example.com;

    ssl_certificate /etc/ssl/certs/eimlia.crt;
    ssl_certificate_key /etc/ssl/private/eimlia.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Authentification

Ajouter JWT (optionnel):

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(credentials = Depends(security)):
    # Vérifier le token
    pass
```

### CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://votre-domaine.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization"]
)
```

---

## Sauvegarde

### Modèles

```bash
# Sauvegarder
tar -czvf models-backup.tar.gz models/

# Restaurer
tar -xzvf models-backup.tar.gz
```

### Données

```bash
# Sauvegarder
pg_dump eimlia > backup.sql

# Restaurer
psql eimlia < backup.sql
```

---

## Mise à jour

### Rolling update (Kubernetes)

```bash
kubectl set image deployment/eimlia-api api=eimlia:v2.0.0 -n eimlia
```

### Docker Compose

```bash
docker-compose pull
docker-compose up -d
```

---

## Troubleshooting

### L'API ne démarre pas

1. Vérifier les logs: `docker logs eimlia-api`
2. Vérifier la mémoire: `docker stats`
3. Vérifier les dépendances: `pip check`

### Latence élevée

1. Activer le GPU
2. Augmenter les workers
3. Vérifier la charge CPU/mémoire

### Erreurs de prédiction

1. Vérifier le format des données
2. Vérifier que le modèle est chargé
3. Consulter les logs

### Contact support

- Email: support@eimlia.fr
- Issues: https://github.com/votre-org/eimlia-teu/issues
