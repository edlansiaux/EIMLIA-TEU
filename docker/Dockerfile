# EIMLIA-TEU Dockerfile
# ========================

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="eimlia@chu-lille.fr"
LABEL version="1.0.0"
LABEL description="EIMLIA-3M-TEU - Simulation IA triage urgences"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Installer le package
RUN pip install -e .

# Créer les répertoires nécessaires
RUN mkdir -p /app/data /app/models /app/results /app/logs

# Exposer le port API
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande par défaut
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
