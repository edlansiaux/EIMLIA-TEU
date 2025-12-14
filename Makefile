# EIMLIA-3M-TEU Makefile
# ======================

.PHONY: help install install-dev test lint format clean docker-build docker-run api train simulate

PYTHON := python
PIP := pip
PYTEST := pytest

# Couleurs
BLUE := \033[34m
GREEN := \033[32m
RESET := \033[0m

help: ## Affiche l'aide
	@echo "$(BLUE)EIMLIA-3M-TEU - Commandes disponibles:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}'

# Installation
install: ## Installe les dépendances
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Installe les dépendances de développement
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install

# Tests
test: ## Lance les tests
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

test-fast: ## Lance les tests rapides uniquement
	$(PYTEST) tests/ -v -m "not slow"

# Qualité du code
lint: ## Vérifie le code
	flake8 src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

format: ## Formate le code
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

check: lint test ## Vérifie tout (lint + tests)

# Nettoyage
clean: ## Nettoie les fichiers temporaires
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker
docker-build: ## Construit l'image Docker
	docker build -t eimlia -f docker/Dockerfile .

docker-run: ## Lance le conteneur Docker
	docker-compose -f docker/docker-compose.yml up -d

docker-stop: ## Arrête les conteneurs
	docker-compose -f docker/docker-compose.yml down

docker-logs: ## Affiche les logs
	docker-compose -f docker/docker-compose.yml logs -f

# API
api: ## Lance l'API
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api-prod: ## Lance l'API en production
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Modèles
train: ## Entraîne tous les modèles
	$(PYTHON) scripts/train_models.py --model all

train-triagemaster: ## Entraîne TRIAGEMASTER
	$(PYTHON) scripts/train_models.py --model triagemaster

train-urgentiaparse: ## Entraîne URGENTIAPARSE
	$(PYTHON) scripts/train_models.py --model urgentiaparse

train-emerginet: ## Entraîne EMERGINET
	$(PYTHON) scripts/train_models.py --model emerginet

# Simulation
simulate: ## Lance la simulation complète
	$(PYTHON) scripts/run_simulation.py --scenario all --days 30

simulate-quick: ## Simulation rapide (7 jours)
	$(PYTHON) scripts/run_simulation.py --scenario all --days 7

stress-test: ## Lance les stress-tests
	$(PYTHON) scripts/run_simulation.py --stress-test

# Documentation
docs: ## Génère la documentation
	cd docs && sphinx-build -b html . _build/html

docs-serve: ## Sert la documentation localement
	cd docs/_build/html && $(PYTHON) -m http.server 8080
