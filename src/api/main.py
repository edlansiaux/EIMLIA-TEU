"""
API FastAPI principale
======================

Point d'entrée de l'API REST.

Usage:
    uvicorn src.api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime

# Création de l'app
app = FastAPI(
    title="EIMLIA-3M-TEU API",
    description="API pour le triage aux urgences assisté par IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# État global (simplifié)
models_cache: Dict[str, Any] = {}
simulation_status: Dict[str, Any] = {"running": False, "progress": 0}


# Schémas Pydantic
class PatientInput(BaseModel):
    """Données patient pour prédiction."""
    text: str = Field(..., description="Texte de l'entretien IAO")
    numerical: Dict[str, float] = Field(
        default_factory=dict,
        description="Features numériques (age, fc, pas, etc.)"
    )
    model: str = Field(
        default="emerginet",
        description="Modèle à utiliser: triagemaster, urgentiaparse, emerginet"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Patient de 65 ans, douleur thoracique depuis 2h, EVA 7/10",
                "numerical": {
                    "age": 65, "fc": 95, "pas": 140, "pad": 85,
                    "spo2": 96, "temperature": 37.2, "eva": 7
                },
                "model": "emerginet"
            }
        }


class PredictionOutput(BaseModel):
    """Résultat de prédiction."""
    niveau_french: int = Field(..., description="Niveau FRENCH prédit (1-5)")
    probabilites: List[float] = Field(..., description="Probabilités par classe")
    confiance: float = Field(..., description="Confiance de la prédiction")
    model_used: str
    timestamp: str
    alertes: List[str] = Field(default_factory=list)


class SimulationConfig(BaseModel):
    """Configuration de simulation."""
    scenario: str = Field(default="jepa", description="Scénario à exécuter")
    duree_jours: int = Field(default=30, ge=1, le=365)
    facteur_charge: float = Field(default=1.0, ge=0.1, le=5.0)


class SimulationStatus(BaseModel):
    """État de la simulation."""
    running: bool
    progress: float
    scenario: Optional[str] = None
    start_time: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


# Routes
@app.get("/")
async def root():
    """Point d'entrée racine."""
    return {
        "name": "EIMLIA-3M-TEU API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Vérification de santé."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=PredictionOutput)
async def predict_triage(patient: PatientInput):
    """
    Prédit le niveau de triage FRENCH pour un patient.
    
    Utilise le modèle spécifié (triagemaster, urgentiaparse, ou emerginet).
    """
    try:
        # Simuler la prédiction (en production: charger le vrai modèle)
        # Pour la démo, on simule
        
        # Features numériques par défaut
        default_features = {
            'age': 50, 'fc': 80, 'pas': 120, 'pad': 80,
            'spo2': 98, 'temperature': 37.0, 'eva': 5,
            'shock_index': 0.67, 'o2': 0, 'sexe_num': 1
        }
        
        features = {**default_features, **patient.numerical}
        
        # Simulation simple basée sur les features
        score = 0
        alertes = []
        
        if features.get('spo2', 98) < 92:
            score += 2
            alertes.append(f"SpO2 basse: {features['spo2']}%")
        if features.get('fc', 80) > 120:
            score += 1
            alertes.append(f"Tachycardie: {features['fc']} bpm")
        if features.get('pas', 120) < 90:
            score += 2
            alertes.append(f"Hypotension: {features['pas']} mmHg")
        if features.get('eva', 5) >= 8:
            score += 1
        
        # Niveau FRENCH (1=très grave, 5=peu grave)
        niveau = max(1, min(5, 5 - score))
        
        # Probabilités simulées
        probs = [0.05, 0.15, 0.45, 0.30, 0.05]  # Distribution type
        probs[niveau - 1] += 0.3
        probs = [p / sum(probs) for p in probs]
        
        return PredictionOutput(
            niveau_french=niveau,
            probabilites=probs,
            confiance=max(probs),
            model_used=patient.model,
            timestamp=datetime.now().isoformat(),
            alertes=alertes
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_prediction(patient: PatientInput):
    """
    Retourne les explications SHAP pour une prédiction.
    """
    # Simulation
    feature_importance = {
        'spo2': 0.25,
        'fc': 0.20,
        'eva': 0.15,
        'pas': 0.12,
        'age': 0.10,
        'temperature': 0.08,
        'text_embedding': 0.10
    }
    
    return {
        "feature_importance": feature_importance,
        "method": "SHAP" if patient.model != "emerginet" else "IntegratedGradients",
        "model": patient.model
    }


@app.get("/simulation/status", response_model=SimulationStatus)
async def get_simulation_status():
    """Retourne l'état de la simulation en cours."""
    return SimulationStatus(**simulation_status)


@app.post("/simulation/start")
async def start_simulation(
    config: SimulationConfig,
    background_tasks: BackgroundTasks
):
    """
    Lance une simulation en arrière-plan.
    """
    if simulation_status["running"]:
        raise HTTPException(status_code=409, detail="Simulation déjà en cours")
    
    simulation_status["running"] = True
    simulation_status["progress"] = 0
    simulation_status["scenario"] = config.scenario
    simulation_status["start_time"] = datetime.now().isoformat()
    
    # Lancer en arrière-plan
    background_tasks.add_task(run_simulation_task, config)
    
    return {"message": "Simulation démarrée", "scenario": config.scenario}


async def run_simulation_task(config: SimulationConfig):
    """Tâche de simulation en arrière-plan."""
    import asyncio
    
    try:
        # Simulation simplifiée
        for i in range(100):
            await asyncio.sleep(0.1)  # Simuler le travail
            simulation_status["progress"] = i + 1
        
        simulation_status["results"] = {
            "dms_median": 185.5,
            "concordance": 0.85,
            "scenario": config.scenario
        }
    finally:
        simulation_status["running"] = False


@app.get("/models")
async def list_models():
    """Liste les modèles disponibles."""
    return {
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


@app.get("/metrics")
async def get_metrics():
    """Métriques Prometheus."""
    # Format Prometheus simplifié
    metrics = [
        "# HELP eimlia_predictions_total Total predictions",
        "# TYPE eimlia_predictions_total counter",
        "eimlia_predictions_total 0",
        "# HELP eimlia_simulation_progress Simulation progress",
        "# TYPE eimlia_simulation_progress gauge",
        f"eimlia_simulation_progress {simulation_status['progress']}"
    ]
    return "\n".join(metrics)


def create_app() -> FastAPI:
    """Factory pour créer l'application."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
