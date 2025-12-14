"""
Tests unitaires EIMLIA
======================
"""

import pytest
import numpy as np
from typing import List


# Fixtures
@pytest.fixture
def sample_texts() -> List[str]:
    """Textes d'exemple pour les tests."""
    return [
        "Patient de 65 ans, douleur thoracique depuis 2h, EVA 7/10",
        "Femme 45 ans, dyspnée progressive, SpO2 94%",
        "Homme 30 ans, traumatisme cheville suite chute"
    ]


@pytest.fixture
def sample_numerical() -> np.ndarray:
    """Features numériques d'exemple."""
    return np.array([
        [65, 95, 140, 85, 96, 37.2, 7, 0.68, 0, 1],
        [45, 88, 130, 80, 94, 37.5, 3, 0.68, 0, 0],
        [30, 75, 120, 75, 98, 36.8, 5, 0.63, 0, 1]
    ])


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Labels d'exemple."""
    return np.array([1, 2, 3])


@pytest.fixture
def feature_names() -> List[str]:
    """Noms des features."""
    return ['Age', 'FC', 'PAS', 'PAD', 'SpO2', 'Temperature', 
            'EVA', 'ShockIndex', 'O2', 'Sexe_num']


# Tests des modèles
class TestModels:
    """Tests pour les modèles d'IA."""
    
    def test_triagemaster_init(self):
        """Test initialisation TRIAGEMASTER."""
        from src.models import TRIAGEMASTER
        
        model = TRIAGEMASTER(doc2vec_dim=50, epochs=5)
        assert model.doc2vec_dim == 50
        assert model.epochs == 5
        assert not model.is_fitted
    
    def test_urgentiaparse_init(self):
        """Test initialisation URGENTIAPARSE."""
        from src.models import URGENTIAPARSE
        
        model = URGENTIAPARSE(fine_tune_epochs=1)
        assert model.fine_tune_epochs == 1
        assert not model.is_fitted
    
    def test_emerginet_init(self):
        """Test initialisation EMERGINET."""
        from src.models import EMERGINET
        
        model = EMERGINET(latent_dim=64, epochs=5)
        assert model.latent_dim == 64
        assert model.epochs == 5
        assert not model.is_fitted
    
    def test_data_loader(self, sample_texts, sample_numerical, sample_labels):
        """Test du data loader."""
        from src.models.data_loader import generate_synthetic_data
        
        texts, numerical, labels, features = generate_synthetic_data(100)
        
        assert len(texts) == 100
        assert numerical.shape[0] == 100
        assert len(labels) == 100
        assert len(features) == 10
    
    def test_evaluation_metrics(self, sample_labels):
        """Test des métriques d'évaluation."""
        from src.models.evaluation import evaluate_model
        
        y_pred = sample_labels.copy()
        y_pred[0] = 2  # Une erreur
        
        results = evaluate_model(sample_labels, y_pred, verbose=False)
        
        assert 'MAE' in results
        assert 'Kappa' in results
        assert results['Exact'] < 1.0
        assert results['Near'] == 1.0


# Tests de simulation
class TestSimulation:
    """Tests pour les simulations."""
    
    def test_config_default(self):
        """Test configuration par défaut."""
        from src.simulation.config import SimulationConfig
        
        config = SimulationConfig()
        
        assert config.duree_jours == 180
        assert config.nb_box_triage == 3
        assert config.agent_ia is None
    
    def test_patient_generator(self):
        """Test générateur de patients."""
        from src.simulation.patient_generator import GenerateurPatientsVirtuels
        
        gen = GenerateurPatientsVirtuels()
        patient = gen.generer_patient()
        
        assert patient.id is not None
        assert 18 <= patient.age <= 100
        assert patient.sexe in ['M', 'F']
        assert 1 <= patient.niveau_french_reel <= 5
    
    def test_patient_cohorte(self):
        """Test génération de cohorte."""
        from src.simulation.patient_generator import GenerateurPatientsVirtuels
        
        gen = GenerateurPatientsVirtuels()
        patients = gen.generer_cohorte(100, verbose=False)
        
        assert len(patients) == 100
    
    def test_simpy_simulation_short(self):
        """Test simulation SimPy courte."""
        from src.simulation.config import SimulationConfig
        from src.simulation.simpy_des import SimulationUrgencesSimPy
        
        config = SimulationConfig(duree_jours=1)
        sim = SimulationUrgencesSimPy(config, verbose=False)
        results = sim.run()
        
        assert 'n_patients' in results
        assert results['n_patients'] > 0
    
    def test_mesa_model(self):
        """Test modèle Mesa."""
        from src.simulation.config import SimulationConfig
        from src.simulation.mesa_sma import ModeleSAU
        from src.simulation.patient_generator import GenerateurPatientsVirtuels
        
        config = SimulationConfig()
        model = ModeleSAU(config, verbose=False)
        
        # Ajouter quelques patients
        gen = GenerateurPatientsVirtuels()
        for _ in range(10):
            model.ajouter_patient(gen.generer_patient())
        
        # Quelques steps
        for _ in range(100):
            model.step()
        
        results = model.get_resultats()
        assert 'n_patients_total' in results


# Tests Process Mining
class TestProcessMining:
    """Tests pour le process mining."""
    
    def test_pipeline_init(self):
        """Test initialisation pipeline."""
        from src.process_mining import ProcessMiningPipeline
        
        pipeline = ProcessMiningPipeline()
        assert pipeline.event_log is None
    
    def test_kpis_empty(self):
        """Test KPIs sur log vide."""
        from src.process_mining import ProcessMiningPipeline
        import pandas as pd
        
        pipeline = ProcessMiningPipeline()
        pipeline.df_log = pd.DataFrame()
        
        kpis = pipeline.compute_kpis()
        assert kpis == {}


# Tests API
class TestAPI:
    """Tests pour l'API."""
    
    def test_app_creation(self):
        """Test création de l'app."""
        from src.api import create_app
        
        app = create_app()
        assert app is not None
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test endpoint health."""
        from fastapi.testclient import TestClient
        from src.api import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_predict_endpoint(self):
        """Test endpoint predict."""
        from fastapi.testclient import TestClient
        from src.api import app
        
        client = TestClient(app)
        response = client.post("/predict", json={
            "text": "Patient 65 ans, douleur thoracique",
            "numerical": {"age": 65, "fc": 95},
            "model": "emerginet"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "niveau_french" in data
        assert 1 <= data["niveau_french"] <= 5


# Tests utils
class TestUtils:
    """Tests pour les utilitaires."""
    
    def test_compute_metrics(self):
        """Test calcul métriques."""
        from src.utils.metrics import compute_metrics
        
        y_true = np.array([1, 2, 3, 2, 1])
        y_pred = np.array([1, 2, 2, 2, 1])
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'f1_micro' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_format_metrics(self):
        """Test formatage métriques."""
        from src.utils.metrics import format_metrics
        
        metrics = {'accuracy': 0.85, 'f1': 0.82}
        formatted = format_metrics(metrics)
        
        assert 'accuracy' in formatted
        assert '0.85' in formatted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
