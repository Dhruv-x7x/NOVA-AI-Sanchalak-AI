"""
MLflow Configuration for SANCHALAK AI
==========================================
Configuration settings for MLflow experiment tracking.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class MLflowConfig:
    """MLflow configuration settings."""
    
    # Tracking URI (local filesystem by default)
    tracking_uri: str = str(Path("data/mlruns").absolute())
    
    # Artifact storage location
    artifact_root: str = str(Path("data/mlruns/artifacts").absolute())
    
    # Default experiment name
    default_experiment: str = "Sanchalak AI NEXUS"
    
    # Model registry backend (if using remote server)
    registry_uri: Optional[str] = None
    
    # Server settings (for mlflow server mode)
    host: str = "127.0.0.1"
    port: int = 5000
    
    # Auto-logging settings
    autolog_sklearn: bool = True
    autolog_xgboost: bool = True
    autolog_lightgbm: bool = True
    
    # Logging settings
    log_models: bool = True
    log_input_examples: bool = True
    log_model_signatures: bool = True
    
    @classmethod
    def from_env(cls) -> "MLflowConfig":
        """Load configuration from environment variables."""
        return cls(
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI",
                str(Path("data/mlruns").absolute())
            ),
            artifact_root=os.getenv(
                "MLFLOW_ARTIFACT_ROOT",
                str(Path("data/mlruns/artifacts").absolute())
            ),
            default_experiment=os.getenv(
                "MLFLOW_EXPERIMENT_NAME",
                "Sanchalak AI NEXUS"
            ),
            registry_uri=os.getenv("MLFLOW_MODEL_REGISTRY_URI"),
            host=os.getenv("MLFLOW_HOST", "127.0.0.1"),
            port=int(os.getenv("MLFLOW_PORT", "5000"))
        )


# Experiment definitions for Sanchalak AI models
EXPERIMENTS = {
    "risk_classifier": {
        "name": "Sanchalak AI Risk Classifier",
        "description": "Patient risk classification model (XGBoost + LightGBM)",
        "tags": {
            "model_type": "classification",
            "target": "risk_level",
            "domain": "clinical_trials"
        }
    },
    "issue_detector": {
        "name": "Sanchalak AI Issue Detector",
        "description": "Multi-label issue type detection model",
        "tags": {
            "model_type": "multi_label_classification",
            "target": "issue_types",
            "domain": "clinical_trials"
        }
    },
    "site_ranker": {
        "name": "Sanchalak AI Site Ranker",
        "description": "Site performance ranking model",
        "tags": {
            "model_type": "ranking",
            "target": "site_performance",
            "domain": "clinical_trials"
        }
    },
    "anomaly_detector": {
        "name": "Sanchalak AI Anomaly Detector",
        "description": "Unsupervised anomaly detection model",
        "tags": {
            "model_type": "anomaly_detection",
            "target": "anomaly_score",
            "domain": "clinical_trials"
        }
    },
    "resolution_predictor": {
        "name": "Sanchalak AI Resolution Time Predictor",
        "description": "Issue resolution time prediction model",
        "tags": {
            "model_type": "regression",
            "target": "resolution_days",
            "domain": "clinical_trials"
        }
    }
}

# Model registry stages
MODEL_STAGES = ["None", "Staging", "Production", "Archived"]

# Default metrics to track for each model type
DEFAULT_METRICS = {
    "classification": [
        "accuracy", "precision", "recall", "f1_score", 
        "roc_auc", "log_loss", "confusion_matrix"
    ],
    "regression": [
        "mse", "rmse", "mae", "r2_score", "mape"
    ],
    "ranking": [
        "ndcg", "map", "mrr", "precision_at_k"
    ],
    "anomaly_detection": [
        "precision", "recall", "f1_score", "auc_roc"
    ]
}


def get_mlflow_config() -> MLflowConfig:
    """Get MLflow configuration instance."""
    return MLflowConfig.from_env()
