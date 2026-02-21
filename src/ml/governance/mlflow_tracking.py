"""
SANCHALAK AI - MLflow Experiment Tracking
==================================================
Production ML experiment tracking and model registry.

Features:
- Experiment management
- Run logging (parameters, metrics, artifacts)
- Model registry with versioning
- Model stages (Staging, Production)
- Artifact storage
- Integration with existing training pipelines

Author: Sanchalak AI Team
Date: 2026-01-24
"""

import os
import sys
import json
import pickle
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from functools import wraps
import hashlib

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check MLflow availability
MLFLOW_AVAILABLE = False

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
    logger.info("✅ MLflow available")
except ImportError:
    logger.warning("⚠️ mlflow not installed - using mock tracking")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExperimentInfo:
    """Information about an MLflow experiment."""
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunInfo:
    """Information about an MLflow run."""
    run_id: str
    experiment_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    artifact_uri: str = ""
    params: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["start_time"] = self.start_time.isoformat() if self.start_time else None
        d["end_time"] = self.end_time.isoformat() if self.end_time else None
        return d


@dataclass
class ModelVersion:
    """Information about a registered model version."""
    name: str
    version: str
    stage: str
    source: str
    run_id: str
    status: str = "READY"
    description: str = ""
    creation_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["creation_timestamp"] = self.creation_timestamp.isoformat() if self.creation_timestamp else None
        return d


# =============================================================================
# MOCK MLFLOW CLIENT
# =============================================================================

class MockMLflowClient:
    """Mock MLflow client for when MLflow is not available."""
    
    def __init__(self, tracking_uri: str = ""):
        self.tracking_uri = tracking_uri
        self.experiments: Dict[str, Dict] = {}
        self.runs: Dict[str, Dict] = {}
        self.models: Dict[str, List[Dict]] = {}
        self._run_counter = 0
        self._exp_counter = 0
        
        # Create default experiment
        self.experiments["0"] = {
            "experiment_id": "0",
            "name": "Default",
            "artifact_location": "mlruns/0",
            "lifecycle_stage": "active"
        }
        
        logger.info("MockMLflowClient initialized")
    
    def create_experiment(self, name: str, artifact_location: str = None, tags: Dict = None) -> str:
        self._exp_counter += 1
        exp_id = str(self._exp_counter)
        self.experiments[exp_id] = {
            "experiment_id": exp_id,
            "name": name,
            "artifact_location": artifact_location or f"mlruns/{exp_id}",
            "lifecycle_stage": "active",
            "tags": tags or {}
        }
        return exp_id
    
    def get_experiment_by_name(self, name: str) -> Optional[Dict]:
        for exp in self.experiments.values():
            if exp["name"] == name:
                return exp
        return None
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        return self.experiments.get(experiment_id)
    
    def create_run(self, experiment_id: str, tags: Dict = None) -> Dict:
        self._run_counter += 1
        run_id = f"run_{self._run_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        run = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "status": "RUNNING",
            "start_time": datetime.now(),
            "end_time": None,
            "artifact_uri": f"mlruns/{experiment_id}/{run_id}/artifacts",
            "params": {},
            "metrics": {},
            "tags": tags or {}
        }
        self.runs[run_id] = run
        return run
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        return self.runs.get(run_id)
    
    def log_param(self, run_id: str, key: str, value: Any):
        if run_id in self.runs:
            self.runs[run_id]["params"][key] = str(value)
    
    def log_params(self, run_id: str, params: Dict[str, Any]):
        for key, value in params.items():
            self.log_param(run_id, key, value)
    
    def log_metric(self, run_id: str, key: str, value: float, step: int = None):
        if run_id in self.runs:
            self.runs[run_id]["metrics"][key] = value
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float]):
        for key, value in metrics.items():
            self.log_metric(run_id, key, value)
    
    def set_terminated(self, run_id: str, status: str = "FINISHED"):
        if run_id in self.runs:
            self.runs[run_id]["status"] = status
            self.runs[run_id]["end_time"] = datetime.now()
    
    def log_artifact(self, run_id: str, local_path: str, artifact_path: str = None):
        # Mock - just log that artifact was stored
        logger.debug(f"Mock artifact logged: {local_path} -> {artifact_path}")
    
    def create_registered_model(self, name: str, tags: Dict = None, description: str = ""):
        if name not in self.models:
            self.models[name] = []
    
    def create_model_version(
        self, name: str, source: str, run_id: str,
        tags: Dict = None, description: str = ""
    ) -> Dict:
        if name not in self.models:
            self.models[name] = []
        
        version = len(self.models[name]) + 1
        model_version = {
            "name": name,
            "version": str(version),
            "stage": "None",
            "source": source,
            "run_id": run_id,
            "status": "READY",
            "description": description,
            "creation_timestamp": datetime.now()
        }
        self.models[name].append(model_version)
        return model_version
    
    def transition_model_version_stage(
        self, name: str, version: str, stage: str
    ):
        if name in self.models:
            for mv in self.models[name]:
                if mv["version"] == version:
                    mv["stage"] = stage
                    break
    
    def get_latest_versions(self, name: str, stages: List[str] = None) -> List[Dict]:
        if name not in self.models:
            return []
        
        versions = self.models[name]
        if stages:
            versions = [v for v in versions if v["stage"] in stages]
        
        return sorted(versions, key=lambda x: int(x["version"]), reverse=True)
    
    def search_runs(
        self, experiment_ids: List[str], filter_string: str = "",
        max_results: int = 100
    ) -> List[Dict]:
        results = []
        for run in self.runs.values():
            if run["experiment_id"] in experiment_ids:
                results.append(run)
        return results[:max_results]


# =============================================================================
# MLFLOW TRACKING SERVICE
# =============================================================================

class MLflowTracker:
    """
    MLflow experiment tracking service for Sanchalak AI NEXUS.
    
    Provides:
    - Experiment management
    - Run tracking with params/metrics/artifacts
    - Model registry with versioning
    - Model stage management (Staging, Production)
    
    Usage:
        tracker = MLflowTracker()
        
        # Start tracking
        with tracker.start_run("risk_classifier_v1") as run:
            tracker.log_params({"n_estimators": 100, "max_depth": 10})
            tracker.log_metrics({"accuracy": 0.95, "f1": 0.93})
            tracker.log_model(model, "risk_classifier")
    """
    
    # Default experiment names
    EXPERIMENTS = {
        "risk_classifier": "Sanchalak AI Risk Classifier",
        "issue_detector": "Sanchalak AI Issue Detector",
        "site_ranker": "Sanchalak AI Site Ranker",
        "anomaly_detector": "Sanchalak AI Anomaly Detector",
        "resolution_predictor": "Sanchalak AI Resolution Time Predictor"
    }
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI (default: local file store)
            artifact_location: Artifact storage location
            experiment_name: Default experiment name
        """
        # Use file:// URI scheme for local file stores
        default_uri = Path("data/mlruns").absolute().as_uri()
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", 
            default_uri
        )
        self.artifact_location = artifact_location or str(Path("data/mlruns/artifacts").absolute())
        self.experiment_name = experiment_name or "Sanchalak AI NEXUS"
        
        self._client = None
        self._current_run = None
        self._current_run_id = None
        self._use_mock = not MLFLOW_AVAILABLE
        self._initialized = False
        
        # Ensure directories exist (strip file:// prefix if present)
        tracking_path = self.tracking_uri.replace("file:///", "").replace("file://", "")
        Path(tracking_path).mkdir(parents=True, exist_ok=True)
        Path(self.artifact_location).mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> bool:
        """Initialize MLflow connection."""
        if self._initialized:
            return True
        
        try:
            if self._use_mock:
                self._client = MockMLflowClient(self.tracking_uri)
                logger.info("✅ Using mock MLflow client")
            else:
                # Set tracking URI
                mlflow.set_tracking_uri(self.tracking_uri)
                self._client = MlflowClient(self.tracking_uri)
                logger.info(f"✅ MLflow connected to: {self.tracking_uri}")
            
            # Create default experiment if needed
            self._ensure_experiment(self.experiment_name)
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}")
            self._use_mock = True
            self._client = MockMLflowClient(self.tracking_uri)
            self._initialized = True
            return True
    
    @property
    def is_ready(self) -> bool:
        """Check if tracker is ready."""
        if not self._initialized:
            self.initialize()
        return self._initialized
    
    @property
    def uses_mock(self) -> bool:
        """Check if using mock client."""
        return self._use_mock
    
    def _ensure_experiment(self, name: str) -> str:
        """Ensure experiment exists, create if needed."""
        try:
            if self._use_mock:
                exp = self._client.get_experiment_by_name(name)
                if not exp:
                    return self._client.create_experiment(name, self.artifact_location)
                return exp["experiment_id"]
            else:
                exp = mlflow.get_experiment_by_name(name)
                if exp is None:
                    return mlflow.create_experiment(name, self.artifact_location)
                return exp.experiment_id
        except Exception as e:
            logger.warning(f"Error ensuring experiment {name}: {e}")
            return "0"
    
    # =========================================================================
    # RUN MANAGEMENT
    # =========================================================================
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> "MLflowTracker":
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            experiment_name: Experiment to log to
            tags: Run tags
            nested: Whether this is a nested run
            
        Returns:
            Self for context manager usage
        """
        if not self.is_ready:
            return self
        
        exp_name = experiment_name or self.experiment_name
        exp_id = self._ensure_experiment(exp_name)
        
        run_tags = tags or {}
        run_tags["mlflow.runName"] = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if self._use_mock:
                run = self._client.create_run(exp_id, run_tags)
                self._current_run = run
                self._current_run_id = run["run_id"]
            else:
                mlflow.set_experiment(exp_name)
                self._current_run = mlflow.start_run(
                    run_name=run_name,
                    nested=nested,
                    tags=run_tags
                )
                self._current_run_id = self._current_run.info.run_id
            
            logger.info(f"Started run: {self._current_run_id}")
            
        except Exception as e:
            logger.error(f"Error starting run: {e}")
        
        return self
    
    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        if self._current_run_id:
            try:
                if self._use_mock:
                    self._client.set_terminated(self._current_run_id, status)
                else:
                    mlflow.end_run(status)
                
                logger.info(f"Ended run: {self._current_run_id} ({status})")
                
            except Exception as e:
                logger.error(f"Error ending run: {e}")
            finally:
                self._current_run = None
                self._current_run_id = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status)
        return False
    
    # =========================================================================
    # LOGGING METHODS
    # =========================================================================
    
    def log_param(self, key: str, value: Any):
        """Log a single parameter."""
        if not self._current_run_id:
            return
        
        try:
            if self._use_mock:
                self._client.log_param(self._current_run_id, key, value)
            else:
                mlflow.log_param(key, value)
        except Exception as e:
            logger.warning(f"Error logging param {key}: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if not self._current_run_id:
            return
        
        try:
            if self._use_mock:
                self._client.log_metric(self._current_run_id, key, value, step)
            else:
                mlflow.log_metric(key, value, step)
        except Exception as e:
            logger.warning(f"Error logging metric {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file as an artifact."""
        if not self._current_run_id:
            return
        
        try:
            if self._use_mock:
                self._client.log_artifact(self._current_run_id, local_path, artifact_path)
            else:
                mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Error logging artifact: {e}")
    
    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log a dictionary as a JSON artifact."""
        if not self._current_run_id:
            return
        
        try:
            if self._use_mock:
                # Just log that we would save it
                logger.debug(f"Would save dict to: {artifact_file}")
            else:
                mlflow.log_dict(dictionary, artifact_file)
        except Exception as e:
            logger.warning(f"Error logging dict: {e}")
    
    def log_figure(self, figure, artifact_file: str):
        """Log a matplotlib/plotly figure."""
        if not self._current_run_id:
            return
        
        try:
            if self._use_mock:
                logger.debug(f"Would save figure to: {artifact_file}")
            else:
                mlflow.log_figure(figure, artifact_file)
        except Exception as e:
            logger.warning(f"Error logging figure: {e}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_name: Optional[str] = None,
        signature: Any = None,
        input_example: Any = None
    ) -> Optional[str]:
        """
        Log a trained model.
        
        Args:
            model: The trained model
            artifact_path: Path within artifacts to store model
            registered_name: Register model with this name
            signature: Model signature (input/output schema)
            input_example: Example input for signature inference
            
        Returns:
            Model URI if successful
        """
        if not self._current_run_id:
            return None
        
        try:
            if self._use_mock:
                # Save model to temp file
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                    pickle.dump(model, f)
                    model_path = f.name
                
                # Mock model registration
                if registered_name:
                    self._client.create_registered_model(registered_name)
                    version = self._client.create_model_version(
                        registered_name,
                        source=model_path,
                        run_id=self._current_run_id
                    )
                    logger.info(f"Registered model: {registered_name} v{version['version']}")
                
                # Cleanup temp file
                os.unlink(model_path)
                
                return f"runs:/{self._current_run_id}/{artifact_path}"
            else:
                # Try sklearn flavor first
                model_info = mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_name,
                    signature=signature,
                    input_example=input_example
                )
                logger.info(f"Logged model to: {model_info.model_uri}")
                return model_info.model_uri
                
        except Exception as e:
            logger.warning(f"Error logging model: {e}")
            # Try pickle fallback
            try:
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                    pickle.dump(model, f)
                    self.log_artifact(f.name, artifact_path)
                    os.unlink(f.name)
                return f"runs:/{self._current_run_id}/{artifact_path}"
            except Exception as e2:
                logger.error(f"Fallback model logging failed: {e2}")
                return None
    
    def log_sklearn_model(
        self,
        model: Any,
        artifact_path: str,
        X_sample: Optional[pd.DataFrame] = None,
        registered_name: Optional[str] = None
    ) -> Optional[str]:
        """Log a scikit-learn model with signature inference."""
        signature = None
        if X_sample is not None and not self._use_mock:
            try:
                signature = infer_signature(X_sample, model.predict(X_sample))
            except Exception:
                pass
        
        return self.log_model(
            model, artifact_path, registered_name,
            signature=signature,
            input_example=X_sample.head(5) if X_sample is not None else None
        )
    
    # =========================================================================
    # MODEL REGISTRY
    # =========================================================================
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        description: str = ""
    ) -> Optional[ModelVersion]:
        """
        Register a model in the model registry.
        
        Args:
            model_uri: URI of the logged model
            name: Registry name
            description: Model description
            
        Returns:
            ModelVersion info
        """
        if not self.is_ready:
            return None
        
        try:
            if self._use_mock:
                self._client.create_registered_model(name, description=description)
                version = self._client.create_model_version(
                    name, model_uri, self._current_run_id or "",
                    description=description
                )
                return ModelVersion(
                    name=version["name"],
                    version=version["version"],
                    stage=version["stage"],
                    source=version["source"],
                    run_id=version["run_id"],
                    description=description,
                    creation_timestamp=version["creation_timestamp"]
                )
            else:
                # Create registered model if not exists
                try:
                    self._client.create_registered_model(name, description=description)
                except Exception:
                    pass  # Model may already exist
                
                # Create version
                mv = mlflow.register_model(model_uri, name)
                return ModelVersion(
                    name=mv.name,
                    version=mv.version,
                    stage=mv.current_stage,
                    source=mv.source,
                    run_id=mv.run_id,
                    status=mv.status,
                    description=description
                )
                
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ) -> bool:
        """
        Transition a model version to a new stage.
        
        Args:
            name: Registered model name
            version: Version number
            stage: New stage (Staging, Production, Archived)
            archive_existing: Archive current production model
            
        Returns:
            True if successful
        """
        if not self.is_ready:
            return False
        
        valid_stages = ["Staging", "Production", "Archived", "None"]
        if stage not in valid_stages:
            logger.error(f"Invalid stage: {stage}. Must be one of {valid_stages}")
            return False
        
        try:
            if self._use_mock:
                self._client.transition_model_version_stage(name, version, stage)
            else:
                self._client.transition_model_version_stage(
                    name=name,
                    version=version,
                    stage=stage,
                    archive_existing_versions=archive_existing
                )
            
            logger.info(f"Transitioned {name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            return False
    
    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """Get the current production model version."""
        return self.get_model_version(name, stage="Production")
    
    def get_model_version(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """Get a specific model version or latest in stage."""
        if not self.is_ready:
            return None
        
        try:
            if self._use_mock:
                stages = [stage] if stage else None
                versions = self._client.get_latest_versions(name, stages)
                if versions:
                    v = versions[0]
                    return ModelVersion(
                        name=v["name"],
                        version=v["version"],
                        stage=v["stage"],
                        source=v["source"],
                        run_id=v["run_id"]
                    )
            else:
                if version:
                    mv = self._client.get_model_version(name, version)
                else:
                    stages = [stage] if stage else None
                    versions = self._client.get_latest_versions(name, stages)
                    mv = versions[0] if versions else None
                
                if mv:
                    return ModelVersion(
                        name=mv.name,
                        version=mv.version,
                        stage=mv.current_stage,
                        source=mv.source,
                        run_id=mv.run_id,
                        status=mv.status
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting model version: {e}")
            return None
    
    def load_model(self, name: str, stage: str = "Production") -> Any:
        """Load a model from the registry."""
        if not self.is_ready:
            return None
        
        try:
            if self._use_mock:
                logger.warning("Cannot load models in mock mode")
                return None
            else:
                model_uri = f"models:/{name}/{stage}"
                return mlflow.sklearn.load_model(model_uri)
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    # =========================================================================
    # SEARCH & QUERY
    # =========================================================================
    
    def search_runs(
        self,
        experiment_names: Optional[List[str]] = None,
        filter_string: str = "",
        max_results: int = 100
    ) -> List[RunInfo]:
        """Search for runs across experiments."""
        if not self.is_ready:
            return []
        
        try:
            exp_names = experiment_names or [self.experiment_name]
            exp_ids = []
            
            for name in exp_names:
                if self._use_mock:
                    exp = self._client.get_experiment_by_name(name)
                    if exp:
                        exp_ids.append(exp["experiment_id"])
                else:
                    exp = mlflow.get_experiment_by_name(name)
                    if exp:
                        exp_ids.append(exp.experiment_id)
            
            if not exp_ids:
                return []
            
            if self._use_mock:
                runs = self._client.search_runs(exp_ids, filter_string, max_results)
                return [
                    RunInfo(
                        run_id=r["run_id"],
                        experiment_id=r["experiment_id"],
                        status=r["status"],
                        start_time=r["start_time"],
                        end_time=r["end_time"],
                        params=r["params"],
                        metrics=r["metrics"],
                        tags=r["tags"]
                    )
                    for r in runs
                ]
            else:
                runs = mlflow.search_runs(
                    experiment_ids=exp_ids,
                    filter_string=filter_string,
                    max_results=max_results,
                    output_format="list"
                )
                return [
                    RunInfo(
                        run_id=r.info.run_id,
                        experiment_id=r.info.experiment_id,
                        status=r.info.status,
                        start_time=datetime.fromtimestamp(r.info.start_time / 1000),
                        end_time=datetime.fromtimestamp(r.info.end_time / 1000) if r.info.end_time else None,
                        artifact_uri=r.info.artifact_uri,
                        params=dict(r.data.params),
                        metrics=dict(r.data.metrics),
                        tags=dict(r.data.tags)
                    )
                    for r in runs
                ]
                
        except Exception as e:
            logger.error(f"Error searching runs: {e}")
            return []
    
    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        ascending: bool = False
    ) -> Optional[RunInfo]:
        """Get the best run by a specific metric."""
        runs = self.search_runs([experiment_name])
        
        if not runs:
            return None
        
        # Filter runs with the metric
        runs_with_metric = [r for r in runs if metric in r.metrics]
        
        if not runs_with_metric:
            return None
        
        # Sort by metric
        runs_with_metric.sort(
            key=lambda r: r.metrics[metric],
            reverse=not ascending
        )
        
        return runs_with_metric[0]
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        stats = {
            "initialized": self._initialized,
            "using_mock": self._use_mock,
            "tracking_uri": self.tracking_uri,
            "current_run": self._current_run_id
        }
        
        if self._use_mock:
            stats["experiments"] = len(self._client.experiments)
            stats["runs"] = len(self._client.runs)
            stats["models"] = len(self._client.models)
        
        return stats


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_mlflow_tracker: Optional[MLflowTracker] = None


def get_mlflow_tracker() -> MLflowTracker:
    """Get singleton MLflow tracker instance."""
    global _mlflow_tracker
    if _mlflow_tracker is None:
        _mlflow_tracker = MLflowTracker()
        _mlflow_tracker.initialize()
    return _mlflow_tracker


def reset_mlflow_tracker():
    """Reset the singleton (for testing)."""
    global _mlflow_tracker
    if _mlflow_tracker and _mlflow_tracker._current_run_id:
        _mlflow_tracker.end_run()
    _mlflow_tracker = None


# =============================================================================
# DECORATOR FOR AUTOMATIC TRACKING
# =============================================================================

def track_training(
    experiment_name: str = None,
    log_params: List[str] = None,
    log_metrics: List[str] = None
):
    """
    Decorator to automatically track model training.
    
    Usage:
        @track_training(experiment_name="risk_classifier")
        def train_model(n_estimators=100, max_depth=10):
            ...
            return {"accuracy": 0.95, "model": trained_model}
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_mlflow_tracker()
            
            exp_name = experiment_name or func.__name__
            
            with tracker.start_run(run_name=func.__name__, experiment_name=exp_name):
                # Log parameters from kwargs
                if log_params:
                    params_to_log = {k: v for k, v in kwargs.items() if k in log_params}
                    tracker.log_params(params_to_log)
                else:
                    tracker.log_params(kwargs)
                
                # Run function
                result = func(*args, **kwargs)
                
                # Log metrics from result if dict
                if isinstance(result, dict):
                    if log_metrics:
                        metrics_to_log = {k: v for k, v in result.items() 
                                         if k in log_metrics and isinstance(v, (int, float))}
                    else:
                        metrics_to_log = {k: v for k, v in result.items() 
                                         if isinstance(v, (int, float))}
                    tracker.log_metrics(metrics_to_log)
                
                return result
        
        return wrapper
    return decorator


# =============================================================================
# MAIN / DEMO
# =============================================================================

def main():
    """Demo the MLflow tracker."""
    print("=" * 70)
    print("SANCHALAK AI - MLFLOW EXPERIMENT TRACKING DEMO")
    print("=" * 70)
    
    # Initialize tracker
    tracker = MLflowTracker()
    tracker.initialize()
    
    print(f"\n✅ Tracker initialized (mock={tracker.uses_mock})")
    print(f"   Tracking URI: {tracker.tracking_uri}")
    
    # Demo: Training run
    print("\n" + "-" * 50)
    print("DEMO: Training Run")
    print("-" * 50)
    
    with tracker.start_run("demo_training", "Sanchalak AI Demo"):
        # Log parameters
        tracker.log_params({
            "model_type": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        })
        print("✅ Logged parameters")
        
        # Log metrics
        tracker.log_metrics({
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.91,
            "f1_score": 0.92,
            "roc_auc": 0.97
        })
        print("✅ Logged metrics")
        
        # Log a simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Quick fit on dummy data
        import numpy as np
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        
        model_uri = tracker.log_sklearn_model(
            model, 
            "model",
            pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)]),
            registered_name="demo_risk_classifier"
        )
        print(f"✅ Logged model: {model_uri}")
    
    print("\n✅ Run completed!")
    
    # Demo: Model registry
    print("\n" + "-" * 50)
    print("DEMO: Model Registry")
    print("-" * 50)
    
    # Transition to production
    tracker.transition_model_stage("demo_risk_classifier", "1", "Production")
    print("✅ Model transitioned to Production")
    
    # Get production model
    prod_model = tracker.get_production_model("demo_risk_classifier")
    if prod_model:
        print(f"   Production model: v{prod_model.version}")
    
    # Search runs
    print("\n" + "-" * 50)
    print("DEMO: Search Runs")
    print("-" * 50)
    
    runs = tracker.search_runs(["Sanchalak AI Demo"])
    print(f"Found {len(runs)} runs")
    for run in runs[:3]:
        print(f"  - {run.run_id}: accuracy={run.metrics.get('accuracy', 'N/A')}")
    
    # Print stats
    print("\n" + "=" * 70)
    print("TRACKER STATISTICS")
    print("=" * 70)
    stats = tracker.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ MLflow tracking demo complete!")
    return tracker


if __name__ == "__main__":
    main()
