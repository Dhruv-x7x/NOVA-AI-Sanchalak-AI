"""
ML Model Loader for a6on-i
Loads trained models for real-time predictions.

Models:
1. Risk Classifier - 4-class patient risk prediction
2. Issue Detector - 14-type issue detection
3. Site Risk Ranker - Site-level risk scoring
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RISK_MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "ml" / "models"
ISSUE_MODEL_DIR = PROJECT_ROOT / "models" / "issue_detector"
SITE_RANKER_DIR = PROJECT_ROOT / "data" / "processed" / "ml" / "site_ranker"
RESOLUTION_MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "ml" / "models" / "resolution_time"
from src.ml.resolution_time_predictor import ResolutionTimePredictor, ResolutionPredictorConfig
ANOMALY_MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "ml" / "models" / "anomaly_detector"


@dataclass
class RiskPrediction:
    """Risk prediction result"""
    risk_level: str
    probabilities: Dict[str, float]
    confidence: float
    top_features: List[Tuple[str, float]]


@dataclass
class IssuePrediction:
    """Issue detection result"""
    detected_issues: List[str]
    probabilities: Dict[str, float]
    total_issues: int


class ModelLoader:
    """
    Singleton loader for trained ML models.
    Loads models lazily and caches them for reuse.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._risk_models = {}
        self._risk_metadata = None
        self._issue_models = {}
        self._issue_metadata = None
        self._resolution_models = {}
        self._anomaly_models = {}
        self._site_ranker = None
        self._initialized = True
        logger.info("ModelLoader initialized")
    
    # ==========================================
    # RISK CLASSIFIER
    # ==========================================
    
    def _load_risk_metadata(self) -> Dict[str, Any]:
        """Load risk classifier metadata"""
        if self._risk_metadata is None:
            metadata_path = RISK_MODEL_DIR / "risk_classifier_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self._risk_metadata = json.load(f)
            else:
                raise FileNotFoundError(f"Risk classifier metadata not found: {metadata_path}")
        return self._risk_metadata
    
    def _load_risk_model(self, use_calibrated: bool = True) -> Any:
        """Load risk classifier model (XGBoost + LightGBM ensemble)"""
        model_key = "calibrated" if use_calibrated else "raw"
        
        if model_key not in self._risk_models:
            suffix = "_calibrated" if use_calibrated else ""
            
            xgb_path = RISK_MODEL_DIR / f"risk_classifier_xgb{suffix}.pkl"
            lgb_path = RISK_MODEL_DIR / f"risk_classifier_lgb{suffix}.pkl"
            
            if xgb_path.exists() and lgb_path.exists():
                self._risk_models[f"xgb_{model_key}"] = joblib.load(xgb_path)
                self._risk_models[f"lgb_{model_key}"] = joblib.load(lgb_path)
                self._risk_models[model_key] = True
                logger.info(f"Loaded risk classifier models ({model_key})")
            else:
                raise FileNotFoundError(f"Risk classifier models not found: {xgb_path}, {lgb_path}")
        
        return self._risk_models
    
    def predict_risk(self, features: pd.DataFrame) -> List[RiskPrediction]:
        """
        Predict patient risk level using trained models.
        
        Args:
            features: DataFrame with feature columns matching training features
            
        Returns:
            List of RiskPrediction objects
        """
        metadata = self._load_risk_metadata()
        models = self._load_risk_model(use_calibrated=True)
        
        feature_names = metadata["feature_names"]
        class_names = metadata["class_names"]
        
        # Ensure features are in correct order
        available_features = [f for f in feature_names if f in features.columns]
        missing_features = [f for f in feature_names if f not in features.columns]
        
        if missing_features:
            logger.warning(f"Missing features (using 0): {missing_features}")
            for f in missing_features:
                features[f] = 0
        
        X = features[feature_names].fillna(0)
        
        # Get predictions from both models
        xgb_model = models["xgb_calibrated"]
        lgb_model = models["lgb_calibrated"]
        
        xgb_proba = xgb_model.predict_proba(X)
        lgb_proba = lgb_model.predict_proba(X)
        
        # Ensemble (weighted average)
        weights = metadata.get("config", {})
        xgb_weight = weights.get("xgb_weight", 0.5)
        lgb_weight = weights.get("lgb_weight", 0.5)
        
        ensemble_proba = xgb_weight * xgb_proba + lgb_weight * lgb_proba
        
        # Generate predictions
        results = []
        for i in range(len(X)):
            probs = ensemble_proba[i]
            pred_idx = np.argmax(probs)
            risk_level = class_names[pred_idx]
            confidence = float(probs[pred_idx])
            
            prob_dict = {class_names[j]: float(probs[j]) for j in range(len(class_names))}
            
            # Get top contributing features (by absolute value)
            row_values = X.iloc[i].to_dict()
            sorted_features = sorted(row_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            
            results.append(RiskPrediction(
                risk_level=risk_level,
                probabilities=prob_dict,
                confidence=confidence,
                top_features=sorted_features
            ))
        
        return results
    
    # ==========================================
    # ISSUE DETECTOR
    # ==========================================
    
    def _load_issue_metadata(self) -> Dict[str, Any]:
        """Load issue detector metadata"""
        if self._issue_metadata is None:
            metadata_path = ISSUE_MODEL_DIR / "issue_detector_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self._issue_metadata = json.load(f)
            else:
                raise FileNotFoundError(f"Issue detector metadata not found: {metadata_path}")
        return self._issue_metadata
    
    def _load_issue_model(self, issue_type: str) -> Any:
        """Load issue detector model for a specific issue type"""
        if issue_type not in self._issue_models:
            model_path = ISSUE_MODEL_DIR / f"issue_detector_{issue_type}.pkl"
            if model_path.exists():
                self._issue_models[issue_type] = joblib.load(model_path)
                logger.info(f"Loaded issue detector for: {issue_type}")
            else:
                raise FileNotFoundError(f"Issue detector model not found: {model_path}")
        return self._issue_models[issue_type]
    
    def detect_issues(self, features: pd.DataFrame, threshold: float = 0.5) -> List[IssuePrediction]:
        """
        Detect issues for patients using trained models.
        
        Args:
            features: DataFrame with feature columns
            threshold: Probability threshold for issue detection
            
        Returns:
            List of IssuePrediction objects
        """
        metadata = self._load_issue_metadata()
        issue_types = metadata["trained_models"]
        
        results = []
        
        for i in range(len(features)):
            row = features.iloc[[i]]
            detected = []
            probs = {}
            
            for issue_type in issue_types:
                try:
                    model = self._load_issue_model(issue_type)
                    
                    # Get probability
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(row)[0, 1]
                    else:
                        prob = float(model.predict(row)[0])
                    
                    probs[issue_type] = float(prob)
                    
                    if prob >= threshold:
                        detected.append(issue_type)
                        
                except Exception as e:
                    logger.warning(f"Error predicting {issue_type}: {e}")
                    probs[issue_type] = 0.0
            
            results.append(IssuePrediction(
                detected_issues=detected,
                probabilities=probs,
                total_issues=len(detected)
            ))
        
        return results
    
    # ==========================================
    # SITE RISK RANKER
    # ==========================================
    
    def get_site_risk_scores(self) -> pd.DataFrame:
        """Get pre-computed site risk rankings"""
        ranking_path = SITE_RANKER_DIR / "site_risk_ranking.csv"
        if ranking_path.exists():
            return pd.read_csv(ranking_path)
        else:
            raise FileNotFoundError(f"Site ranking not found: {ranking_path}")
    
    def get_site_rank(self, site_id: str) -> Dict[str, Any]:
        """Get risk ranking for a specific site"""
        rankings = self.get_site_risk_scores()
        site_data = rankings[rankings["site_id"] == site_id]
        
        if len(site_data) == 0:
            return {"error": f"Site {site_id} not found in rankings"}
        
        row = site_data.iloc[0]
        return {
            "site_id": site_id,
            "risk_score": float(row.get("risk_score", row.get("composite_score", 0))),
            "rank": int(row.get("rank", 0)),
            "percentile": float(row.get("percentile", 0)),
            "risk_tier": row.get("risk_tier", row.get("tier", "Unknown")),
        }
    
    # ==========================================
    # RESOLUTION TIME PREDICTOR
    # ==========================================

    def _load_resolution_models(self):
        """Load resolution time prediction models."""
        if not self._resolution_models or self._resolution_models == {}:
            try:
                predictor = ResolutionTimePredictor()
                model_path = RESOLUTION_MODEL_DIR / "resolution_time_model.pkl"
                
                # Check if unified model exists
                if model_path.exists():
                    predictor.load(model_path)
                    self._resolution_models = predictor
                else:
                    # Fallback to loading individual quantiles (Legacy support)
                    logger.warning(f"Unified resolution model not found at {model_path}. Attempting legacy load...")
                    legacy_models = {}
                    for q in [0.1, 0.5, 0.9]:
                        path = RESOLUTION_MODEL_DIR / f"resolution_time_q{q}_latest.pkl"
                        if path.exists():
                            legacy_models[str(q)] = joblib.load(path)
                    
                    if legacy_models:
                        predictor.models = legacy_models
                        predictor.is_fitted = True
                        predictor.feature_names = ['priority_score', 'site_performance_score', 'workload_index', 'issue_type_cat']
                        self._resolution_models = predictor
                    else:
                        logger.error("No resolution time models found. Predictions will be unavailable.")
                        # Return an empty object that won't crash but won't predict
                        self._resolution_models = None
            except Exception as e:
                logger.error(f"Failed to load resolution models: {e}")
                self._resolution_models = None
                    
        return self._resolution_models

    def predict_resolution_time(self, features: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Predict resolution time with confidence intervals using the predictor class.
        Returns list of dicts: {prediction_days, lower_bound_days, upper_bound_days}
        """
        if self._resolution_models is None:
             # Return dummy results if model not available
             return [{'prediction_days': 0.0, 'lower_bound_days': 0.0, 'upper_bound_days': 0.0}] * len(features)
             
        predictor = self._load_resolution_models()
        if predictor is None:
             return [{'prediction_days': 0.0, 'lower_bound_days': 0.0, 'upper_bound_days': 0.0}] * len(features)
             
        results_df = predictor.predict(features)
        
        return results_df[[
            'predicted_resolution_days', 
            'resolution_lower_bound', 
            'resolution_upper_bound'
        ]].rename(columns={
            'predicted_resolution_days': 'prediction_days',
            'resolution_lower_bound': 'lower_bound_days',
            'resolution_upper_bound': 'upper_bound_days'
        }).to_dict('records')

    # ==========================================
    # ANOMALY DETECTOR
    # ==========================================

    def _load_anomaly_model(self) -> Any:
        """Load anomaly detection model."""
        if 'model' not in self._anomaly_models:
            model_path = ANOMALY_MODEL_DIR / "anomaly_model_latest.pkl"
            scaler_path = ANOMALY_MODEL_DIR / "anomaly_scaler_latest.pkl"
            
            if model_path.exists() and scaler_path.exists():
                self._anomaly_models['model'] = joblib.load(model_path)
                self._anomaly_models['scaler'] = joblib.load(scaler_path)
            else:
                raise FileNotFoundError("Anomaly detector model/scaler not found.")
                
        return self._anomaly_models

    def detect_site_anomalies(self, site_metrics: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalies in site metrics.
        Returns list of dicts: {is_anomaly, anomaly_score}
        """
        components = self._load_anomaly_model()
        model = components['model']
        scaler = components['scaler']
        
        # Features used in training: ['query_rate', 'visit_compliance', 'dqi_score', 'enrollment_rate']
        features = ['query_rate', 'visit_compliance', 'dqi_score', 'enrollment_rate']
        
        X = site_metrics.reindex(columns=features, fill_value=0)
        X_scaled = scaler.transform(X)
        
        is_anomaly = model.predict(X_scaled) # -1 is anomaly, 1 is normal
        scores = model.decision_function(X_scaled) # Lower is more anomalous
        
        results = []
        for i in range(len(X)):
            results.append({
                "is_anomaly": bool(is_anomaly[i] == -1),
                "anomaly_score": float(scores[i])
            })
            
        return results
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def health_check(self) -> Dict[str, Any]:
        """Check if all models are loadable"""
        status = {
            "risk_classifier": False,
            "issue_detector": False,
            "site_ranker": False,
        }
        
        try:
            self._load_risk_metadata()
            self._load_risk_model(use_calibrated=True)
            status["risk_classifier"] = True
        except Exception as e:
            status["risk_classifier_error"] = str(e)
        
        try:
            self._load_issue_metadata()
            self._load_issue_model("open_queries")  # Test one model
            status["issue_detector"] = True
        except Exception as e:
            status["issue_detector_error"] = str(e)
        
        try:
            self.get_site_risk_scores()
            status["site_ranker"] = True
        except Exception as e:
            status["site_ranker_error"] = str(e)

        try:
            self._load_resolution_models()
            status["resolution_predictor"] = True
        except Exception as e:
            status["resolution_predictor_error"] = str(e)
            
        try:
            self._load_anomaly_model()
            status["anomaly_detector"] = True
        except Exception as e:
            status["anomaly_detector_error"] = str(e)
        
        return status
    
    def get_risk_feature_names(self) -> List[str]:
        """Get required feature names for risk prediction"""
        metadata = self._load_risk_metadata()
        return metadata["feature_names"]
    
    def get_issue_types(self) -> List[str]:
        """Get available issue types"""
        metadata = self._load_issue_metadata()
        return metadata["trained_models"]


# Singleton instance
_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create the model loader singleton"""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader
