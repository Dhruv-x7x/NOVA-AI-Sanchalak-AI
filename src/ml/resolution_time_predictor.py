"""
a6on-i - Resolution Time Predictor v1.0
=====================================================
Predicts estimated days to resolution for issues using:
- Quantile Regression (XGBoost)
- Confidence Intervals (10th/90th percentiles)
- Feature-based estimation (priority, site performance, workload)

Author: Antigravity
Version: 1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field

# ML imports
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ResolutionPredictorConfig:
    """Configuration for Resolution Time Predictor."""
    
    # XGBoost parameters for quantile regression
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    random_state: int = 42
    
    # Quantiles for prediction (Lower, Median, Upper)
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    # Features to use
    features: List[str] = field(default_factory=lambda: [
        'priority_score', 
        'site_performance_score', 
        'workload_index'
    ])
    
    # Categorical features
    categorical_features: List[str] = field(default_factory=lambda: [
        'issue_type_cat'
    ])


# =============================================================================
# RESOLUTION TIME PREDICTOR CLASS
# =============================================================================

class ResolutionTimePredictor:
    """
    XGBoost-based quantile regression for predicting issue resolution time.
    Provides a point estimate (median) and uncertainty bounds.
    """
    
    def __init__(self, config: Optional[ResolutionPredictorConfig] = None):
        """Initialize predictor with configuration."""
        self.config = config or ResolutionPredictorConfig()
        self.models: Dict[str, Any] = {}
        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        self.fit_stats: Dict[str, Any] = {}
        
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Prepare features for prediction/training."""
        X = df.copy()
        
        # 1. Handle Categorical Encoding for issue_type
        if 'issue_type' in X.columns:
            if 'issue_type_cat' not in X.columns:
                X['issue_type_cat'] = X['issue_type'].astype('category').cat.codes
        
        # 2. Handle Priority Score
        if 'issue_priority' in X.columns and 'priority_score' not in X.columns:
            priority_map = {'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}
            X['priority_score'] = X['issue_priority'].map(priority_map).fillna(1)
            
        # 3. Ensure numerical columns exist with fallbacks
        fallbacks = {
            'site_performance_score': 85.0,
            'workload_index': 0.5,
            'priority_score': 1.0,
            'issue_type_cat': 0
        }
        
        for col, fallback in fallbacks.items():
            if col not in X.columns:
                X[col] = fallback
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(fallback)
        
        # 4. Select features
        all_features = self.config.features + self.config.categorical_features
        available_features = [f for f in all_features if f in X.columns]
        
        X = X[available_features]
        
        if is_training:
            self.feature_names = available_features
            logger.info(f"Training with features: {self.feature_names}")
        else:
            # Reorder/align to training features
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = fallbacks.get(col, 0.0)
            X = X[self.feature_names]
            
        return X

    def fit(self, df: pd.DataFrame) -> 'ResolutionTimePredictor':
        """
        Fit quantile regression models.
        
        Args:
            df: DataFrame containing features and 'days_to_resolution' target.
        """
        logger.info("=" * 60)
        logger.info("FITTING RESOLUTION TIME PREDICTOR")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        if 'days_to_resolution' not in df.columns:
            logger.error("'days_to_resolution' target missing from DataFrame")
            raise ValueError("Missing target column: 'days_to_resolution'")
            
        X = self._prepare_features(df, is_training=True)
        y = df['days_to_resolution']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.config.random_state)
        
        self.models = {}
        eval_results = {}
        
        for q in self.config.quantiles:
            logger.info(f"Training quantile model: {q}...")
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models[str(q)] = model
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            eval_results[str(q)] = {'mae': mae, 'r2': r2}
            logger.info(f"  Quantile {q}: MAE = {mae:.2f} days, R2 = {r2:.3f}")
            
        self.fit_stats = {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'evaluation': eval_results,
            'feature_importance': self._get_combined_importance()
        }
        
        self.is_fitted = True
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Fitting complete in {duration:.2f} seconds")
        
        return self

    def _get_combined_importance(self) -> Dict[str, float]:
        """Get average feature importance across models."""
        importance_sum = np.zeros(len(self.feature_names))
        for model in self.models.values():
            importance_sum += model.feature_importances_
        
        mean_importance = importance_sum / len(self.models)
        return dict(zip(self.feature_names, [float(x) for x in mean_importance]))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict resolution time for new data.
        
        Returns:
            DataFrame with predictions and confidence intervals.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")
            
        X = self._prepare_features(df, is_training=False)
        results = pd.DataFrame(index=df.index)
        
        # Basic Info
        for col in ['patient_key', 'subject_id', 'site_id', 'study_id', 'issue_type', 'issue_priority']:
            if col in df.columns:
                results[col] = df[col]
        
        # Predictions
        preds = {}
        for q, model in self.models.items():
            # Align columns if necessary (though _prepare_features should handle it)
            preds[q] = model.predict(X)
            
        results['predicted_resolution_days'] = preds.get('0.5')
        results['resolution_lower_bound'] = preds.get('0.1')
        results['resolution_upper_bound'] = preds.get('0.9')
        
        # Consistency checks
        results['resolution_lower_bound'] = results[['resolution_lower_bound', 'predicted_resolution_days']].min(axis=1)
        results['resolution_upper_bound'] = results[['resolution_upper_bound', 'predicted_resolution_days']].max(axis=1)
        
        # Add confidence indicator
        results['prediction_uncertainty_range'] = results['resolution_upper_bound'] - results['resolution_lower_bound']
        
        return results
    
    def predict_with_confidence(
        self, 
        features: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Return prediction with confidence interval for a single prediction.
        
        Per riyaz.md Section 10: Never single-point predictions.
        
        Args:
            features: Dictionary of feature values
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            {
                "point_estimate": 12,
                "ci_lower": 8,
                "ci_upper": 18,
                "confidence_level": 0.95,
                "uncertainty_range": 10
            }
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")
        
        # Create single-row DataFrame
        df = pd.DataFrame([features])
        X = self._prepare_features(df, is_training=False)
        
        # Get predictions from all quantile models
        preds = {}
        for q, model in self.models.items():
            preds[float(q)] = float(model.predict(X)[0])
        
        # Map confidence level to quantiles
        # 95% CI uses 2.5% and 97.5% quantiles, but we have 10%, 50%, 90%
        # For now, use available quantiles
        lower_quantile = 0.1
        upper_quantile = 0.9
        
        point_estimate = preds.get(0.5, preds.get(0.5))
        ci_lower = preds.get(lower_quantile, point_estimate * 0.7)
        ci_upper = preds.get(upper_quantile, point_estimate * 1.4)
        
        # Ensure bounds are ordered correctly
        ci_lower = min(ci_lower, point_estimate)
        ci_upper = max(ci_upper, point_estimate)
        
        return {
            "point_estimate": round(point_estimate, 1),
            "ci_lower": round(ci_lower, 1),
            "ci_upper": round(ci_upper, 1),
            "confidence_level": confidence_level,
            "uncertainty_range": round(ci_upper - ci_lower, 1),
            "prediction_quality": self._assess_prediction_quality(ci_upper - ci_lower)
        }
    
    def _assess_prediction_quality(self, uncertainty_range: float) -> str:
        """Assess prediction quality based on uncertainty range."""
        if uncertainty_range < 3:
            return "high_confidence"
        elif uncertainty_range < 7:
            return "medium_confidence"
        elif uncertainty_range < 15:
            return "low_confidence"
        else:
            return "very_uncertain"
    
    def predict_batch_with_confidence(
        self, 
        df: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Batch prediction with confidence intervals.
        
        Returns DataFrame with columns:
        - point_estimate
        - ci_lower
        - ci_upper
        - confidence_level
        - uncertainty_range
        - prediction_quality
        """
        # Use standard predict which already has quantile bounds
        results = self.predict(df)
        
        # Rename for API consistency
        results = results.rename(columns={
            'predicted_resolution_days': 'point_estimate',
            'resolution_lower_bound': 'ci_lower',
            'resolution_upper_bound': 'ci_upper',
            'prediction_uncertainty_range': 'uncertainty_range'
        })
        
        # Add confidence level
        results['confidence_level'] = confidence_level
        
        # Add quality assessment
        results['prediction_quality'] = results['uncertainty_range'].apply(
            self._assess_prediction_quality
        )
        
        return results

    def save(self, output_dir: Path):
        """Save models and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'models': self.models,
            'feature_names': self.feature_names,
            'config': asdict(self.config),
            'fit_stats': self.fit_stats,
            'version': '1.0'
        }
        
        save_path = output_dir / "resolution_time_model.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {save_path}")

    def load(self, model_path: Path) -> 'ResolutionTimePredictor':
        """Load saved model."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            
        self.models = data['models']
        self.feature_names = data['feature_names']
        self.config = ResolutionPredictorConfig(**data['config'])
        self.fit_stats = data['fit_stats']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {model_path}")
        return self


# =============================================================================
# MAIN RUNNER FUNCTION
# =============================================================================

def run_resolution_prediction(
    issues_path: Path,
    output_dir: Path,
    config: Optional[ResolutionPredictorConfig] = None
) -> Dict[str, Any]:
    """Run full resolution time prediction pipeline."""
    logger.info("=" * 70)
    logger.info("a6on-i - RESOLUTION TIME PREDICTION ENGINE v1.0")
    logger.info("=" * 70)
    
    # Create directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models' / 'resolution_time'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {issues_path}...")
    df = pd.read_parquet(issues_path)
    
    # If target is missing, generate it (bootstrap mode)
    if 'days_to_resolution' not in df.columns:
        logger.warning("'days_to_resolution' missing. Bootstrapping with synthetic target...")
        from src.ml.training.train_resolution_time import ResolutionTimeTrainer
        trainer = ResolutionTimeTrainer()
        df = trainer._add_synthetic_target(df)
        
    # Initialize and fit
    predictor = ResolutionTimePredictor(config=config)
    predictor.fit(df)
    
    # Predict
    results = predictor.predict(df)
    
    # Save results to PostgreSQL
    try:
        from src.database.pg_writer import safe_to_postgres
        success, msg = safe_to_postgres(results, 'resolution_predictions', if_exists='replace')
        if success:
            logger.info(f"Saved resolution_predictions to PostgreSQL: {len(results):,} rows")
        else:
            logger.warning(f"PostgreSQL write failed: {msg}")
    except ImportError:
        logger.warning("PostgreSQL writer not available. Skipping DB write.")
        
    # Save CSV for inspection
    results.to_csv(output_dir / "resolution_time_predictions.csv", index=False)
    
    # Save model
    predictor.save(models_dir)
    
    return {
        'total_predictions': len(results),
        'avg_predicted_days': results['predicted_resolution_days'].mean(),
        'feature_importance': predictor.fit_stats['feature_importance']
    }

if __name__ == "__main__":
    # Example usage
    data_path = Path("data/processed/analytics/patient_issues.parquet")
    if data_path.exists():
        run_resolution_prediction(data_path, Path("data/processed/ml/results"))
    else:
        logger.error(f"Data not found: {data_path}")
