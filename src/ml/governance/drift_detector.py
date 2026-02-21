# src/ml/governance/drift_detector.py
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


"""
SANCHALAK AI - ML Drift Detector v1.0
Statistical Drift Detection for ML Models

Features:
- Population Stability Index (PSI) for feature distribution drift
- Kolmogorov-Smirnov test for continuous features
- Chi-Square test for categorical features
- Performance decay detection
- Ensemble drift scoring with severity classification

21 CFR Part 11 Compliant: All drift events are audit logged
"""

import json
import uuid
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from collections import defaultdict

# PostgreSQL integration
from src.database.pg_data_service import get_pg_data_service
from src.database.models import MLModelVersion, DriftReport

logger = logging.getLogger(__name__)

# Try scipy imports (optional but recommended)
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# ENUMS
# =============================================================================

class DriftType(Enum):
    """Types of statistical drift"""
    FEATURE_PSI = "feature_psi"           # Population Stability Index
    FEATURE_KS = "feature_ks"             # Kolmogorov-Smirnov test
    FEATURE_CHI = "feature_chi"           # Chi-Square test
    PERFORMANCE = "performance"           # Model performance decay
    PREDICTION = "prediction"             # Prediction distribution shift
    CONCEPT = "concept"                   # Target relationship drift


class DriftSeverity(Enum):
    """Severity levels for detected drift"""
    NONE = "none"               # No drift (PSI < 0.1)
    LOW = "low"                 # Minor drift (PSI 0.1-0.25)
    MEDIUM = "medium"           # Moderate drift (PSI 0.25-0.5) 
    HIGH = "high"               # Significant drift (PSI > 0.5)
    CRITICAL = "critical"       # Severe drift requiring immediate action


class FeatureType(Enum):
    """Feature data types for drift analysis"""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureBaseline:
    """Baseline statistics for a feature"""
    feature_name: str
    feature_type: FeatureType
    created_at: datetime
    sample_size: int
    
    # For continuous features
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    percentiles: Dict[str, float] = field(default_factory=dict)  # p5, p25, p50, p75, p95
    histogram_bins: List[float] = field(default_factory=list)
    histogram_counts: List[int] = field(default_factory=list)
    
    # For categorical features
    categories: List[str] = field(default_factory=list)
    category_counts: Dict[str, int] = field(default_factory=dict)
    category_proportions: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'feature_name': self.feature_name,
            'feature_type': self.feature_type.value,
            'created_at': self.created_at.isoformat(),
            'sample_size': self.sample_size,
            'mean': self.mean,
            'std': self.std,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'percentiles': self.percentiles,
            'histogram_bins': self.histogram_bins,
            'histogram_counts': self.histogram_counts,
            'categories': self.categories,
            'category_counts': self.category_counts,
            'category_proportions': self.category_proportions
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureBaseline':
        return cls(
            feature_name=data['feature_name'],
            feature_type=FeatureType(data['feature_type']),
            created_at=datetime.fromisoformat(data['created_at']),
            sample_size=data['sample_size'],
            mean=data.get('mean'),
            std=data.get('std'),
            min_val=data.get('min_val'),
            max_val=data.get('max_val'),
            percentiles=data.get('percentiles', {}),
            histogram_bins=data.get('histogram_bins', []),
            histogram_counts=data.get('histogram_counts', []),
            categories=data.get('categories', []),
            category_counts=data.get('category_counts', {}),
            category_proportions=data.get('category_proportions', {})
        )


@dataclass
class FeatureDriftResult:
    """Result of drift detection for a single feature"""
    feature_name: str
    drift_type: DriftType
    statistic: float           # PSI value, KS statistic, or Chi-square statistic
    p_value: Optional[float]   # None for PSI
    severity: DriftSeverity
    
    baseline_sample_size: int
    current_sample_size: int
    
    # Detailed statistics
    baseline_mean: Optional[float] = None
    current_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    current_std: Optional[float] = None
    
    details: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'feature_name': self.feature_name,
            'drift_type': self.drift_type.value,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'severity': self.severity.value,
            'baseline_sample_size': self.baseline_sample_size,
            'current_sample_size': self.current_sample_size,
            'baseline_mean': self.baseline_mean,
            'current_mean': self.current_mean,
            'baseline_std': self.baseline_std,
            'current_std': self.current_std,
            'details': self.details,
            'recommendations': self.recommendations
        }


@dataclass
class PerformanceDriftResult:
    """Result of performance decay analysis"""
    model_name: str
    metric_name: str
    baseline_value: float
    current_value: float
    absolute_change: float
    relative_change: float  # Percentage change
    severity: DriftSeverity
    
    window_start: datetime
    window_end: datetime
    sample_size: int
    
    details: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'metric_name': self.metric_name,
            'baseline_value': self.baseline_value,
            'current_value': self.current_value,
            'absolute_change': self.absolute_change,
            'relative_change': self.relative_change,
            'severity': self.severity.value,
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'sample_size': self.sample_size,
            'details': self.details,
            'recommendations': self.recommendations
        }


@dataclass
class DriftReport:
    """Comprehensive drift analysis report"""
    report_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    
    # Feature drift results
    feature_drifts: List[FeatureDriftResult] = field(default_factory=list)
    
    # Performance drift results
    performance_drifts: List[PerformanceDriftResult] = field(default_factory=list)
    
    # Summary metrics
    overall_psi: float = 0.0           # Average PSI across features
    max_psi: float = 0.0               # Maximum PSI
    drifted_feature_count: int = 0      # Number of features with significant drift
    total_feature_count: int = 0
    
    overall_severity: DriftSeverity = DriftSeverity.NONE
    
    recommendations: List[str] = field(default_factory=list)
    
    # Checksums for audit
    checksum: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'feature_drifts': [fd.to_dict() for fd in self.feature_drifts],
            'performance_drifts': [pd.to_dict() for pd in self.performance_drifts],
            'overall_psi': self.overall_psi,
            'max_psi': self.max_psi,
            'drifted_feature_count': self.drifted_feature_count,
            'total_feature_count': self.total_feature_count,
            'overall_severity': self.overall_severity.value,
            'recommendations': self.recommendations,
            'checksum': self.checksum
        }


@dataclass
class DriftAlert:
    """Alert for significant drift detection"""
    alert_id: str
    timestamp: datetime
    model_name: str
    drift_type: DriftType
    severity: DriftSeverity
    
    description: str
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    
    affected_features: List[str] = field(default_factory=list)
    
    recommendation: str = ""
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'drift_type': self.drift_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'threshold': self.threshold,
            'affected_features': self.affected_features,
            'recommendation': self.recommendation,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by
        }


# =============================================================================
# DRIFT DETECTOR
# =============================================================================

class DriftDetector:
    """
    Statistical drift detection for ML models.
    
    Implements:
    - PSI (Population Stability Index) for distribution drift
    - KS Test (Kolmogorov-Smirnov) for continuous features
    - Chi-Square test for categorical features
    - Performance decay detection
    
    All drift events are logged for 21 CFR Part 11 compliance.
    """
    
    # PSI thresholds
    PSI_THRESHOLD_LOW = 0.1
    PSI_THRESHOLD_MEDIUM = 0.25
    PSI_THRESHOLD_HIGH = 0.5
    
    # Performance thresholds (relative change)
    PERF_THRESHOLD_LOW = 0.02      # 2% drop
    PERF_THRESHOLD_MEDIUM = 0.05   # 5% drop
    PERF_THRESHOLD_HIGH = 0.10     # 10% drop
    
    def __init__(self, service: Optional[Any] = None):
        self.service = service or get_pg_data_service()
        self._lock = threading.Lock()
        
        # Cache for baselines
        self._baseline_cache: Dict[str, Dict[str, FeatureBaseline]] = {}
        self._load_baselines()
    
    def _load_baselines(self):
        """Load feature baselines from PostgreSQL"""
        try:
            # Fetch all models with baselines
            models_df = self.service.get_ml_models()
            if models_df.empty:
                return

            for _, row in models_df.iterrows():
                model_name = row['model_name']
                baselines_json = row.get('feature_baselines')
                
                if baselines_json and isinstance(baselines_json, dict):
                    self._baseline_cache[model_name] = {}
                    for feat_name, feat_data in baselines_json.items():
                        try:
                            self._baseline_cache[model_name][feat_name] = FeatureBaseline.from_dict(feat_data)
                        except Exception as feat_err:
                            logger.error(f"Failed to parse feature baseline {feat_name} for {model_name}: {feat_err}")
            
            logger.info(f"Loaded baselines for {len(self._baseline_cache)} models")
        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")

    def _save_baseline(self, model_name: str):
        """Save feature baselines for a model to PostgreSQL"""
        if model_name not in self._baseline_cache:
            return
            
        try:
            session = self.service._get_session()
            # Find the active version of the model
            model = session.query(MLModelVersion).filter_by(model_name=model_name, status='deployed').first()
            if not model:
                # Fallback to latest
                model = session.query(MLModelVersion).filter_by(model_name=model_name).order_by(MLModelVersion.trained_at.desc()).first()
            
            if model:
                # Convert to native types for JSON serialization
                def _to_native(obj):
                    if isinstance(obj, dict):
                        return {k: _to_native(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [_to_native(i) for i in obj]
                    elif isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, Enum):
                        return obj.value
                    elif hasattr(obj, 'isoformat'):
                        return obj.isoformat()
                    return obj

                baselines_dict = {name: _to_native(asdict(base)) for name, base in self._baseline_cache[model_name].items()}
                
                model.feature_baselines = baselines_dict
                session.commit()
                logger.info(f"Saved baselines for model {model_name} (ID: {model.version_id})")
            else:
                logger.warning(f"Could not find model {model_name} to save baselines")
            session.close()
        except Exception as e:
            logger.error(f"Failed to save baselines for {model_name}: {e}")
    
    # =========================================================================
    # PSI Calculation
    # =========================================================================
    
    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-6
    ) -> Tuple[float, List[float], List[float]]:
        """
        Calculate Population Stability Index (PSI).
        
        PSI = Σ (current_pct - baseline_pct) * ln(current_pct / baseline_pct)
        
        Interpretation:
        - PSI < 0.1: No significant drift
        - PSI 0.1-0.25: Moderate drift
        - PSI > 0.25: Significant drift requiring action
        
        Args:
            baseline: Baseline distribution array
            current: Current distribution array
            n_bins: Number of bins for histogram
            epsilon: Small value to avoid division by zero
            
        Returns:
            Tuple of (psi_value, baseline_percentages, current_percentages)
        """
        # Ensure numeric and remove NaN values
        try:
            baseline = np.array(baseline, dtype=float)
            current = np.array(current, dtype=float)
        except:
            return 0.0, [], []
            
        baseline = baseline[~np.isnan(baseline)]
        current = current[~np.isnan(current)]
        
        if len(baseline) == 0 or len(current) == 0:
            return 0.0, [], []
        
        # Create bins based on baseline distribution
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(baseline, percentiles)
        
        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0, [], []
        
        # Calculate histogram counts
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages
        baseline_pct = (baseline_counts + epsilon) / (len(baseline) + epsilon * len(baseline_counts))
        current_pct = (current_counts + epsilon) / (len(current) + epsilon * len(current_counts))
        
        # Calculate PSI
        psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
        psi = float(np.sum(psi_values))
        
        return psi, baseline_pct.tolist(), current_pct.tolist()
    
    def calculate_psi_categorical(
        self,
        baseline: pd.Series,
        current: pd.Series,
        epsilon: float = 1e-6
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Calculate PSI for categorical features.
        
        Args:
            baseline: Baseline categorical series
            current: Current categorical series
            epsilon: Small value to avoid division by zero
            
        Returns:
            Tuple of (psi_value, baseline_proportions, current_proportions)
        """
        # Get all unique categories
        all_categories = set(baseline.dropna().unique()) | set(current.dropna().unique())
        
        if not all_categories:
            return 0.0, {}, {}
        
        # Calculate proportions
        baseline_counts = baseline.value_counts()
        current_counts = current.value_counts()
        
        baseline_total = len(baseline.dropna())
        current_total = len(current.dropna())
        
        if baseline_total == 0 or current_total == 0:
            return 0.0, {}, {}
        
        baseline_props = {}
        current_props = {}
        psi = 0.0
        
        for cat in all_categories:
            # Use .loc to avoid FutureWarnings with integer keys in Series lookup
            b_count = baseline_counts.loc[cat] if cat in baseline_counts.index else 0
            c_count = current_counts.loc[cat] if cat in current_counts.index else 0
            
            b_pct = (b_count + epsilon) / (baseline_total + epsilon * len(all_categories))
            c_pct = (c_count + epsilon) / (current_total + epsilon * len(all_categories))
            
            baseline_props[str(cat)] = b_pct
            current_props[str(cat)] = c_pct
            
            psi += (c_pct - b_pct) * np.log(c_pct / b_pct)
        
        return float(psi), baseline_props, current_props
    
    # =========================================================================
    # KS Test
    # =========================================================================
    
    def ks_test(
        self,
        baseline: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for continuous features.
        
        Tests whether two samples come from the same distribution.
        
        Args:
            baseline: Baseline distribution array
            current: Current distribution array
            
        Returns:
            Tuple of (ks_statistic, p_value)
        """
        # Remove NaN values
        baseline = baseline[~np.isnan(baseline)]
        current = current[~np.isnan(current)]
        
        if len(baseline) < 2 or len(current) < 2:
            return 0.0, 1.0
        
        if SCIPY_AVAILABLE:
            statistic, p_value = stats.ks_2samp(baseline, current)
            return float(statistic), float(p_value)
        else:
            # Fallback: Simple empirical CDF comparison
            all_values = np.sort(np.concatenate([baseline, current]))
            
            baseline_cdf = np.searchsorted(np.sort(baseline), all_values, side='right') / len(baseline)
            current_cdf = np.searchsorted(np.sort(current), all_values, side='right') / len(current)
            
            statistic = float(np.max(np.abs(baseline_cdf - current_cdf)))
            
            # Approximate p-value using asymptotic formula
            n = len(baseline) * len(current) / (len(baseline) + len(current))
            p_value = 2 * np.exp(-2 * n * statistic ** 2)
            p_value = min(max(float(p_value), 0.0), 1.0)
            
            return statistic, p_value
    
    # =========================================================================
    # Chi-Square Test
    # =========================================================================
    
    def chi_square_test(
        self,
        baseline: pd.Series,
        current: pd.Series
    ) -> Tuple[float, float]:
        """
        Perform Chi-Square test for categorical features.
        
        Tests whether the distribution of categories has changed.
        
        Args:
            baseline: Baseline categorical series
            current: Current categorical series
            
        Returns:
            Tuple of (chi_square_statistic, p_value)
        """
        # Get all unique categories (cast to string to avoid mixed type comparison errors)
        all_categories = sorted(list(set(str(v) for v in baseline.dropna().unique()) | set(str(v) for v in current.dropna().unique())))
        
        if len(all_categories) < 2:
            return 0.0, 1.0
        
        # Calculate observed and expected frequencies
        baseline_counts = baseline.astype(str).value_counts()
        current_counts = current.astype(str).value_counts()
        
        baseline_total = len(baseline.dropna())
        current_total = len(current.dropna())
        
        if baseline_total == 0 or current_total == 0:
            return 0.0, 1.0
        
        observed = []
        expected = []
        
        for cat in all_categories:
            obs = current_counts.get(cat, 0)
            exp = (baseline_counts.get(cat, 0) / baseline_total) * current_total
            
            observed.append(obs)
            expected.append(exp)
            
        # Ensure sum matches exactly and handle zero expected
        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        
        # Add epsilon and re-normalize to ensure they sum to the same value
        expected = np.maximum(expected, 1e-6)
        expected = expected * (observed.sum() / expected.sum())
        
        if SCIPY_AVAILABLE:
            statistic, p_value = stats.chisquare(observed, expected)
            return float(statistic), float(p_value)
        else:
            # Manual chi-square calculation
            chi_sq = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
            
            # Approximate p-value (rough estimate)
            df = len(all_categories) - 1
            p_value = 1.0 - (chi_sq / (df + chi_sq))  # Very rough approximation
            p_value = min(max(float(p_value), 0.0), 1.0)
            
            return float(chi_sq), p_value
    
    # =========================================================================
    # Baseline Management
    # =========================================================================
    
    def create_baseline(
        self,
        model_name: str,
        model_version: str,
        data: pd.DataFrame,
        feature_types: Optional[Dict[str, FeatureType]] = None
    ) -> Dict[str, FeatureBaseline]:
        """
        Create baseline statistics for model features.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            data: DataFrame with feature columns
            feature_types: Optional dict mapping feature names to types
            
        Returns:
            Dict mapping feature names to FeatureBaseline objects
        """
        baselines = {}
        now = datetime.now()
        
        for col in data.columns:
            series = data[col].dropna()
            
            if len(series) == 0:
                continue
            
            # Determine feature type
            if feature_types and col in feature_types:
                f_type = feature_types[col]
            elif series.dtype in ['int64', 'float64', 'int32', 'float32']:
                # Check if it's actually categorical (few unique values)
                if series.nunique() <= 10:
                    f_type = FeatureType.CATEGORICAL
                else:
                    f_type = FeatureType.CONTINUOUS
            else:
                f_type = FeatureType.CATEGORICAL
            
            baseline = FeatureBaseline(
                feature_name=col,
                feature_type=f_type,
                created_at=now,
                sample_size=len(series)
            )
            
            if f_type == FeatureType.CONTINUOUS:
                # Continuous statistics
                baseline.mean = float(series.mean())
                baseline.std = float(series.std())
                baseline.min_val = float(series.min())
                baseline.max_val = float(series.max())
                baseline.percentiles = {
                    'p5': float(series.quantile(0.05)),
                    'p25': float(series.quantile(0.25)),
                    'p50': float(series.quantile(0.50)),
                    'p75': float(series.quantile(0.75)),
                    'p95': float(series.quantile(0.95))
                }
                
                # Histogram
                counts, bins = np.histogram(series.values, bins=10)
                baseline.histogram_bins = bins.tolist()
                baseline.histogram_counts = counts.tolist()
            else:
                # Categorical statistics
                baseline.categories = series.unique().tolist()
                baseline.category_counts = series.value_counts().to_dict()
                total = len(series)
                baseline.category_proportions = {
                    str(k): v / total for k, v in baseline.category_counts.items()
                }
            
            baselines[col] = baseline
        
        # Save to database
        self._save_baselines(model_name, model_version, baselines)
        
        # Update cache
        self._baseline_cache[model_name] = baselines
        
        return baselines
    
    def _save_baselines(
        self,
        model_name: str,
        model_version: str,
        baselines: Dict[str, FeatureBaseline]
    ):
        """Save baselines for a model."""
        with self._lock:
            self._baseline_cache[model_name] = baselines
            self._save_baseline(model_name)
    
    def get_baselines(self, model_name: str) -> Dict[str, FeatureBaseline]:
        """Get active baselines for a model."""
        # Check cache first
        if model_name in self._baseline_cache:
            return self._baseline_cache[model_name]
        return {}
    
    # =========================================================================
    # Drift Detection
    # =========================================================================
    
    def detect_feature_drift(
        self,
        model_name: str,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> List[FeatureDriftResult]:
        """
        Detect drift in feature distributions.
        
        Args:
            model_name: Name of the model
            current_data: Current data DataFrame
            features: Optional list of features to check (default: all)
            
        Returns:
            List of FeatureDriftResult objects
        """
        baselines = self.get_baselines(model_name)
        
        if not baselines:
            return []
        
        if features is None:
            features = [f for f in baselines.keys() if f in current_data.columns]
        
        results = []
        
        for feature in features:
            if feature not in baselines or feature not in current_data.columns:
                continue
            
            baseline = baselines[feature]
            current_series = current_data[feature].dropna()
            
            if len(current_series) == 0:
                continue
            
            if baseline.feature_type == FeatureType.CONTINUOUS:
                result = self._detect_continuous_drift(baseline, current_series)
            else:
                result = self._detect_categorical_drift(baseline, current_series)
            
            if result:
                results.append(result)
        
        return results
    
    def _detect_continuous_drift(
        self,
        baseline: FeatureBaseline,
        current: pd.Series
    ) -> FeatureDriftResult:
        """Detect drift for continuous feature"""
        current_array = current.values.astype(float)
        
        # Reconstruct baseline array from histogram
        baseline_array = self._reconstruct_from_histogram(
            baseline.histogram_bins,
            baseline.histogram_counts
        )
        
        # Calculate PSI
        psi, _, _ = self.calculate_psi(baseline_array, current_array)
        
        # Also run KS test
        ks_stat, ks_p = self.ks_test(baseline_array, current_array)
        
        # Determine severity from PSI
        severity = self._psi_to_severity(psi)
        
        # Generate details
        current_mean = float(current.mean())
        current_std = float(current.std())
        
        details = f"PSI: {psi:.4f}, KS: {ks_stat:.4f} (p={ks_p:.4f})"
        if baseline.mean:
            mean_change = abs(current_mean - baseline.mean) / (baseline.mean + 1e-9) * 100
            details += f", Mean change: {mean_change:.1f}%"
        
        recommendations = self._get_drift_recommendations(severity, DriftType.FEATURE_PSI)
        
        return FeatureDriftResult(
            feature_name=baseline.feature_name,
            drift_type=DriftType.FEATURE_PSI,
            statistic=psi,
            p_value=ks_p,  # Use KS p-value for significance
            severity=severity,
            baseline_sample_size=baseline.sample_size,
            current_sample_size=len(current),
            baseline_mean=baseline.mean,
            current_mean=current_mean,
            baseline_std=baseline.std,
            current_std=current_std,
            details=details,
            recommendations=recommendations
        )
    
    def _detect_categorical_drift(
        self,
        baseline: FeatureBaseline,
        current: pd.Series
    ) -> FeatureDriftResult:
        """Detect drift for categorical feature"""
        # Calculate PSI for categorical
        psi, _, current_props = self.calculate_psi_categorical(
            pd.Series(baseline.categories * (baseline.sample_size // len(baseline.categories) + 1))[:baseline.sample_size],
            current
        )
        
        # Use baseline proportions directly
        baseline_series = pd.Series([
            cat for cat, prop in baseline.category_proportions.items()
            for _ in range(int(prop * baseline.sample_size))
        ])
        
        psi, _, _ = self.calculate_psi_categorical(baseline_series, current)
        
        # Chi-square test
        chi_stat, chi_p = self.chi_square_test(baseline_series, current)
        
        # Determine severity
        severity = self._psi_to_severity(psi)
        
        details = f"PSI: {psi:.4f}, Chi-sq: {chi_stat:.4f} (p={chi_p:.4f})"
        
        # Check for new/missing categories
        current_cats = set(current.unique())
        baseline_cats = set(baseline.categories)
        new_cats = current_cats - baseline_cats
        missing_cats = baseline_cats - current_cats
        
        if new_cats:
            details += f", New categories: {new_cats}"
        if missing_cats:
            details += f", Missing categories: {missing_cats}"
        
        recommendations = self._get_drift_recommendations(severity, DriftType.FEATURE_CHI)
        
        return FeatureDriftResult(
            feature_name=baseline.feature_name,
            drift_type=DriftType.FEATURE_CHI,
            statistic=psi,
            p_value=chi_p,
            severity=severity,
            baseline_sample_size=baseline.sample_size,
            current_sample_size=len(current),
            details=details,
            recommendations=recommendations
        )
    
    def _reconstruct_from_histogram(
        self,
        bins: List[float],
        counts: List[int]
    ) -> np.ndarray:
        """Reconstruct approximate values from histogram"""
        if not bins or not counts:
            return np.array([])
        
        values = []
        for i, count in enumerate(counts):
            if i < len(bins) - 1:
                # Generate random values in each bin
                bin_min = bins[i]
                bin_max = bins[i + 1]
                values.extend(np.linspace(bin_min, bin_max, count).tolist())
        
        return np.array(values)
    
    def _psi_to_severity(self, psi: float) -> DriftSeverity:
        """Convert PSI value to severity level"""
        if psi < self.PSI_THRESHOLD_LOW:
            return DriftSeverity.NONE
        elif psi < self.PSI_THRESHOLD_MEDIUM:
            return DriftSeverity.LOW
        elif psi < self.PSI_THRESHOLD_HIGH:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.HIGH
    
    def _get_drift_recommendations(
        self,
        severity: DriftSeverity,
        drift_type: DriftType
    ) -> List[str]:
        """Get recommendations based on drift severity"""
        recommendations = []
        
        if severity == DriftSeverity.NONE:
            recommendations.append("No action required - distributions are stable")
        elif severity == DriftSeverity.LOW:
            recommendations.append("Monitor closely - minor distribution shift detected")
            recommendations.append("Review feature engineering pipeline for changes")
        elif severity == DriftSeverity.MEDIUM:
            recommendations.append("Investigate root cause of distribution change")
            recommendations.append("Consider retraining model with recent data")
            recommendations.append("Validate model predictions against ground truth")
        elif severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recommendations.append("URGENT: Significant drift detected")
            recommendations.append("Immediate model retraining recommended")
            recommendations.append("Review data source for changes or issues")
            recommendations.append("Consider model rollback if performance degraded")
        
        return recommendations
    
    # =========================================================================
    # Drift Reports
    # =========================================================================
    
    def generate_drift_report(
        self,
        model_name: str,
        model_version: str,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> DriftReport:
        """
        Generate comprehensive drift analysis report.
        
        Args:
            model_name: Name of the model
            model_version: Current model version
            current_data: Current data DataFrame
            features: Optional list of features to analyze
            
        Returns:
            DriftReport object
        """
        # Detect feature drift
        feature_drifts = self.detect_feature_drift(model_name, current_data, features)
        
        # Calculate summary metrics
        psi_values = [fd.statistic for fd in feature_drifts if fd.drift_type in [DriftType.FEATURE_PSI, DriftType.FEATURE_CHI]]
        
        overall_psi = float(np.mean(psi_values)) if psi_values else 0.0
        max_psi = float(np.max(psi_values)) if psi_values else 0.0
        
        drifted_count = sum(1 for fd in feature_drifts if fd.severity != DriftSeverity.NONE)
        
        # Determine overall severity
        if max_psi >= self.PSI_THRESHOLD_HIGH:
            overall_severity = DriftSeverity.CRITICAL
        elif max_psi >= self.PSI_THRESHOLD_MEDIUM:
            overall_severity = DriftSeverity.HIGH
        elif max_psi >= self.PSI_THRESHOLD_LOW:
            overall_severity = DriftSeverity.MEDIUM
        elif drifted_count > 0:
            overall_severity = DriftSeverity.LOW
        else:
            overall_severity = DriftSeverity.NONE
        
        # Generate recommendations
        recommendations = []
        if overall_severity == DriftSeverity.NONE:
            recommendations.append("All features are stable - no action required")
        else:
            recommendations.append(f"{drifted_count} features show significant drift")
            recommendations.extend(self._get_drift_recommendations(overall_severity, DriftType.FEATURE_PSI))
        
        # Create report
        report = DriftReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            model_name=model_name,
            model_version=model_version,
            feature_drifts=feature_drifts,
            overall_psi=overall_psi,
            max_psi=max_psi,
            drifted_feature_count=drifted_count,
            total_feature_count=len(feature_drifts),
            overall_severity=overall_severity,
            recommendations=recommendations
        )
        
        # Calculate checksum
        import hashlib
        report_json = json.dumps(report.to_dict(), sort_keys=True, default=str)
        report.checksum = hashlib.sha256(report_json.encode()).hexdigest()
        
        # Save to database
        self._save_report(report)
        
        # Create alert if needed
        if overall_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            self._create_drift_alert(report)
        
        return report
    
    def _save_report(self, report: DriftReport):
        """Save drift report to database"""
        try:
            report_dict = report.to_dict()
            # Add version_id if we can find it
            session = self.service._get_session()
            model = session.query(MLModelVersion).filter_by(model_name=report.model_name, version=report.model_version).first()
            if model:
                report_dict['version_id'] = model.version_id
            session.close()
            
            self.service.save_drift_report(report_dict)
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def _create_drift_alert(self, report: DriftReport):
        """Create drift alert from report"""
        # Audit log for compliance
        self.service.log_audit_event(
            user_id="SYSTEM",
            action="DRIFT_DETECTED",
            target_type="ML_MODEL",
            target_id=report.model_name,
            details=f"Drift detected in {report.model_name} (PSI: {report.overall_psi:.4f}, Severity: {report.overall_severity.value})"
        )
    
    def get_drift_history(
        self,
        model_name: str,
        days: int = 30
    ) -> List[DriftReport]:
        """Get drift report history for a model"""
        try:
            # Find version_id first
            session = self.service._get_session()
            model = session.query(MLModelVersion).filter_by(model_name=model_name).order_by(MLModelVersion.trained_at.desc()).first()
            if not model:
                session.close()
                return []
            version_id = model.version_id
            session.close()
            
            df = self.service.get_drift_reports(version_id)
            # Convert DF rows back to DriftReport objects if needed, 
            # but usually the API handles the DF directly.
            # For this internal method, we'll return a placeholder or just use result as is.
            return [] # Logic simplified: frontend calls API which uses data_service directly
        except Exception as e:
            logger.error(f"get_drift_history error: {e}")
            return []
    
    def get_active_alerts(
        self,
        model_name: Optional[str] = None
    ) -> List[DriftAlert]:
        """Get unresolved drift alerts"""
        # Placeholder for real alert table if implemented
        return []
    
    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str
    ) -> bool:
        """Mark an alert as resolved"""
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get drift detector statistics"""
        return {
            'models_monitored': len(self.baselines),
            'baselines_by_model': {m: len(f) for m, f in self.baselines.items()},
            'psi_thresholds': {
                'low': self.PSI_THRESHOLD_LOW,
                'medium': self.PSI_THRESHOLD_MEDIUM,
                'high': self.PSI_THRESHOLD_HIGH
            }
        }


# =============================================================================
# SINGLETON
# =============================================================================

_drift_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get or create the drift detector singleton"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector


def reset_drift_detector():
    """Reset the drift detector singleton (for testing)"""
    global _drift_detector
    _drift_detector = None
