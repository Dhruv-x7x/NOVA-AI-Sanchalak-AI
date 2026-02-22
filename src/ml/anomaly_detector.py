"""
a6on-i - Anomaly Detection Engine v1.1
=====================================================
Detects unusual patterns in patient and site data using:
- Isolation Forest
- Autoencoder (PCA-based for simplicity)
- Ensemble scoring
- Severity classification

Author: a6on-i Team
Version: 1.1 - Fixed feature alignment issue
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
from dataclasses import dataclass, asdict

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

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
class AnomalyConfig:
    """Configuration for anomaly detection."""
    
    # Contamination rate (expected proportion of outliers)
    contamination: float = 0.05  # 5% expected anomalies
    
    # Isolation Forest parameters
    if_n_estimators: int = 200
    if_max_samples: str = 'auto'
    if_max_features: float = 0.8
    
    # Local Outlier Factor parameters
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.05
    
    # Autoencoder parameters (PCA-based reconstruction)
    ae_n_components: float = 0.95  # Explain 95% variance
    ae_reconstruction_threshold_percentile: float = 95
    
    # Ensemble weights
    weight_isolation_forest: float = 0.4
    weight_lof: float = 0.3
    weight_reconstruction: float = 0.3
    
    # Severity thresholds (percentiles of anomaly scores)
    severity_critical_percentile: float = 99
    severity_high_percentile: float = 95
    severity_medium_percentile: float = 90
    
    # Random state for reproducibility
    random_state: int = 42


# Features to use for anomaly detection (RAW features only - no leakage)
ANOMALY_FEATURE_WHITELIST = [
    # Query metrics
    'total_open_queries',
    'dm_queries',
    'clinical_queries',
    'medical_queries',
    'site_queries',
    'coding_queries',
    'safety_queries',
    
    # CRF metrics
    'total_crfs',
    'crfs_signed',
    'crfs_never_signed',
    'crfs_overdue_for_signs_within_45_days',
    'crfs_overdue_for_signs_beyond_90_days',
    'broken_signatures',
    
    # SDV metrics
    'total_crfs_requiring_sdv',
    'crfs_source_data_verified',
    
    # Completeness
    'visit_missing_visit_count',
    'pages_missing_page_count',
    
    # SAE metrics
    'sae_dm_sae_dm_total',
    'sae_dm_sae_dm_completed',
    'sae_safety_sae_safety_total',
    'sae_safety_sae_safety_completed',
    
    # Other
    'inactivated_inactivated_total',
    'lab_lab_issue_count',
    'edrr_edrr_issue_count',
    
    # Coding
    'meddra_total_terms',
    'meddra_coded_terms',
    'whodrug_total_terms',
    'whodrug_coded_terms',
]


# =============================================================================
# ANOMALY DETECTOR CLASS
# =============================================================================

class AnomalyDetector:
    """
    Multi-method anomaly detection for clinical trial data.
    
    Methods:
    1. Isolation Forest - Tree-based anomaly detection
    2. Local Outlier Factor - Density-based detection
    3. Reconstruction Error - PCA-based autoencoder
    4. Statistical - Z-score based detection
    
    Ensemble combines all methods for robust detection.
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        """Initialize anomaly detector with configuration."""
        self.config = config or AnomalyConfig()
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.thresholds: Dict[str, float] = {}
        self.feature_names: List[str] = []  # Final feature names after all processing
        self.raw_feature_names: List[str] = []  # Raw features before derived
        self.is_fitted: bool = False
        self.fit_stats: Dict[str, Any] = {}
        
    def _get_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract raw features from dataframe."""
        # Get available features from whitelist
        available_features = [f for f in ANOMALY_FEATURE_WHITELIST if f in df.columns]
        
        if len(available_features) < 5:
            raise ValueError(f"Insufficient features. Found only {len(available_features)}")
        
        logger.info(f"Found {len(available_features)} raw features for anomaly detection")
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Fill missing values with median
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X[col] = X[col].fillna(median_val)
        
        # Convert to numeric (safety)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return X
    
    def _create_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for better anomaly detection."""
        X_derived = X.copy()
        
        # Safe ratio function
        def safe_ratio(num_col: str, denom_col: str) -> np.ndarray:
            if num_col in X.columns and denom_col in X.columns:
                num = X[num_col].values
                denom = X[denom_col].values
                return np.where(denom > 0, num / denom, 0)
            return None
        
        # SDV completion ratio
        ratio = safe_ratio('crfs_source_data_verified', 'total_crfs_requiring_sdv')
        if ratio is not None:
            X_derived['derived_sdv_ratio'] = ratio
        
        # Signature ratio
        ratio = safe_ratio('crfs_signed', 'total_crfs')
        if ratio is not None:
            X_derived['derived_signature_ratio'] = ratio
        
        # Query density (queries per CRF)
        ratio = safe_ratio('total_open_queries', 'total_crfs')
        if ratio is not None:
            X_derived['derived_query_density'] = ratio
        
        # SAE DM completion ratio
        ratio = safe_ratio('sae_dm_sae_dm_completed', 'sae_dm_sae_dm_total')
        if ratio is not None:
            X_derived['derived_sae_dm_ratio'] = ratio
        
        # MedDRA coding completion
        ratio = safe_ratio('meddra_coded_terms', 'meddra_total_terms')
        if ratio is not None:
            X_derived['derived_meddra_ratio'] = ratio
        
        return X_derived
    
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepare features for anomaly detection.
        
        Args:
            df: Input dataframe
            is_training: If True, we're fitting. If False, we're predicting.
        """
        logger.info("Preparing features for anomaly detection...")
        
        # Get raw features
        X = self._get_raw_features(df)
        
        # Create derived features
        X = self._create_derived_features(X)
        
        if is_training:
            # During training, store the final feature names
            self.feature_names = list(X.columns)
            logger.info(f"Training with {len(self.feature_names)} features (raw + derived)")
        else:
            # During prediction, ensure we have exactly the same features
            # Add any missing features with zeros
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            
            # Remove any extra features
            extra_cols = set(X.columns) - set(self.feature_names)
            if extra_cols:
                X = X.drop(columns=list(extra_cols))
            
            # Reorder columns to match training order
            X = X[self.feature_names]
            
            logger.info(f"Prediction with {len(X.columns)} features (aligned to training)")
        
        return X
    
    def fit(self, df: pd.DataFrame) -> 'AnomalyDetector':
        """
        Fit all anomaly detection models.
        
        Args:
            df: Patient-level DataFrame with features
            
        Returns:
            self
        """
        logger.info("=" * 60)
        logger.info("FITTING ANOMALY DETECTION MODELS")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Prepare features (training mode)
        X = self._prepare_features(df, is_training=True)
        
        n_samples, n_features = X.shape
        logger.info(f"Training data: {n_samples:,} samples, {n_features} features")
        
        # Scale features
        self.scalers['robust'] = RobustScaler()
        X_scaled = self.scalers['robust'].fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Also fit standard scaler for Z-scores
        self.scalers['standard'] = StandardScaler()
        self.scalers['standard'].fit(X)
        
        # =================================================================
        # 1. ISOLATION FOREST
        # =================================================================
        logger.info("\n1. Training Isolation Forest...")
        
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=self.config.if_n_estimators,
            max_samples=self.config.if_max_samples,
            max_features=self.config.if_max_features,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        self.models['isolation_forest'].fit(X_scaled)
        
        # Get scores (-1 to 1, lower = more anomalous)
        if_scores = self.models['isolation_forest'].decision_function(X_scaled)
        logger.info(f"   IF scores range: [{if_scores.min():.4f}, {if_scores.max():.4f}]")
        
        # =================================================================
        # 2. LOCAL OUTLIER FACTOR
        # =================================================================
        logger.info("\n2. Training Local Outlier Factor...")
        
        # Use smaller n_neighbors if dataset is small
        n_neighbors = min(self.config.lof_n_neighbors, n_samples - 1, 50)
        
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.config.lof_contamination,
            novelty=True,  # Enable predict on new data
            n_jobs=-1
        )
        self.models['lof'].fit(X_scaled)
        
        # Get scores (note: LOF can have extreme values, we'll handle this)
        lof_scores_raw = self.models['lof'].decision_function(X_scaled)
        
        # Clip extreme LOF values (can be very negative for outliers)
        lof_scores = np.clip(lof_scores_raw, np.percentile(lof_scores_raw, 1), np.percentile(lof_scores_raw, 99))
        logger.info(f"   LOF scores range (clipped): [{lof_scores.min():.4f}, {lof_scores.max():.4f}]")
        
        # Store clipping thresholds for prediction
        self.thresholds['lof_clip_min'] = float(np.percentile(lof_scores_raw, 1))
        self.thresholds['lof_clip_max'] = float(np.percentile(lof_scores_raw, 99))
        
        # =================================================================
        # 3. PCA-BASED RECONSTRUCTION (Autoencoder Alternative)
        # =================================================================
        logger.info("\n3. Training PCA Reconstruction Model...")
        
        # Determine number of components (at least 2, at most 80% of features)
        n_components = max(2, min(int(n_features * 0.8), n_samples - 1, n_features - 1))
        
        self.models['pca'] = PCA(n_components=n_components, random_state=self.config.random_state)
        X_pca = self.models['pca'].fit_transform(X_scaled)
        X_reconstructed = self.models['pca'].inverse_transform(X_pca)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Store threshold
        self.thresholds['reconstruction'] = float(np.percentile(
            reconstruction_errors, 
            self.config.ae_reconstruction_threshold_percentile
        ))
        
        logger.info(f"   PCA components: {n_components}")
        logger.info(f"   Explained variance: {self.models['pca'].explained_variance_ratio_.sum():.2%}")
        logger.info(f"   Reconstruction threshold: {self.thresholds['reconstruction']:.6f}")
        
        # =================================================================
        # 4. STATISTICAL THRESHOLDS (Z-SCORES)
        # =================================================================
        logger.info("\n4. Computing Statistical Thresholds...")
        
        # Store means and stds for Z-score calculation
        self.fit_stats['means'] = X.mean().to_dict()
        self.fit_stats['stds'] = X.std().to_dict()
        self.fit_stats['medians'] = X.median().to_dict()
        self.fit_stats['iqr'] = (X.quantile(0.75) - X.quantile(0.25)).to_dict()
        
        # =================================================================
        # SEVERITY THRESHOLDS
        # =================================================================
        logger.info("\n5. Computing Severity Thresholds...")
        
        # Compute ensemble scores for threshold calibration
        ensemble_scores = self._compute_ensemble_scores(
            if_scores, lof_scores, reconstruction_errors
        )
        
        self.thresholds['severity_critical'] = float(np.percentile(
            ensemble_scores, self.config.severity_critical_percentile
        ))
        self.thresholds['severity_high'] = float(np.percentile(
            ensemble_scores, self.config.severity_high_percentile
        ))
        self.thresholds['severity_medium'] = float(np.percentile(
            ensemble_scores, self.config.severity_medium_percentile
        ))
        
        logger.info(f"   Critical threshold (P{self.config.severity_critical_percentile}): "
                   f"{self.thresholds['severity_critical']:.4f}")
        logger.info(f"   High threshold (P{self.config.severity_high_percentile}): "
                   f"{self.thresholds['severity_high']:.4f}")
        logger.info(f"   Medium threshold (P{self.config.severity_medium_percentile}): "
                   f"{self.thresholds['severity_medium']:.4f}")
        
        self.is_fitted = True
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nFitting complete in {duration:.2f} seconds")
        
        return self
    
    def _compute_ensemble_scores(
        self, 
        if_scores: np.ndarray, 
        lof_scores: np.ndarray, 
        reconstruction_errors: np.ndarray
    ) -> np.ndarray:
        """Compute weighted ensemble anomaly scores."""
        
        # Normalize scores to [0, 1] range (higher = more anomalous)
        def normalize_scores(scores: np.ndarray, invert: bool = False) -> np.ndarray:
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s < 1e-10:
                return np.zeros_like(scores)
            normalized = (scores - min_s) / (max_s - min_s)
            return 1 - normalized if invert else normalized
        
        # IF and LOF: lower = more anomalous, so invert
        if_normalized = normalize_scores(if_scores, invert=True)
        lof_normalized = normalize_scores(lof_scores, invert=True)
        
        # Reconstruction error: higher = more anomalous
        recon_normalized = normalize_scores(reconstruction_errors, invert=False)
        
        # Weighted ensemble
        ensemble = (
            self.config.weight_isolation_forest * if_normalized +
            self.config.weight_lof * lof_normalized +
            self.config.weight_reconstruction * recon_normalized
        )
        
        return ensemble
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomaly scores and severity for new data.
        
        Args:
            df: Patient-level DataFrame
            
        Returns:
            DataFrame with anomaly scores and classifications
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        logger.info("Computing anomaly scores...")
        
        # Prepare features (prediction mode - aligns to training features)
        X = self._prepare_features(df, is_training=False)
        
        # Scale using fitted scaler
        X_scaled = self.scalers['robust'].transform(X)
        
        # Get scores from each model
        if_scores = self.models['isolation_forest'].decision_function(X_scaled)
        
        lof_scores_raw = self.models['lof'].decision_function(X_scaled)
        lof_scores = np.clip(
            lof_scores_raw, 
            self.thresholds['lof_clip_min'], 
            self.thresholds['lof_clip_max']
        )
        
        # Reconstruction error
        X_pca = self.models['pca'].transform(X_scaled)
        X_reconstructed = self.models['pca'].inverse_transform(X_pca)
        reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Compute ensemble scores
        ensemble_scores = self._compute_ensemble_scores(
            if_scores, lof_scores, reconstruction_errors
        )
        
        # Compute Z-scores for each feature
        z_scores = self._compute_z_scores(X)
        max_z_scores = np.abs(z_scores).max(axis=1)
        
        # Classify severity
        severity = self._classify_severity(ensemble_scores)
        
        # Build results DataFrame
        results = pd.DataFrame({
            'patient_key': df['patient_key'].values if 'patient_key' in df.columns else df.index,
            'study_id': df['study_id'].values if 'study_id' in df.columns else 'Unknown',
            'site_id': df['site_id'].values if 'site_id' in df.columns else 'Unknown',
            'subject_id': df['subject_id'].values if 'subject_id' in df.columns else 'Unknown',
            
            # Scores
            'anomaly_score_ensemble': ensemble_scores,
            'anomaly_score_if': -if_scores,  # Invert so higher = more anomalous
            'anomaly_score_lof': -lof_scores,
            'anomaly_score_reconstruction': reconstruction_errors,
            'anomaly_score_zscore': max_z_scores,
            
            # Classifications
            'is_anomaly': ensemble_scores >= self.thresholds['severity_medium'],
            'anomaly_severity': severity,
            'anomaly_percentile': stats.rankdata(ensemble_scores) / len(ensemble_scores) * 100,
        })
        
        # Add top contributing features
        top_features = self._get_top_anomaly_features(X, z_scores)
        results['top_anomaly_features'] = top_features
        
        logger.info(f"Anomalies detected: {results['is_anomaly'].sum():,} "
                   f"({results['is_anomaly'].mean():.1%})")
        
        return results
    
    def _compute_z_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Compute Z-scores for all features."""
        z_scores = np.zeros_like(X.values, dtype=float)
        
        for i, col in enumerate(X.columns):
            mean = self.fit_stats['means'].get(col, 0)
            std = self.fit_stats['stds'].get(col, 1)
            if std == 0 or pd.isna(std):
                std = 1
            z_scores[:, i] = (X[col].values - mean) / std
        
        return z_scores
    
    def _classify_severity(self, scores: np.ndarray) -> List[str]:
        """Classify anomaly severity based on ensemble scores."""
        severity = []
        for score in scores:
            if score >= self.thresholds['severity_critical']:
                severity.append('Critical')
            elif score >= self.thresholds['severity_high']:
                severity.append('High')
            elif score >= self.thresholds['severity_medium']:
                severity.append('Medium')
            else:
                severity.append('Normal')
        return severity
    
    def _get_top_anomaly_features(
        self, 
        X: pd.DataFrame, 
        z_scores: np.ndarray, 
        top_n: int = 3
    ) -> List[str]:
        """Get top contributing features for each anomaly."""
        top_features = []
        columns = list(X.columns)
        
        for i in range(len(X)):
            # Get absolute Z-scores for this sample
            abs_z = np.abs(z_scores[i])
            
            # Get top N feature indices
            top_idx = np.argsort(abs_z)[-top_n:][::-1]
            
            # Build feature string
            feature_strs = []
            for idx in top_idx:
                if abs_z[idx] > 2:  # Only include if Z > 2
                    feature_strs.append(f"{columns[idx]}(z={z_scores[i, idx]:.1f})")
            
            top_features.append('; '.join(feature_strs) if feature_strs else 'None')
        
        return top_features
    
    def aggregate_to_site(self, patient_results: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate patient-level anomalies to site level.
        
        Args:
            patient_results: Output from predict()
            
        Returns:
            Site-level anomaly summary
        """
        logger.info("Aggregating anomalies to site level...")
        
        # Group by study and site
        site_agg = patient_results.groupby(['study_id', 'site_id']).agg({
            'patient_key': 'count',
            'anomaly_score_ensemble': ['mean', 'max', 'std'],
            'is_anomaly': 'sum',
            'anomaly_percentile': 'mean',
        }).reset_index()
        
        # Flatten column names
        site_agg.columns = [
            'study_id', 'site_id', 'patient_count',
            'avg_anomaly_score', 'max_anomaly_score', 'std_anomaly_score',
            'anomaly_count', 'avg_anomaly_percentile'
        ]
        
        # Fill NaN std with 0
        site_agg['std_anomaly_score'] = site_agg['std_anomaly_score'].fillna(0)
        
        # Calculate anomaly rate
        site_agg['anomaly_rate'] = site_agg['anomaly_count'] / site_agg['patient_count']
        
        # Classify site severity
        def classify_site(row):
            if row['anomaly_rate'] > 0.2 or row['max_anomaly_score'] > 0.9:
                return 'Critical'
            elif row['anomaly_rate'] > 0.1 or row['max_anomaly_score'] > 0.7:
                return 'High'
            elif row['anomaly_rate'] > 0.05:
                return 'Medium'
            else:
                return 'Normal'
        
        site_agg['site_severity'] = site_agg.apply(classify_site, axis=1)
        
        # Count severity by site
        severity_pivot = patient_results.pivot_table(
            index=['study_id', 'site_id'],
            columns='anomaly_severity',
            values='patient_key',
            aggfunc='count',
            fill_value=0
        ).reset_index()
        
        # Rename columns
        severity_cols = {'Critical': 'critical_count', 'High': 'high_count', 
                        'Medium': 'medium_count', 'Normal': 'normal_count'}
        for old_name, new_name in severity_cols.items():
            if old_name in severity_pivot.columns:
                severity_pivot = severity_pivot.rename(columns={old_name: new_name})
            else:
                severity_pivot[new_name] = 0
        
        # Keep only the columns we need
        severity_cols_to_keep = ['study_id', 'site_id', 'critical_count', 'high_count', 
                                  'medium_count', 'normal_count']
        severity_pivot = severity_pivot[[c for c in severity_cols_to_keep if c in severity_pivot.columns]]
        
        # Merge
        site_agg = site_agg.merge(severity_pivot, on=['study_id', 'site_id'], how='left')
        
        # Fill NaN counts
        for col in ['critical_count', 'high_count', 'medium_count', 'normal_count']:
            if col in site_agg.columns:
                site_agg[col] = site_agg[col].fillna(0).astype(int)
            else:
                site_agg[col] = 0
        
        logger.info(f"Site-level aggregation complete: {len(site_agg):,} sites")
        
        return site_agg
    
    def save(self, output_dir: Path):
        """Save all models and artifacts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        with open(output_dir / 'anomaly_detector_models.pkl', 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'thresholds': self.thresholds,
                'feature_names': self.feature_names,
                'config': asdict(self.config),
                'fit_stats': self.fit_stats,
            }, f)
        
        logger.info(f"Models saved to {output_dir}")
    
    def load(self, model_path: Path) -> 'AnomalyDetector':
        """Load saved models."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data['models']
        self.scalers = data['scalers']
        self.thresholds = data['thresholds']
        self.feature_names = data['feature_names']
        self.config = AnomalyConfig(**data['config'])
        self.fit_stats = data['fit_stats']
        self.is_fitted = True
        
        logger.info(f"Models loaded from {model_path}")
        return self


# =============================================================================
# MAIN RUNNER FUNCTION
# =============================================================================

def run_anomaly_detection(
    upr_path: Path,
    output_dir: Path,
    config: Optional[AnomalyConfig] = None
) -> Dict[str, Any]:
    """
    Run complete anomaly detection pipeline.
    
    Args:
        upr_path: Path to unified patient record parquet
        output_dir: Output directory for results
        config: Optional configuration
        
    Returns:
        Summary dictionary
    """
    logger.info("=" * 70)
    logger.info("a6on-i - ANOMALY DETECTION ENGINE v1.1")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models' / 'anomaly_detector'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"\nLoading data from {upr_path}...")
    df = pd.read_parquet(upr_path)
    logger.info(f"Loaded {len(df):,} patients")
    
    # Initialize detector
    detector = AnomalyDetector(config=config or AnomalyConfig())
    
    # Fit models
    detector.fit(df)
    
    # Predict anomalies
    patient_results = detector.predict(df)
    
    # Aggregate to site level
    site_results = detector.aggregate_to_site(patient_results)
    
    # =================================================================
    # SAVE RESULTS
    # =================================================================
    logger.info("\nSaving results...")
    
    # Save patient-level results to PostgreSQL
    from src.database.pg_writer import safe_to_postgres
    success, msg = safe_to_postgres(patient_results, 'patient_anomalies', if_exists='replace')
    if success:
        logger.info(f"✅ Saved patient_anomalies to PostgreSQL: {len(patient_results):,} rows")
    else:
        logger.warning(f"PostgreSQL write failed: {msg}")
    
    # Save site-level results to PostgreSQL
    success, msg = safe_to_postgres(site_results, 'site_anomalies', if_exists='replace')
    if success:
        logger.info(f"✅ Saved site_anomalies to PostgreSQL: {len(site_results):,} rows")
    else:
        logger.warning(f"PostgreSQL write failed: {msg}")
    
    site_results.to_csv(output_dir / 'site_anomalies.csv', index=False)

    
    # Save high-severity anomalies
    high_severity = patient_results[
        patient_results['anomaly_severity'].isin(['Critical', 'High'])
    ].sort_values('anomaly_score_ensemble', ascending=False)
    high_severity.to_csv(output_dir / 'high_severity_anomalies.csv', index=False)
    
    # Save models
    detector.save(models_dir)
    
    # =================================================================
    # GENERATE SUMMARY
    # =================================================================
    severity_dist = patient_results['anomaly_severity'].value_counts().to_dict()
    
    summary = {
        'run_timestamp': datetime.now().isoformat(),
        'version': '1.1',
        'input': {
            'source': str(upr_path),
            'total_patients': int(len(df)),
        },
        'features': {
            'count': len(detector.feature_names),
            'names': detector.feature_names,
        },
        'anomaly_detection': {
            'total_anomalies': int(patient_results['is_anomaly'].sum()),
            'anomaly_rate': float(patient_results['is_anomaly'].mean()),
            'severity_distribution': {k: int(v) for k, v in severity_dist.items()},
            'score_stats': {
                'mean': float(patient_results['anomaly_score_ensemble'].mean()),
                'std': float(patient_results['anomaly_score_ensemble'].std()),
                'min': float(patient_results['anomaly_score_ensemble'].min()),
                'max': float(patient_results['anomaly_score_ensemble'].max()),
            }
        },
        'site_level': {
            'total_sites': int(len(site_results)),
            'sites_with_anomalies': int((site_results['anomaly_count'] > 0).sum()),
            'critical_sites': int((site_results['site_severity'] == 'Critical').sum()),
            'high_sites': int((site_results['site_severity'] == 'High').sum()),
        },
        'thresholds': {
            'critical': float(detector.thresholds['severity_critical']),
            'high': float(detector.thresholds['severity_high']),
            'medium': float(detector.thresholds['severity_medium']),
        },
        'models': {
            'isolation_forest': 'fitted',
            'local_outlier_factor': 'fitted',
            'pca_reconstruction': 'fitted',
        },
        'output_files': [
            'patient_anomalies.parquet',
            'site_anomalies.parquet',
            'site_anomalies.csv',
            'high_severity_anomalies.csv',
            'models/anomaly_detector/anomaly_detector_models.pkl',
        ],
        'duration_seconds': float((datetime.now() - start_time).total_seconds())
    }
    
    # Save summary
    with open(output_dir / 'anomaly_detection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # =================================================================
    # PRINT FINAL SUMMARY
    # =================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANOMALY DETECTION COMPLETE")
    logger.info("=" * 70)
    
    logger.info(f"\nPatient-Level Results:")
    logger.info(f"  Total Patients:    {len(df):,}")
    logger.info(f"  Total Anomalies:   {summary['anomaly_detection']['total_anomalies']:,} "
               f"({summary['anomaly_detection']['anomaly_rate']:.1%})")
    
    logger.info(f"\nSeverity Distribution:")
    for sev in ['Critical', 'High', 'Medium', 'Normal']:
        count = severity_dist.get(sev, 0)
        pct = count / len(df) * 100
        logger.info(f"    {sev}: {count:,} ({pct:.1f}%)")
    
    logger.info(f"\nSite-Level Results:")
    logger.info(f"  Total Sites:       {summary['site_level']['total_sites']:,}")
    logger.info(f"  Sites w/ Anomalies: {summary['site_level']['sites_with_anomalies']:,}")
    logger.info(f"  Critical Sites:    {summary['site_level']['critical_sites']:,}")
    logger.info(f"  High Sites:        {summary['site_level']['high_sites']:,}")
    
    logger.info(f"\nDuration: {summary['duration_seconds']:.2f} seconds")
    logger.info(f"Output: {output_dir}")
    
    return summary


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from config.settings import DATA_PROCESSED

# PostgreSQL data access
try:
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

    
    upr_path = DATA_PROCESSED / 'upr' / 'unified_patient_record.parquet'
    output_dir = DATA_PROCESSED / 'analytics'
    
    run_anomaly_detection(upr_path, output_dir)
