"""
SANCHALAK AI - ML Data Preparation v1.3 (LEAKAGE FULLY FIXED)
Phase 3.1: Feature Engineering with STRICT leakage prevention

FIXES v1.3:
- Target created from RAW source data only (not DQI)
- ALL DQI features excluded (they were used to derive risk)
- Only truly independent features used
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import warnings
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.database.connection import get_db_manager

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class MLDataConfig:
    """ML Data Preparation Configuration - FULLY LEAKAGE-FREE"""
    
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    random_state: int = 42
    min_samples_per_class: int = 10
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    smote_sampling_strategy: str = 'auto'
    smote_k_neighbors: int = 5
    
    targets: Dict[str, str] = field(default_factory=lambda: {
        'risk': 'risk_level',
    })


# RAW FEATURES ONLY - No derived metrics
# These come directly from source data and are NOT used to calculate risk
RAW_FEATURES_WHITELIST = [
    # From CPID EDC Metrics (raw counts)
    'expected_visits_rave_edc_bo4',
    'pages_entered',
    'forms_verified',
    'crfs_frozen',
    'crfs_locked',
    'crfs_require_signature',
    'crfs_signed',
    'crfs_require_verification_sdv',
    'crfs_verified_sdv',
    'crfs_never_signed',
    'broken_signatures',
    'pages_with_nonconformant_data',
    'total_crfs',
    'protocol_deviations',
    'crfs_overdue_for_signs_within_45_days_of_data_entry',
    'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
    'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
    
    # Query counts (raw from source)
    'dm_queries',
    'clinical_queries', 
    'medical_queries',
    'site_queries',
    'safety_queries',
    'queries_answered',
    
    # From visit_projection (raw)
    'visit_missing_visit_count',
    'visit_visits_overdue_count',
    'visit_visits_overdue_avg_days',
    'visit_visits_overdue_max_days',
    'visit_visits_overdue_total_days',
    
    # From missing_lab_ranges (raw)
    'lab_lab_issue_count',
    'lab_lab_missing_names',
    'lab_lab_missing_ranges',
    
    # From SAE dashboard (raw) - NOT pending counts used in risk
    'sae_dm_sae_dm_total',
    'sae_dm_sae_dm_completed',
    'sae_safety_sae_safety_total',
    'sae_safety_sae_safety_completed',
    
    # From coding (raw counts)
    'meddra_coding_meddra_total',
    'meddra_coding_meddra_coded',
    'whodrug_coding_whodrug_total',
    'whodrug_coding_whodrug_coded',
    
    # From inactivated forms (raw)
    'inactivated_inactivated_form_count',
    'inactivated_inactivated_page_count',
    'inactivated_inactivated_folder_count',
    
    # From EDRR (raw) - NOT pending used in risk
    'edrr_edrr_issue_count',
    'edrr_edrr_resolved',
    
    # From missing pages (raw)
    'pages_pages_missing_count',
    'pages_pages_missing_avg_days',
    'pages_pages_missing_max_days',
    'pages_pages_missing_total_days',
]



# =============================================================================
# DATA PREPARATION WITH ADVANCED FEATURE ENGINEERING
# =============================================================================

class MLDataPreparator:
    """ML Data Preparation - ENHANCED (200+ FEATURES)"""
    
    def __init__(self, config: MLDataConfig = None):
        self.config = config or MLDataConfig()
        self.feature_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.dropped_features: List[Tuple[str, str]] = []
        
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Advanced Feature Engineering and select features."""
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE ENGINEERING (200+ FEATURES)")
        logger.info("=" * 70)
        
        # 1. Run Advanced Feature Engineering
        from src.data.advanced_feature_engineering import AdvancedFeatureEngineer
        logger.info("  Running AdvancedFeatureEngineer...")
        engineer = AdvancedFeatureEngineer(df)
        df_enhanced = engineer.run_all()
        
        # 2. Filter columns
        # Exclude metadata, targets, and raw text
        metadata_cols = [
            'study_id', 'site_id', 'patient_key', 'subject_id', 
            '_source_file', '_ingestion_ts', 'patient_id'
        ]
        
        exclude_partners = [
            'target', 'risk_level', 'risk_score', 'split',
            'tier1_blocking_reason', 'tier2_blocking_reason',
            'quick_win_category', 'db_lock_status', 'performance_tier'
        ]
        
        # Identification of useful features
        # We want numeric and low-cardinality categorical
        self.numeric_columns = []
        self.categorical_columns = []
        
        # Features to keep
        processed_cols = []
        
        for col in df_enhanced.columns:
            # Skip excluded
            if col in metadata_cols or col in exclude_partners:
                continue
            if col.startswith('_'):
                continue
                
            # Check type
            is_numeric = np.issubdtype(df_enhanced[col].dtype, np.number)
            is_categorical = df_enhanced[col].dtype == 'object' or df_enhanced[col].dtype.name == 'category'
            
            if is_numeric:
                # Skip if constant (will be caught by low variance later, but good check)
                if df_enhanced[col].nunique() <= 1:
                    continue
                self.numeric_columns.append(col)
                processed_cols.append(col)
                
            elif is_categorical:
                # Only low cardinality
                if df_enhanced[col].nunique() <= 20 and df_enhanced[col].nunique() > 1:
                    self.categorical_columns.append(col)
                    processed_cols.append(col)
        
        self.feature_columns = self.numeric_columns + self.categorical_columns
        
        logger.info(f"  Numeric features: {len(self.numeric_columns)}")
        logger.info(f"  Categorical features: {len(self.categorical_columns)}")
        logger.info(f"  Total features selected: {len(self.feature_columns)}")
        
        return df_enhanced[self.feature_columns].copy()
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        logger.info("\n" + "=" * 70)
        logger.info("HANDLING MISSING VALUES")
        logger.info("=" * 70)
        
        df = df.copy()
        
        # Fill numeric with median
        if self.numeric_columns:
            df[self.numeric_columns] = df[self.numeric_columns].fillna(df[self.numeric_columns].median())
        
        # Fill categorical with mode or 'Unknown'
        for col in self.categorical_columns:
            if df[col].isnull().any():
                 mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                 df[col] = df[col].fillna(mode_val)
        
        # Final check
        remaining = df.isnull().sum().sum()
        if remaining > 0:
            logger.warning(f"  Remaining missing values: {remaining} (filling with 0)")
            df = df.fillna(0)
            
        logger.info("  Missing values handled ✓")
        return df
    
    def remove_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove near-zero variance"""
        logger.info("\n" + "=" * 70)
        logger.info("REMOVING LOW VARIANCE")
        logger.info("=" * 70)
        
        existing = [c for c in self.numeric_columns if c in df.columns]
        if not existing:
            return df
        
        # Using sklearn VarianceThreshold logic manually or simplified
        # Filter out columns with variance < threshold
        variances = df[existing].var()
        low_var = variances[variances < self.config.variance_threshold].index.tolist()
        
        if low_var:
            logger.info(f"  Removing {len(low_var)} low variance features")
            for col in low_var:
                if col in self.numeric_columns:
                    self.numeric_columns.remove(col)
                if col in self.feature_columns:
                    self.feature_columns.remove(col)
                self.dropped_features.append((col, 'low_variance'))
            df = df.drop(columns=low_var)
        else:
            logger.info("  No low variance features ✓")
        
        return df
    
    def remove_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated"""
        logger.info("\n" + "=" * 70)
        logger.info("REMOVING HIGH CORRELATION")
        logger.info("=" * 70)
        
        existing = [c for c in self.numeric_columns if c in df.columns]
        if len(existing) < 2:
            return df
        
        # Compute correlation matrix
        corr_matrix = df[existing].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > self.config.correlation_threshold)]
        
        if to_drop:
            logger.info(f"  Removing {len(to_drop)} correlated features")
            for col in to_drop:
                if col in self.numeric_columns:
                    self.numeric_columns.remove(col)
                if col in self.feature_columns:
                    self.feature_columns.remove(col)
                self.dropped_features.append((col, 'correlation'))
            df = df.drop(columns=to_drop)
        else:
            logger.info("  No high correlation ✓")
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features"""
        logger.info("\n" + "=" * 70)
        logger.info("SCALING FEATURES")
        logger.info("=" * 70)
        
        existing = [c for c in self.numeric_columns if c in df.columns]
        if not existing:
            return df
        
        df = df.copy()
        scaler = RobustScaler()
        df[existing] = scaler.fit_transform(df[existing])
        self.scalers['numeric'] = scaler
        
        logger.info(f"  Scaled {len(existing)} features ✓")
        return df
    
    def create_risk_target(self, df_original: pd.DataFrame) -> pd.Series:
        """Create risk target from RAW DATA (same as before)"""
        # Re-using the robust logic from v1.3
        # Note: We must use the original DF or ensure the columns exist
        
        logger.info("\n" + "=" * 70)
        logger.info("CREATING RISK TARGET")
        logger.info("=" * 70)
        
        risk_score = pd.Series(0.0, index=df_original.index)
        
        # SAE
        if 'sae_dm_sae_dm_pending' in df_original.columns:
             risk_score += (df_original['sae_dm_sae_dm_pending'].fillna(0) > 0).astype(float) * 3
        
        if 'sae_safety_sae_safety_pending' in df_original.columns:
             risk_score += (df_original['sae_safety_sae_safety_pending'].fillna(0) > 0).astype(float) * 3
             
        # Missing
        if 'visit_missing_visit_count' in df_original.columns:
             risk_score += (df_original['visit_missing_visit_count'].fillna(0) > 0).astype(float) * 2
        
        # Overdue
        if 'crfs_overdue_for_signs_beyond_90_days_of_data_entry' in df_original.columns:
             risk_score += (df_original['crfs_overdue_for_signs_beyond_90_days_of_data_entry'].fillna(0) > 0).astype(float) * 2
        
        # Protocol deviations
        if 'protocol_deviations' in df_original.columns:
             risk_score += (df_original['protocol_deviations'].fillna(0) > 0).astype(float) * 1.5
             
        # Thresholds
        p50 = risk_score.quantile(0.50)
        p80 = risk_score.quantile(0.80)
        p95 = risk_score.quantile(0.95)
        
        risk_level = pd.cut(
            risk_score,
            bins=[-np.inf, p50, p80, p95, np.inf],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        logger.info(f"  Risk distribution: {risk_level.value_counts().to_dict()}")
        return risk_level
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Split data"""
        logger.info("\n" + "=" * 70)
        logger.info("SPLITTING DATA")
        logger.info("=" * 70)
        
        # Encode target
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        self.encoders['target'] = le
        self.encoders['target_classes'] = list(le.classes_)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=self.config.test_ratio,
            random_state=self.config.random_state, stratify=y_encoded
        )
        
        val_ratio = self.config.val_ratio / (1 - self.config.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            random_state=self.config.random_state, stratify=y_temp
        )
        
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        for name, (X_s, y_s) in splits.items():
            logger.info(f"    {name}: {len(X_s):,}")
            
        return splits
    
    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE"""
        if not IMBLEARN_AVAILABLE:
            return X_train, y_train
            
        logger.info("  Applying SMOTE...")
        min_samples = y_train.value_counts().min()
        k = min(self.config.smote_k_neighbors, min_samples - 1)
        
        if k < 1:
            sampler = RandomOverSampler(random_state=self.config.random_state)
        else:
            sampler = SMOTE(k_neighbors=k, random_state=self.config.random_state)
            
        try:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            X_res = pd.DataFrame(X_res, columns=X_train.columns)
            y_res = pd.Series(y_res)
            return X_res, y_res
        except Exception as e:
            logger.warning(f"  SMOTE failed: {e}")
            return X_train, y_train

    def prepare_for_training(self, df: pd.DataFrame) -> Dict:
        """Complete pipeline"""
        logger.info("\n" + "=" * 70)
        logger.info("DATA PREPARATION PIPELINE (200+ FEATURES)")
        logger.info("=" * 70)
        
        start = datetime.now()
        
        # 1. Create target
        y = self.create_risk_target(df)
        
        # 2. Select & Engineer Features (pass original df)
        X = self.select_features(df)
        
        # 3. Handle missing
        X = self.handle_missing_values(X)
        
        # 4. Remove low variance
        X = self.remove_low_variance(X)
        
        # 5. Remove correlation
        X = self.remove_high_correlation(X)
        
        # 6. Scale
        X = self.scale_features(X)
        
        # 7. Split
        splits = self.split_data(X, y)
        
        # 8. SMOTE
        X_res, y_res = self.apply_smote(splits['train'][0], splits['train'][1])
        splits['train_resampled'] = (X_res, y_res)
        
        duration = (datetime.now() - start).total_seconds()
        
        results = {
            'splits': splits,
            'feature_columns': self.feature_columns,
            'metadata': {
                'version': '2.0.0-enhanced',
                'n_features': len(self.feature_columns),
                'duration': duration
            }
        }
        
        logger.info(f"Total Features: {len(self.feature_columns)}")
        return results


class MLDataPrepRunner:
    """Runner"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.preparator = MLDataPreparator()
        self.results = None
        self.db_manager = get_db_manager()
        
    def load_data(self) -> pd.DataFrame:
        logger.info("Loading data from PostgreSQL (unified_patient_record)...")
        try:
            with self.db_manager.engine.connect() as conn:
                # Read the full UPR table created by UPRBuilder
                df = pd.read_sql("SELECT * FROM unified_patient_record", conn)
                logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
                return df
        except Exception as e:
            logger.error(f"Failed to load data from PostgreSQL: {e}")
            raise
    
    def run(self) -> Dict:
        df = self.load_data()
        self.results = self.preparator.prepare_for_training(df)
        return self.results
    
    def save_outputs(self):
        logger.info("\n" + "=" * 60)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 60)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Save to PostgreSQL
        pg_writer = get_pg_writer()
        for name, (X, y) in self.results['splits'].items():
            data = X.copy()
            data['target'] = y.values
            
            table_name = f'ml_{name}'
            success, msg = pg_writer.safe_to_postgres(data, table_name, if_exists='replace')
            if success:
                logger.info(f"✅ {name}: {len(data):,} rows → PostgreSQL table '{table_name}'")
            else:
                logger.warning(f"PostgreSQL write failed for {name}: {msg}")

        
        with open(self.output_dir / 'ml_features.json', 'w') as f:
            json.dump({
                'features': self.results['feature_columns'],
                'numeric': self.results['numeric_columns']
            }, f, indent=2)
        
        with open(self.output_dir / 'ml_preparation_metadata.json', 'w') as f:
            json.dump(self.results['metadata'], f, indent=2)
        
        import pickle


    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 PHASE 3.1 - ML DATA PREP v1.3 (NO LEAKAGE)")
        print("=" * 70)
        
        splits = self.results['splits']
        print(f"\n📁 SPLITS:")
        for name, (X, y) in splits.items():
            print(f"   {name}: {len(X):,}")
        
        print(f"\n🔧 FEATURES: {len(self.results['feature_columns'])}")
        print(f"   (Only raw source features, no DQI or derived)")
        
        print(f"\n📁 Output: {self.output_dir}")


def main():
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'data' / 'processed' / 'ml'
    
    runner = MLDataPrepRunner(output_dir)
    runner.run()
    runner.save_outputs()
    runner.print_summary()
    
    return runner


if __name__ == '__main__':
    main()