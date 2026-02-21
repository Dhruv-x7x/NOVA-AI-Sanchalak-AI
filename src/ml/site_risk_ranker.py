"""
SANCHALAK AI - Site Risk Ranker v1.0
==============================================
Learning-to-Rank model for site operational risk prioritization.

This is a TRIAGE TOOL, not a verdict engine.
Labels are NOISY PROXIES, not ground truth.

Author: Sanchalak AI Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import logging
import warnings
from itertools import combinations
import random

from sklearn.model_selection import GroupKFold
from scipy.stats import kendalltau, spearmanr

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not installed - Site Risk Ranker requires XGBoost")

# PostgreSQL data access
try:
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

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
class SiteRankerConfig:
    """Configuration for Site Risk Ranker"""
    
    random_state: int = 42
    
    # Pair sampling to prevent explosion (2500 sites = ~3M pairs!)
    max_pairs_per_site: int = 75  # Limit pairs per site
    prefer_within_study: bool = True  # Prefer same-study comparisons
    include_hard_negatives: bool = True  # Include similar-score pairs
    
    # XGBoost Ranker params
    xgb_params: Dict = field(default_factory=lambda: {
        'objective': 'rank:pairwise',
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    })
    
    # Pairwise labeling rule weights
    rule_weights: Dict[str, float] = field(default_factory=lambda: {
        'issue_density': 3.0,
        'dqi_score': 2.0,
        'concurrent_issues': 2.0,
        'signature_backlog': 2.0,
        'safety_sensitivity': 3.0
    })
    
    # Red flag thresholds
    max_acceptable_ndcg: float = 0.95  # > this = leakage
    min_acceptable_ndcg: float = 0.55  # < this = not learning
    max_acceptable_tau: float = 0.95   # ≈ 1.0 = identity ranking
    max_feature_importance_pct: float = 0.40  # Single feature dominance
    min_top10_bootstrap_overlap: float = 0.70  # Rank stability
    
    # Evaluation K values
    ndcg_k_values: List[int] = field(default_factory=lambda: [5, 10, 20])


# =============================================================================
# FEATURE DEFINITIONS (STRICT WHITELIST)
# =============================================================================

ALLOWED_FEATURES = {
    # Query burden (aggregate from patient level)
    'query_burden': [
        'dm_queries', 'clinical_queries', 'medical_queries', 
        'safety_queries', 'site_queries', 'queries_answered'
    ],
    
    # SDV completion
    'sdv_completion': [
        'crfs_require_verification_sdv', 'crfs_verified_sdv'
    ],
    
    # Signature delays
    'signature_delays': [
        'crfs_overdue_for_signs_within_45_days_of_data_entry',
        'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
        'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
        'broken_signatures', 'crfs_never_signed', 'crfs_signed'
    ],
    
    # Issue prevalence
    'issue_prevalence': [
        'edrr_edrr_issue_count', 'lab_lab_issue_count',
        'inactivated_inactivated_form_count', 'protocol_deviations'
    ],
    
    # Completeness
    'completeness': [
        'visit_missing_visit_count', 'pages_pages_missing_count',
        'pages_entered', 'total_crfs', 'crfs_frozen', 'crfs_locked'
    ],
    
    # Coding
    'coding': [
        'meddra_coding_meddra_total', 'meddra_coding_meddra_coded',
        'whodrug_coding_whodrug_total', 'whodrug_coding_whodrug_coded'
    ],
    
    # SAE workload
    'sae_workload': [
        'sae_dm_sae_dm_pending', 'sae_dm_sae_dm_total',
        'sae_safety_sae_safety_pending', 'sae_safety_sae_safety_total'
    ]
}

# FORBIDDEN FEATURES (will be actively removed)
FORBIDDEN_FEATURES = [
    'site_rank', 'rank', 'site_performance_index', 'performance_tier',
    'performance_index', 'risk_tier', 'risk_level', 'risk_score',
    'escalation', 'escalated', 'cra_flag', 'audit_finding',
    'dqi_band', 'quick_win_category'
]


# =============================================================================
# SITE AGGREGATOR
# =============================================================================

class SiteAggregator:
    """Aggregates patient-level UPR data to site level"""
    
    def __init__(self):
        self.site_features: List[str] = []
        self.aggregation_stats: Dict = {}
    
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate patient-level data to site level.
        
        Creates:
        - Mean, sum, max for continuous metrics
        - Rates (per patient)
        - Volatility (std across patients) as stability proxy
        """
        logger.info("=" * 70)
        logger.info("AGGREGATING PATIENT DATA TO SITE LEVEL")
        logger.info("=" * 70)
        
        # Identify site grouping columns
        group_cols = ['study_id', 'site_id']
        for col in group_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        logger.info(f"  Input: {len(df):,} patients")
        
        # Find available features from whitelist
        available_features = []
        for category, features in ALLOWED_FEATURES.items():
            for feature in features:
                if feature in df.columns:
                    available_features.append(feature)
        
        logger.info(f"  Available whitelisted features: {len(available_features)}")
        
        # Build aggregation dictionary
        agg_dict = {}
        
        for feature in available_features:
            if df[feature].dtype in ['float64', 'int64', 'float32', 'int32']:
                agg_dict[feature] = ['sum', 'mean', 'max', 'std']
        
        # Always count patients
        agg_dict['patient_key'] = 'count' if 'patient_key' in df.columns else 'size'
        
        # Perform aggregation
        site_df = df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Flatten column names
        new_cols = []
        for col in site_df.columns:
            if isinstance(col, tuple):
                if col[1] == '' or col[1] == 'count' or col[1] == 'size':
                    new_cols.append(col[0] if col[0] != 'patient_key' else 'patient_count')
                else:
                    new_cols.append(f"{col[0]}_{col[1]}")
            else:
                new_cols.append(col)
        site_df.columns = new_cols
        
        # Rename patient count
        if 'patient_key' in site_df.columns:
            site_df = site_df.rename(columns={'patient_key': 'patient_count'})
        
        # Calculate derived rates
        site_df = self._calculate_rates(site_df)
        
        # Remove any forbidden features that snuck through
        site_df = self._remove_forbidden(site_df)
        
        # Store feature list
        self.site_features = [c for c in site_df.columns 
                              if c not in group_cols + ['patient_count']]
        
        logger.info(f"  Output: {len(site_df):,} sites")
        logger.info(f"  Features: {len(self.site_features)}")
        
        self.aggregation_stats = {
            'n_patients': len(df),
            'n_sites': len(site_df),
            'n_features': len(self.site_features),
            'features': self.site_features
        }
        
        return site_df
    
    def _calculate_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-patient rates and normalized metrics"""
        
        df = df.copy()
        patient_count = df['patient_count'].replace(0, 1)  # Avoid division by zero
        
        # Issue density (total issues per patient)
        issue_cols = [c for c in df.columns if 'issue' in c.lower() and '_sum' in c]
        if issue_cols:
            df['issue_density'] = df[issue_cols].sum(axis=1) / patient_count
        
        # Query density
        query_cols = [c for c in df.columns if 'queries' in c.lower() and '_sum' in c]
        if query_cols:
            df['query_density'] = df[query_cols].sum(axis=1) / patient_count
        
        # SDV completion rate
        if 'crfs_verified_sdv_sum' in df.columns and 'crfs_require_verification_sdv_sum' in df.columns:
            required = df['crfs_require_verification_sdv_sum'].replace(0, 1)
            df['sdv_completion_rate'] = (df['crfs_verified_sdv_sum'] / required * 100).clip(0, 100)
        
        # Signature backlog rate
        overdue_cols = [c for c in df.columns if 'overdue' in c.lower() and 'sign' in c.lower() and '_sum' in c]
        if overdue_cols and 'crfs_signed_sum' in df.columns:
            total_sigs = df['crfs_signed_sum'].replace(0, 1)
            df['signature_backlog_rate'] = df[overdue_cols].sum(axis=1) / total_sigs * 100
        
        # SAE pending flag
        sae_cols = [c for c in df.columns if 'sae' in c.lower() and 'pending' in c.lower() and '_sum' in c]
        if sae_cols:
            df['has_sae_pending'] = (df[sae_cols].sum(axis=1) > 0).astype(int)
        
        # Coding completion rate
        if 'meddra_coding_meddra_coded_sum' in df.columns and 'meddra_coding_meddra_total_sum' in df.columns:
            total = df['meddra_coding_meddra_total_sum'].replace(0, 1)
            df['meddra_completion_rate'] = (df['meddra_coding_meddra_coded_sum'] / total * 100).clip(0, 100)
        
        # Volatility metrics (std already computed, just rename for clarity)
        for col in df.columns:
            if col.endswith('_std'):
                df[col.replace('_std', '_volatility')] = df[col]
        
        # Log-scale patient count
        df['log_patient_count'] = np.log1p(df['patient_count'])
        
        return df
    
    def _remove_forbidden(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove any forbidden features"""
        to_remove = []
        for col in df.columns:
            col_lower = col.lower()
            for forbidden in FORBIDDEN_FEATURES:
                if forbidden.lower() in col_lower:
                    to_remove.append(col)
                    break
        
        if to_remove:
            logger.warning(f"  Removing {len(to_remove)} forbidden features: {to_remove}")
            df = df.drop(columns=to_remove, errors='ignore')
        
        return df


# =============================================================================
# BASELINE RANKER (Required for comparison)
# =============================================================================

class BaselineRanker:
    """
    Simple rule-based ranker for comparison.
    If the LTR model doesn't beat this, we recommend hybrid use.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'issue_density': 0.30,
            'signature_backlog_rate': 0.25,
            'has_sae_pending': 0.25,
            'inverse_sdv': 0.20
        }
    
    def compute_baseline_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute rule-based risk score.
        
        Formula:
            risk = w1 * issue_density + 
                   w2 * signature_backlog_rate + 
                   w3 * has_sae_pending +
                   w4 * (100 - sdv_completion_rate)
        """
        score = pd.Series(0.0, index=df.index)
        
        # Issue density component
        if 'issue_density' in df.columns:
            normalized = self._normalize(df['issue_density'])
            score += self.weights['issue_density'] * normalized
        
        # Signature backlog component
        if 'signature_backlog_rate' in df.columns:
            normalized = self._normalize(df['signature_backlog_rate'])
            score += self.weights['signature_backlog_rate'] * normalized
        
        # SAE pending component
        if 'has_sae_pending' in df.columns:
            score += self.weights['has_sae_pending'] * df['has_sae_pending']
        
        # Inverse SDV (lower SDV = higher risk)
        if 'sdv_completion_rate' in df.columns:
            inverse_sdv = 100 - df['sdv_completion_rate'].fillna(100)
            normalized = self._normalize(inverse_sdv)
            score += self.weights['inverse_sdv'] * normalized
        
        return score
    
    def _normalize(self, series: pd.Series) -> pd.Series:
        """Normalize to 0-1 range"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)


# =============================================================================
# PAIRWISE LABELER
# =============================================================================

class SitePairwiseLabeler:
    """
    Creates pairwise training samples with transparent labeling rules.
    
    IMPORTANT: Labels are NOISY PROXIES, not ground truth.
    """
    
    def __init__(self, config: SiteRankerConfig):
        self.config = config
        self.labeling_stats: Dict = {}
    
    def create_pairs(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Create pairwise training samples.
        
        Returns:
            X: Feature differences (site_a features - site_b features)
            y: Labels (+1 if A > B risk, -1 if A < B risk)
            groups: Study groups for GroupKFold
        """
        logger.info("=" * 70)
        logger.info("CREATING PAIRWISE TRAINING SAMPLES")
        logger.info("=" * 70)
        
        # Get numeric feature columns
        feature_cols = [c for c in df.columns 
                       if c not in ['study_id', 'site_id', 'patient_count', 'log_patient_count']
                       and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        logger.info(f"  Features for pairing: {len(feature_cols)}")
        
        # Sample pairs with constraints
        pairs, labels, groups = self._sample_pairs(df, feature_cols)
        
        # Log statistics
        n_positive = (labels == 1).sum()
        n_negative = (labels == -1).sum()
        
        logger.info(f"  Total pairs: {len(pairs):,}")
        logger.info(f"  Positive (A > B): {n_positive:,} ({n_positive/len(labels)*100:.1f}%)")
        logger.info(f"  Negative (A < B): {n_negative:,} ({n_negative/len(labels)*100:.1f}%)")
        
        self.labeling_stats = {
            'n_pairs': len(pairs),
            'n_positive': int(n_positive),
            'n_negative': int(n_negative),
            'balance_ratio': n_positive / max(n_negative, 1)
        }
        
        return pairs, pd.Series(labels), groups
    
    def _sample_pairs(self, df: pd.DataFrame, feature_cols: List[str]
                     ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Sample pairs with constraints to prevent explosion"""
        
        all_pairs = []
        all_labels = []
        all_groups = []
        
        random.seed(self.config.random_state)
        
        # Group by study for within-study preference
        studies = df['study_id'].unique()
        
        for study in studies:
            study_df = df[df['study_id'] == study].reset_index(drop=True)
            
            if len(study_df) < 2:
                continue
            
            # Get all possible pairs within this study
            site_indices = list(range(len(study_df)))
            possible_pairs = list(combinations(site_indices, 2))
            
            # Sample pairs (limit per site)
            sampled = self._sample_site_pairs(possible_pairs, len(study_df))
            
            for i, j in sampled:
                site_a = study_df.iloc[i]
                site_b = study_df.iloc[j]
                
                # Compute label using transparent rules
                label = self._compute_pair_label(site_a, site_b)
                
                if label == 0:  # Tie - exclude
                    continue
                
                # Compute feature differences
                diff = site_a[feature_cols].values - site_b[feature_cols].values
                all_pairs.append(diff)
                all_labels.append(label)
                all_groups.append(study)
        
        # Convert to arrays
        X = pd.DataFrame(all_pairs, columns=feature_cols)
        y = np.array(all_labels)
        groups = np.array(all_groups)
        
        return X, y, groups
    
    def _sample_site_pairs(self, pairs: List[Tuple[int, int]], n_sites: int) -> List[Tuple[int, int]]:
        """Sample pairs with max per site constraint"""
        
        max_per_site = self.config.max_pairs_per_site
        site_counts = {i: 0 for i in range(n_sites)}
        
        # Shuffle for randomness
        random.shuffle(pairs)
        
        sampled = []
        for i, j in pairs:
            if site_counts[i] < max_per_site and site_counts[j] < max_per_site:
                sampled.append((i, j))
                site_counts[i] += 1
                site_counts[j] += 1
        
        return sampled
    
    def _compute_pair_label(self, site_a: pd.Series, site_b: pd.Series) -> int:
        """
        Compute pairwise label using transparent rules.
        
        Returns:
            +1 if site_a has HIGHER risk than site_b
            -1 if site_a has LOWER risk than site_b
             0 if tie (excluded from training)
        """
        weights = self.config.rule_weights
        score_a = 0.0
        score_b = 0.0
        
        # Rule 1: Issue Density
        issue_a = site_a.get('issue_density', 0) or 0
        issue_b = site_b.get('issue_density', 0) or 0
        if issue_a > issue_b:
            score_a += weights['issue_density']
        elif issue_b > issue_a:
            score_b += weights['issue_density']
        
        # Rule 2: DQI Score (lower = higher risk)
        # Use mean DQI if available, or derive from components
        dqi_cols = [c for c in site_a.index if 'dqi' in c.lower() and 'mean' in c.lower()]
        if dqi_cols:
            dqi_a = site_a.get(dqi_cols[0], 100) or 100
            dqi_b = site_b.get(dqi_cols[0], 100) or 100
            if dqi_a < dqi_b:  # Lower DQI = higher risk
                score_a += weights['dqi_score']
            elif dqi_b < dqi_a:
                score_b += weights['dqi_score']
        
        # Rule 3: Concurrent Issue Types
        issue_type_cols_a = [c for c in site_a.index if 'issue' in c.lower() and '_sum' in c.lower()]
        count_a = sum(1 for c in issue_type_cols_a if (site_a.get(c, 0) or 0) > 0)
        count_b = sum(1 for c in issue_type_cols_a if (site_b.get(c, 0) or 0) > 0)
        if count_a > count_b:
            score_a += weights['concurrent_issues']
        elif count_b > count_a:
            score_b += weights['concurrent_issues']
        
        # Rule 4: Signature Backlog
        sig_a = site_a.get('signature_backlog_rate', 0) or 0
        sig_b = site_b.get('signature_backlog_rate', 0) or 0
        if sig_a > sig_b:
            score_a += weights['signature_backlog']
        elif sig_b > sig_a:
            score_b += weights['signature_backlog']
        
        # Rule 5: SAE Pending (safety critical)
        sae_a = site_a.get('has_sae_pending', 0) or 0
        sae_b = site_b.get('has_sae_pending', 0) or 0
        if sae_a > sae_b:  # A has pending, B doesn't
            score_a += weights['safety_sensitivity']
        elif sae_b > sae_a:
            score_b += weights['safety_sensitivity']
        
        # Determine winner
        if score_a > score_b:
            return 1  # A is higher risk
        elif score_b > score_a:
            return -1  # B is higher risk
        else:
            return 0  # Tie - exclude


# =============================================================================
# SITE RISK RANKER (XGBoost LTR)
# =============================================================================

class SiteRiskRanker:
    """
    XGBoost Learning-to-Rank model for site risk prioritization.
    """
    
    def __init__(self, config: SiteRankerConfig = None):
        self.config = config or SiteRankerConfig()
        self.model = None
        self.feature_importance: Dict[str, float] = {}
        self.feature_columns: List[str] = []
        self.training_stats: Dict = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray):
        """Train the XGBoost ranker on pairwise data"""
        
        logger.info("=" * 70)
        logger.info("TRAINING XGBOOST RANKER")
        logger.info("=" * 70)
        
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not installed")
        
        self.feature_columns = X.columns.tolist()
        
        # Handle NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Convert labels to binary for pairwise
        y_binary = ((y + 1) / 2).astype(int)  # -1,1 -> 0,1
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y_binary)
        
        # Train
        params = self.config.xgb_params.copy()
        n_estimators = params.pop('n_estimators', 200)
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            verbose_eval=False
        )
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        total_importance = sum(importance.values()) if importance else 1
        
        self.feature_importance = {
            k: v / total_importance for k, v in importance.items()
        }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        logger.info(f"  Features used: {len(self.feature_columns)}")
        logger.info(f"  Top 5 features:")
        for i, (feat, imp) in enumerate(list(self.feature_importance.items())[:5]):
            logger.info(f"    {i+1}. {feat}: {imp:.3f}")
        
        self.training_stats = {
            'n_samples': len(X),
            'n_features': len(self.feature_columns),
            'top_features': list(self.feature_importance.items())[:10]
        }
    
    def predict_scores(self, site_df: pd.DataFrame) -> pd.Series:
        """
        Predict risk scores for sites.
        
        Note: Scores are for RANKING only, not probability interpretation.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Get features
        available_features = [c for c in self.feature_columns if c in site_df.columns]
        X = site_df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Pad missing features with zeros
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.feature_columns]
        
        dtest = xgb.DMatrix(X)
        scores = self.model.predict(dtest)
        
        return pd.Series(scores, index=site_df.index)


# =============================================================================
# EVALUATOR
# =============================================================================

class SiteRiskEvaluator:
    """Evaluation metrics for ranking quality"""
    
    def __init__(self, config: SiteRankerConfig = None):
        self.config = config or SiteRankerConfig()
        self.metrics: Dict = {}
    
    def evaluate(self, true_scores: pd.Series, pred_scores: pd.Series,
                baseline_scores: pd.Series = None) -> Dict:
        """
        Comprehensive ranking evaluation.
        
        Args:
            true_scores: Ground truth risk scores (from rule-based labeling)
            pred_scores: Model predicted scores
            baseline_scores: Simple rule-based baseline scores
        """
        logger.info("=" * 70)
        logger.info("EVALUATING RANKING QUALITY")
        logger.info("=" * 70)
        
        results = {}
        
        # NDCG@K
        for k in self.config.ndcg_k_values:
            ndcg = self._compute_ndcg(true_scores, pred_scores, k)
            results[f'ndcg@{k}'] = ndcg
            logger.info(f"  NDCG@{k}: {ndcg:.4f}")
        
        # MAP
        results['map'] = self._compute_map(true_scores, pred_scores)
        logger.info(f"  MAP: {results['map']:.4f}")
        
        # Kendall's Tau
        tau, p_value = kendalltau(true_scores, pred_scores)
        results['kendall_tau'] = tau
        results['kendall_p_value'] = p_value
        logger.info(f"  Kendall's Tau: {tau:.4f} (p={p_value:.4f})")
        
        # Spearman correlation
        spearman, _ = spearmanr(true_scores, pred_scores)
        results['spearman'] = spearman
        logger.info(f"  Spearman: {spearman:.4f}")
        
        # Compare to baseline if provided
        if baseline_scores is not None:
            baseline_tau, _ = kendalltau(true_scores, baseline_scores)
            results['baseline_tau'] = baseline_tau
            results['lift_over_baseline'] = tau - baseline_tau
            logger.info(f"\n  Baseline Kendall's Tau: {baseline_tau:.4f}")
            logger.info(f"  Lift over baseline: {results['lift_over_baseline']:.4f}")
        
        self.metrics = results
        return results
    
    def _compute_ndcg(self, true_scores: pd.Series, pred_scores: pd.Series, k: int) -> float:
        """Compute NDCG@K"""
        # Get top K by predicted scores
        pred_ranking = pred_scores.sort_values(ascending=False).head(k).index
        
        # Compute DCG
        dcg = 0.0
        for i, idx in enumerate(pred_ranking):
            relevance = true_scores.get(idx, 0)
            dcg += relevance / np.log2(i + 2)
        
        # Compute ideal DCG
        ideal_ranking = true_scores.sort_values(ascending=False).head(k)
        idcg = 0.0
        for i, relevance in enumerate(ideal_ranking):
            idcg += relevance / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _compute_map(self, true_scores: pd.Series, pred_scores: pd.Series) -> float:
        """Compute Mean Average Precision"""
        # Define "relevant" as top 25% by true scores
        threshold = true_scores.quantile(0.75)
        relevant = set(true_scores[true_scores >= threshold].index)
        
        if not relevant:
            return 0.0
        
        # Rank by predicted scores
        ranked = pred_scores.sort_values(ascending=False).index
        
        # Compute precision at each relevant item
        precisions = []
        relevant_count = 0
        
        for i, idx in enumerate(ranked):
            if idx in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precisions.append(precision_at_i)
        
        if not precisions:
            return 0.0
        
        return np.mean(precisions)


# =============================================================================
# RED FLAG DETECTOR
# =============================================================================

class RedFlagDetector:
    """
    Detects signs of leakage, overfitting, or invalid models.
    """
    
    def __init__(self, config: SiteRankerConfig = None):
        self.config = config or SiteRankerConfig()
        self.flags: List[Dict] = []
    
    def check(self, metrics: Dict, feature_importance: Dict[str, float],
              ranker: SiteRiskRanker = None, site_df: pd.DataFrame = None) -> List[Dict]:
        """
        Run all red flag checks.
        
        Returns list of detected flags.
        """
        logger.info("=" * 70)
        logger.info("RED FLAG DETECTION")
        logger.info("=" * 70)
        
        self.flags = []
        
        # Check 1: NDCG too high (leakage)
        for k in [5, 10, 20]:
            key = f'ndcg@{k}'
            if key in metrics and metrics[key] > self.config.max_acceptable_ndcg:
                self.flags.append({
                    'type': 'LEAKAGE',
                    'severity': 'CRITICAL',
                    'metric': key,
                    'value': metrics[key],
                    'threshold': self.config.max_acceptable_ndcg,
                    'message': f'{key} = {metrics[key]:.4f} > {self.config.max_acceptable_ndcg} - POSSIBLE LEAKAGE'
                })
        
        # Check 2: NDCG too low (not learning)
        if 'ndcg@10' in metrics and metrics['ndcg@10'] < self.config.min_acceptable_ndcg:
            self.flags.append({
                'type': 'NOT_LEARNING',
                'severity': 'WARNING',
                'metric': 'ndcg@10',
                'value': metrics['ndcg@10'],
                'threshold': self.config.min_acceptable_ndcg,
                'message': f'NDCG@10 = {metrics["ndcg@10"]:.4f} < {self.config.min_acceptable_ndcg} - MODEL NOT LEARNING'
            })
        
        # Check 3: Kendall's Tau too high (identity ranking)
        if 'kendall_tau' in metrics and metrics['kendall_tau'] > self.config.max_acceptable_tau:
            self.flags.append({
                'type': 'IDENTITY_RANKING',
                'severity': 'CRITICAL',
                'metric': 'kendall_tau',
                'value': metrics['kendall_tau'],
                'threshold': self.config.max_acceptable_tau,
                'message': f"Tau = {metrics['kendall_tau']:.4f} ≈ 1.0 - IDENTITY RANKING (model = rules)"
            })
        
        # Check 4: Single feature dominance
        if feature_importance:
            top_feature = list(feature_importance.items())[0]
            if top_feature[1] > self.config.max_feature_importance_pct:
                self.flags.append({
                    'type': 'FEATURE_DOMINANCE',
                    'severity': 'WARNING',
                    'feature': top_feature[0],
                    'value': top_feature[1],
                    'threshold': self.config.max_feature_importance_pct,
                    'message': f'Feature {top_feature[0]} = {top_feature[1]*100:.1f}% importance - SINGLE FEATURE DOMINANCE'
                })
            
            # Top 5 combined dominance
            top5_sum = sum(v for _, v in list(feature_importance.items())[:5])
            if top5_sum > 0.80:
                self.flags.append({
                    'type': 'TOP5_DOMINANCE',
                    'severity': 'INFO',
                    'value': top5_sum,
                    'threshold': 0.80,
                    'message': f'Top 5 features = {top5_sum*100:.1f}% importance - Consider more features'
                })
        
        # Check 5: Rank instability (bootstrap)
        if ranker is not None and site_df is not None:
            overlap = self._check_rank_stability(ranker, site_df)
            if overlap < self.config.min_top10_bootstrap_overlap:
                self.flags.append({
                    'type': 'RANK_INSTABILITY',
                    'severity': 'WARNING',
                    'value': overlap,
                    'threshold': self.config.min_top10_bootstrap_overlap,
                    'message': f'Top-10 bootstrap overlap = {overlap*100:.1f}% < {self.config.min_top10_bootstrap_overlap*100:.0f}% - UNSTABLE RANKINGS'
                })
        
        # Log results
        if not self.flags:
            logger.info("  ✅ No red flags detected")
        else:
            for flag in self.flags:
                severity = flag['severity']
                icon = "🔴" if severity == 'CRITICAL' else "🟡" if severity == 'WARNING' else "🔵"
                logger.warning(f"  {icon} [{severity}] {flag['message']}")
        
        return self.flags
    
    def _check_rank_stability(self, ranker: SiteRiskRanker, site_df: pd.DataFrame,
                              n_bootstrap: int = 10) -> float:
        """Check rank stability via bootstrapping"""
        
        top10_sets = []
        
        np.random.seed(self.config.random_state)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample = site_df.sample(frac=0.8, replace=True)
            
            # Predict scores
            scores = ranker.predict_scores(sample)
            
            # Get top 10
            top10 = set(scores.nlargest(10).index)
            top10_sets.append(top10)
        
        # Compute pairwise overlaps
        overlaps = []
        for i in range(len(top10_sets)):
            for j in range(i + 1, len(top10_sets)):
                overlap = len(top10_sets[i] & top10_sets[j]) / 10
                overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 1.0
    
    def has_critical_flags(self) -> bool:
        """Check if any critical flags were raised"""
        return any(f['severity'] == 'CRITICAL' for f in self.flags)


# =============================================================================
# MAIN RUNNER
# =============================================================================

class SiteRiskRankerRunner:
    """Main runner for Site Risk Ranker training and evaluation"""
    
    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = SiteRankerConfig()
        
        self.aggregator = SiteAggregator()
        self.labeler = SitePairwiseLabeler(self.config)
        self.ranker = SiteRiskRanker(self.config)
        self.baseline = BaselineRanker()
        self.evaluator = SiteRiskEvaluator(self.config)
        self.red_flag_detector = RedFlagDetector(self.config)
        
        self.patient_df: pd.DataFrame = None
        self.site_df: pd.DataFrame = None
        self.results: Dict = {}
        self.iteration: int = 0
        self.max_iterations: int = 5
    
    def load_data(self) -> pd.DataFrame:
        """Load patient-level UPR data from PostgreSQL"""
        logger.info(f"Loading data from PostgreSQL...")
        
        from src.database.pg_data_service import PostgreSQLDataService
        db_service = PostgreSQLDataService()
        
        # Assuming we have a view or table for UPR
        self.patient_df = db_service.get_patients()
        logger.info(f"Loaded {len(self.patient_df):,} patients, {len(self.patient_df.columns)} columns")
        
        return self.patient_df
    
    def run(self) -> Dict:
        """
        Run the complete training pipeline with iteration until defensible.
        """
        logger.info("\n" + "=" * 70)
        logger.info("SANCHALAK AI - SITE RISK RANKER v1.0")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # 1. Load patient data
        self.load_data()
        
        # 2. Aggregate to site level
        self.site_df = self.aggregator.aggregate(self.patient_df)
        
        # 3. Compute baseline scores
        baseline_scores = self.baseline.compute_baseline_score(self.site_df)
        self.site_df['baseline_risk_score'] = baseline_scores
        
        # Iterative training until defensible
        while self.iteration < self.max_iterations:
            self.iteration += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"ITERATION {self.iteration}")
            logger.info("=" * 70)
            
            # 4. Create pairwise training samples
            X_pairs, y_pairs, groups = self.labeler.create_pairs(self.site_df)
            
            # 5. Train ranker
            self.ranker.train(X_pairs, y_pairs, groups)
            
            # 6. Predict site scores
            pred_scores = self.ranker.predict_scores(self.site_df)
            self.site_df['model_risk_score'] = pred_scores
            
            # 7. Evaluate
            metrics = self.evaluator.evaluate(
                true_scores=baseline_scores,  # Use baseline as "ground truth" proxy
                pred_scores=pred_scores,
                baseline_scores=baseline_scores
            )
            
            # 8. Check red flags
            flags = self.red_flag_detector.check(
                metrics=metrics,
                feature_importance=self.ranker.feature_importance,
                ranker=self.ranker,
                site_df=self.site_df
            )
            
            # 9. Check if model is defensible
            if not self.red_flag_detector.has_critical_flags():
                logger.info(f"\n✅ MODEL IS DEFENSIBLE (Iteration {self.iteration})")
                break
            else:
                logger.warning(f"\n⚠️ CRITICAL FLAGS DETECTED - Will iterate")
                # Could add feature removal logic here
        
        # 10. Compile results
        duration = (datetime.now() - start_time).total_seconds()
        
        self.results = {
            'metrics': metrics,
            'flags': flags,
            'feature_importance': self.ranker.feature_importance,
            'aggregation_stats': self.aggregator.aggregation_stats,
            'labeling_stats': self.labeler.labeling_stats,
            'training_stats': self.ranker.training_stats,
            'iterations': self.iteration,
            'duration_seconds': duration,
            'is_defensible': not self.red_flag_detector.has_critical_flags()
        }
        
        return self.results
    
    def get_ranked_sites(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N high-risk sites with explanations"""
        
        if 'model_risk_score' not in self.site_df.columns:
            raise ValueError("Model not trained yet")
        
        # Get top sites
        ranked = self.site_df.nlargest(top_n, 'model_risk_score').copy()
        
        # Add rank
        ranked['risk_rank'] = range(1, len(ranked) + 1)
        
        # Identify primary risk drivers
        ranked['primary_drivers'] = ranked.apply(self._get_primary_drivers, axis=1)
        
        # Add confidence band (based on volatility of scores)
        score_std = self.site_df['model_risk_score'].std()
        ranked['confidence_band'] = ranked['model_risk_score'].apply(
            lambda x: f"±{score_std*0.5:.2f}"
        )
        
        # Select output columns
        output_cols = ['study_id', 'site_id', 'risk_rank', 'model_risk_score',
                       'primary_drivers', 'confidence_band', 'patient_count']
        
        available = [c for c in output_cols if c in ranked.columns]
        
        return ranked[available]
    
    def _get_primary_drivers(self, row: pd.Series) -> str:
        """Identify primary risk drivers for a site"""
        drivers = []
        
        if row.get('has_sae_pending', 0) > 0:
            drivers.append("SAE Pending")
        
        if row.get('issue_density', 0) > self.site_df['issue_density'].quantile(0.75):
            drivers.append("High Issue Density")
        
        if row.get('signature_backlog_rate', 0) > 50:
            drivers.append("Signature Backlog")
        
        if row.get('sdv_completion_rate', 100) < 70:
            drivers.append("Low SDV")
        
        if row.get('query_density', 0) > self.site_df['query_density'].quantile(0.75):
            drivers.append("High Query Burden")
        
        return ", ".join(drivers[:3]) if drivers else "Multiple Minor Issues"
    
    def save(self):
        """Save model and outputs"""
        logger.info("\n" + "=" * 70)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 70)
        
        # Save ranked sites
        ranked_sites = self.get_ranked_sites(top_n=50)
        ranked_path = self.output_dir / 'site_risk_ranking.csv'
        ranked_sites.to_csv(ranked_path, index=False)
        logger.info(f"  ✅ Ranked sites: {ranked_path}")
        
        # Save full site data to PostgreSQL
        from src.database.pg_writer import safe_to_postgres
        success, msg = safe_to_postgres(self.site_df, 'site_metrics_scores', if_exists='replace')
        if success:
             logger.info(f"  ✅ Site metrics saved to PostgreSQL")
        else:
             logger.error(f"  ❌ Failed to save site metrics: {msg}")
        
        # Save model results
        results_path = self.output_dir / 'site_ranker_results.json'
        
        # Convert numpy types for JSON
        results_json = self._prepare_for_json(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"  ✅ Results: {results_path}")
        
        # Save XGBoost model
        if self.ranker.model is not None:
            model_path = self.output_dir / 'site_ranker_model.json'
            self.ranker.model.save_model(str(model_path))
            logger.info(f"  ✅ Model: {model_path}")
    
    def _prepare_for_json(self, obj):
        """Convert numpy types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "=" * 70)
        print("📊 SITE RISK RANKER - TRAINING SUMMARY")
        print("=" * 70)
        
        print(f"\n📁 DATA:")
        print(f"   Patients: {len(self.patient_df):,}")
        print(f"   Sites: {len(self.site_df):,}")
        print(f"   Features: {len(self.aggregator.site_features)}")
        
        print(f"\n📈 METRICS:")
        for key, value in self.results.get('metrics', {}).items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
        
        print(f"\n🔍 STATUS:")
        print(f"   Iterations: {self.results.get('iterations', 0)}")
        print(f"   Defensible: {'✅ YES' if self.results.get('is_defensible') else '❌ NO'}")
        
        flags = self.results.get('flags', [])
        if flags:
            print(f"\n⚠️ FLAGS ({len(flags)}):")
            for flag in flags:
                print(f"   [{flag['severity']}] {flag['type']}")
        
        print(f"\n📁 Output: {self.output_dir}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent.parent  # src/ml -> src -> project_root
    
    # Input: Unified Patient Record (now from DB)
    input_path = Path("DB")
    
    # Output directory
    output_dir = project_root / 'data' / 'processed' / 'ml' / 'site_ranker'
    
    # Output directory
    output_dir = project_root / 'data' / 'processed' / 'ml' / 'site_ranker'
    
    # Run
    runner = SiteRiskRankerRunner(input_path, output_dir)
    runner.run()
    runner.save()
    runner.print_summary()
    
    return runner


if __name__ == '__main__':
    main()
