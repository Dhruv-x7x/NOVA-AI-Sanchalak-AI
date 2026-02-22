"""
a6on-i - Metrics Engine v2.0 (FIXED)
Phase 1.5: DQI, Clean Patient, DB Lock Ready, Site Performance

FIXED: Column mappings now match actual UPR columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# PostgreSQL writer
from src.database.pg_writer import get_pg_writer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ACTUAL COLUMN MAPPINGS (Based on UPR diagnostic)
# =============================================================================

COLUMN_MAP = {
    # Visit columns
    'visit_missing_count': 'visit_missing_visit_count',
    'visit_overdue_avg_days': 'visit_visits_overdue_avg_days',
    'visit_overdue_max_days': 'visit_visits_overdue_max_days',
    
    # Pages columns  
    'pages_missing_count': 'pages_missing_page_count',
    'pages_missing_avg_days': 'pages_pages_missing_avg_days',
    'pages_missing_max_days': 'pages_pages_missing_max_days',
    
    # Query columns
    'total_open_queries': 'total_queries',  # Use total queries
    'open_dm_queries': 'dm_queries',
    'open_clinical_queries': 'clinical_queries',
    'open_medical_queries': 'medical_queries',
    'open_safety_queries': 'safety_queries',
    'open_site_queries': 'site_queries',
    'open_coding_queries': 'coding_queries',
    
    # SDV columns
    'sdv_pct': 'forms_verified',  # Will need to calculate rate
    'crfs_require_sdv': 'crfs_require_verification_sdv',
    
    # Signature columns
    'pending_esignatures': 'crfs_never_signed',
    'broken_signatures': 'broken_signatures',
    'overdue_signatures_45': 'crfs_overdue_for_signs_within_45_days_of_data_entry',
    'overdue_signatures_90': 'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
    'overdue_signatures_90plus': 'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
    
    # SAE columns
    'sae_dm_total': 'sae_dm_sae_dm_total',
    'sae_dm_pending': 'sae_dm_sae_dm_pending',
    'sae_dm_completed': 'sae_dm_sae_dm_completed',
    'sae_safety_total': 'sae_safety_sae_safety_total',
    'sae_safety_pending': 'sae_safety_sae_safety_pending',
    'sae_safety_completed': 'sae_safety_sae_safety_completed',
    
    # Coding columns
    'meddra_total': 'meddra_coding_meddra_total',
    'meddra_coded': 'meddra_coding_meddra_coded',
    'meddra_uncoded': 'meddra_coding_meddra_uncoded',
    'whodrug_total': 'whodrug_coding_whodrug_total',
    'whodrug_coded': 'whodrug_coding_whodrug_coded',
    'whodrug_uncoded': 'whodrug_coding_whodrug_uncoded',
    
    # Lab columns
    'lab_issues': 'lab_lab_issue_count',
    'open_issues_lnr': 'open_issues_lnr',
    
    # EDRR columns
    'edrr_issues': 'edrr_edrr_issue_count',
    'open_issues_edrr': 'open_issues_edrr',
    
    # Inactivated columns
    'inactivated_count': 'inactivated_inactivated_form_count',
    'inactivated_unique': 'inactivated_inactivated_unique_forms',
    
    # CRF columns
    'crfs_frozen': 'crfs_frozen',
    'crfs_not_frozen': 'crfs_not_frozen',
    'crfs_locked': 'crfs_locked',
    'crfs_unlocked': 'crfs_unlocked',
    'crfs_signed': 'crfs_signed',
    
    # Derived columns (already exist)
    'total_issues': 'total_issues_all_sources',
    'total_sae_pending': 'total_sae_pending',
    'coding_completion_rate': 'coding_completion_rate',
    'completeness_score': 'completeness_score',
}


def get_col(df: pd.DataFrame, key: str) -> pd.Series:
    """Get column by key, checking both mapped and direct names"""
    # Check mapped name first
    if key in COLUMN_MAP:
        actual_col = COLUMN_MAP[key]
        if actual_col in df.columns:
            return df[actual_col]
    
    # Check direct name
    if key in df.columns:
        return df[key]
    
    # Return zeros if not found
    return pd.Series(0, index=df.index)


def has_col(df: pd.DataFrame, key: str) -> bool:
    """Check if column exists (mapped or direct)"""
    if key in COLUMN_MAP:
        return COLUMN_MAP[key] in df.columns
    return key in df.columns


# =============================================================================
# DQI CONFIGURATION
# =============================================================================

@dataclass
class DQIConfig:
    """DQI Configuration with weights, penalties, and multipliers"""
    
    # Component weights (must sum to 100)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'safety': 25.0,       # SAE discrepancies (HIGHEST)
        'query': 20.0,        # Open queries, aging
        'completeness': 15.0, # Missing visits, pages
        'coding': 12.0,       # Uncoded terms
        'lab': 10.0,          # Missing labs, ranges
        'sdv': 8.0,           # Verification completion
        'signature': 5.0,     # Overdue signatures
        'edrr': 5.0           # Third-party issues
    })
    
    # DQI Bands
    dqi_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'Pristine': (95, 100),
        'Excellent': (85, 95),
        'Good': (75, 85),
        'Fair': (65, 75),
        'Poor': (50, 65),
        'Critical': (25, 50),
        'Emergency': (0, 25)
    })


# =============================================================================
# DQI CALCULATOR
# =============================================================================

class DQICalculator:
    """8-Component Data Quality Index Calculator"""
    
    def __init__(self, config: DQIConfig = None):
        self.config = config or DQIConfig()
        
    def calculate_component_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate individual component scores"""
        
        logger.info("Calculating 8 DQI component scores...")
        
        result = df.copy()
        
        # 1. SAFETY SCORE (25%) - SAE pending issues
        result['dqi_safety_raw'] = self._calc_safety_score(df)
        
        # 2. QUERY SCORE (20%) - Open queries
        result['dqi_query_raw'] = self._calc_query_score(df)
        
        # 3. COMPLETENESS SCORE (15%) - Missing visits/pages
        result['dqi_completeness_raw'] = self._calc_completeness_score(df)
        
        # 4. CODING SCORE (12%) - Uncoded terms
        result['dqi_coding_raw'] = self._calc_coding_score(df)
        
        # 5. LAB SCORE (10%) - Lab issues
        result['dqi_lab_raw'] = self._calc_lab_score(df)
        
        # 6. SDV SCORE (8%) - Verification status
        result['dqi_sdv_raw'] = self._calc_sdv_score(df)
        
        # 7. SIGNATURE SCORE (5%) - Pending/overdue signatures
        result['dqi_signature_raw'] = self._calc_signature_score(df)
        
        # 8. EDRR SCORE (5%) - Third-party reconciliation
        result['dqi_edrr_raw'] = self._calc_edrr_score(df)
        
        return result
    
    def _calc_safety_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Safety component (SAE pending issues)"""
        # Get SAE pending counts
        sae_dm_pending = get_col(df, 'sae_dm_pending')
        sae_safety_pending = get_col(df, 'sae_safety_pending')
        
        total_pending = sae_dm_pending + sae_safety_pending
        
        # Penalty: -15 per pending SAE (safety critical), max 100
        penalty = np.minimum(total_pending * 15, 100)
        score = 100 - penalty
        
        issues = (total_pending > 0).sum()
        logger.info(f"  Safety: {issues:,} patients with pending SAE")
        
        return score
    
    def _calc_query_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Query component"""
        # Get total queries
        total_queries = get_col(df, 'total_open_queries')
        
        # Also check individual query types
        dm_queries = get_col(df, 'open_dm_queries')
        clinical_queries = get_col(df, 'open_clinical_queries')
        medical_queries = get_col(df, 'open_medical_queries')
        safety_queries = get_col(df, 'open_safety_queries')
        
        # Use max of total or sum of components
        query_sum = dm_queries + clinical_queries + medical_queries + safety_queries
        open_queries = np.maximum(total_queries, query_sum)
        
        # Penalty: -3 per open query, max 100
        penalty = np.minimum(open_queries * 3, 100)
        score = 100 - penalty
        
        issues = (open_queries > 0).sum()
        total = open_queries.sum()
        logger.info(f"  Query: {issues:,} patients with open queries (total: {total:,.0f})")
        
        return score
    
    def _calc_completeness_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Completeness component (missing visits/pages)"""
        # Get missing counts
        missing_visits = get_col(df, 'visit_missing_count')
        missing_pages = get_col(df, 'pages_missing_count')
        
        total_missing = missing_visits + missing_pages
        
        # Penalty: -8 per missing item, max 100
        penalty = np.minimum(total_missing * 8, 100)
        score = 100 - penalty
        
        issues = (total_missing > 0).sum()
        logger.info(f"  Completeness: {issues:,} patients with missing visits/pages")
        
        return score
    
    def _calc_coding_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Coding component (MedDRA + WHODrug)"""
        # Check for existing completion rate
        if 'coding_completion_rate' in df.columns:
            score = df['coding_completion_rate'].fillna(100)
            issues = (score < 100).sum()
            logger.info(f"  Coding: {issues:,} patients with incomplete coding")
            return score
        
        # Otherwise calculate from uncoded counts
        meddra_uncoded = get_col(df, 'meddra_uncoded')
        whodrug_uncoded = get_col(df, 'whodrug_uncoded')
        
        total_uncoded = meddra_uncoded + whodrug_uncoded
        
        # Penalty: -5 per uncoded term, max 100
        penalty = np.minimum(total_uncoded * 5, 100)
        score = 100 - penalty
        
        issues = (total_uncoded > 0).sum()
        logger.info(f"  Coding: {issues:,} patients with uncoded terms")
        
        return score
    
    def _calc_lab_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Lab component"""
        # Get lab issues
        lab_issues = get_col(df, 'lab_issues')
        lnr_issues = get_col(df, 'open_issues_lnr')
        
        total_lab = lab_issues + lnr_issues
        
        # Penalty: -10 per lab issue, max 100
        penalty = np.minimum(total_lab * 10, 100)
        score = 100 - penalty
        
        issues = (total_lab > 0).sum()
        logger.info(f"  Lab: {issues:,} patients with lab issues")
        
        return score
    
    def _calc_sdv_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate SDV component"""
        # Get forms verified and required
        forms_verified = get_col(df, 'sdv_pct')
        forms_required = get_col(df, 'crfs_require_sdv')
        
        # Calculate SDV rate
        if forms_required.sum() > 0:
            # Calculate patient-level SDV rate
            sdv_rate = np.where(
                forms_required > 0,
                (forms_verified / forms_required * 100).clip(0, 100),
                100  # No SDV required = 100%
            )
        else:
            sdv_rate = pd.Series(100, index=df.index)
        
        incomplete = (sdv_rate < 100).sum()
        logger.info(f"  SDV: {incomplete:,} patients with incomplete SDV")
        
        return pd.Series(sdv_rate, index=df.index)
    
    def _calc_signature_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Signature component"""
        # Get signature issues
        never_signed = get_col(df, 'pending_esignatures')
        broken_sigs = get_col(df, 'broken_signatures')
        overdue_45 = get_col(df, 'overdue_signatures_45')
        overdue_90 = get_col(df, 'overdue_signatures_90')
        overdue_90plus = get_col(df, 'overdue_signatures_90plus')
        
        # Weighted penalty based on age
        total_sig_issues = (
            never_signed * 1.0 +
            broken_sigs * 2.0 +  # Broken sigs are more serious
            overdue_45 * 1.0 +
            overdue_90 * 1.5 +
            overdue_90plus * 2.0  # Very old overdue are worse
        )
        
        # Penalty: -5 per weighted issue, max 100
        penalty = np.minimum(total_sig_issues * 5, 100)
        score = 100 - penalty
        
        issues = (total_sig_issues > 0).sum()
        logger.info(f"  Signature: {issues:,} patients with signature issues")
        
        return score
    
    def _calc_edrr_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate EDRR component"""
        # Get EDRR issues
        edrr_issues = get_col(df, 'edrr_issues')
        open_edrr = get_col(df, 'open_issues_edrr')
        
        total_edrr = np.maximum(edrr_issues, open_edrr)
        
        # Penalty: -10 per EDRR issue, max 100
        penalty = np.minimum(total_edrr * 10, 100)
        score = 100 - penalty
        
        issues = (total_edrr > 0).sum()
        logger.info(f"  EDRR: {issues:,} patients with EDRR issues")
        
        return score
    
    def calculate_weighted_dqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted DQI from component scores"""
        
        result = df.copy()
        
        # Component mapping
        components = {
            'safety': ('dqi_safety_raw', self.config.weights['safety']),
            'query': ('dqi_query_raw', self.config.weights['query']),
            'completeness': ('dqi_completeness_raw', self.config.weights['completeness']),
            'coding': ('dqi_coding_raw', self.config.weights['coding']),
            'lab': ('dqi_lab_raw', self.config.weights['lab']),
            'sdv': ('dqi_sdv_raw', self.config.weights['sdv']),
            'signature': ('dqi_signature_raw', self.config.weights['signature']),
            'edrr': ('dqi_edrr_raw', self.config.weights['edrr'])
        }
        
        # Calculate weighted components
        weighted_sum = pd.Series(0.0, index=df.index)
        
        for name, (col, weight) in components.items():
            if col in result.columns:
                weighted_component = result[col] * (weight / 100)
                result[f'dqi_{name}_weighted'] = weighted_component
                weighted_sum += weighted_component
            else:
                result[f'dqi_{name}_weighted'] = weight  # Full score if no data
                weighted_sum += weight
        
        # Final DQI
        result['dqi_score'] = weighted_sum.clip(0, 100)
        
        return result
    
    def assign_dqi_band(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign DQI band based on score"""
        
        result = df.copy()
        
        def get_band(score):
            if score >= 95:
                return 'Pristine'
            elif score >= 85:
                return 'Excellent'
            elif score >= 75:
                return 'Good'
            elif score >= 65:
                return 'Fair'
            elif score >= 50:
                return 'Poor'
            elif score >= 25:
                return 'Critical'
            else:
                return 'Emergency'
        
        result['dqi_band'] = result['dqi_score'].apply(get_band)
        
        return result
    
    def calculate_full_dqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete DQI calculation pipeline"""
        
        logger.info("=" * 60)
        logger.info("CALCULATING 8-COMPONENT DQI")
        logger.info("=" * 60)
        
        # Step 1: Component scores
        result = self.calculate_component_scores(df)
        
        # Step 2: Weighted DQI
        result = self.calculate_weighted_dqi(result)
        
        # Step 3: Assign bands
        result = self.assign_dqi_band(result)
        
        # Log summary
        avg_dqi = result['dqi_score'].mean()
        median_dqi = result['dqi_score'].median()
        min_dqi = result['dqi_score'].min()
        max_dqi = result['dqi_score'].max()
        
        logger.info(f"\nDQI SUMMARY:")
        logger.info(f"  Mean: {avg_dqi:.2f}")
        logger.info(f"  Median: {median_dqi:.2f}")
        logger.info(f"  Range: {min_dqi:.2f} - {max_dqi:.2f}")
        
        logger.info(f"\n  DQI Band Distribution:")
        for band in ['Pristine', 'Excellent', 'Good', 'Fair', 'Poor', 'Critical', 'Emergency']:
            count = (result['dqi_band'] == band).sum()
            pct = count / len(result) * 100 if len(result) > 0 else 0
            if count > 0:
                logger.info(f"    {band}: {count:,} ({pct:.1f}%)")
        
        return result


# =============================================================================
# CLEAN PATIENT CALCULATOR
# =============================================================================

class CleanPatientCalculator:
    """Two-Tier Clean Patient Derivation using actual column names"""
    
    def __init__(self):
        # Tier 1: Clinical Clean (7 Hard Blocks) - ACTUAL COLUMN NAMES
        self.tier1_criteria = {
            'no_missing_visits': {
                'description': 'No missing visits',
                'column': 'visit_missing_visit_count',
                'condition': 'equals_zero'
            },
            'no_missing_pages': {
                'description': 'No missing CRF pages',
                'column': 'pages_missing_page_count',
                'condition': 'equals_zero'
            },
            'no_open_queries': {
                'description': 'No open queries',
                'column': 'total_queries',
                'condition': 'equals_zero'
            },
            'sdv_complete': {
                'description': 'SDV complete (no CRFs requiring verification)',
                'column': 'crfs_require_verification_sdv',
                'condition': 'equals_zero'
            },
            'signatures_complete': {
                'description': 'All signatures complete',
                'column': 'crfs_never_signed',
                'condition': 'equals_zero'
            },
            'meddra_coded': {
                'description': 'MedDRA coding complete',
                'column': 'meddra_coding_meddra_uncoded',
                'condition': 'equals_zero'
            },
            'whodrug_coded': {
                'description': 'WHODrug coding complete',
                'column': 'whodrug_coding_whodrug_uncoded',
                'condition': 'equals_zero'
            }
        }
        
        # Tier 2: Operational Clean (7 Soft Blocks) - ACTUAL COLUMN NAMES
        self.tier2_criteria = {
            'no_lab_issues': {
                'description': 'No lab issues',
                'column': 'lab_lab_issue_count',
                'condition': 'equals_zero'
            },
            'no_sae_dm_pending': {
                'description': 'No SAE DM issues pending',
                'column': 'sae_dm_sae_dm_pending',
                'condition': 'equals_zero'
            },
            'no_sae_safety_pending': {
                'description': 'No SAE Safety issues pending',
                'column': 'sae_safety_sae_safety_pending',
                'condition': 'equals_zero'
            },
            'no_edrr_issues': {
                'description': 'No EDRR reconciliation issues',
                'column': 'edrr_edrr_issue_count',
                'condition': 'equals_zero'
            },
            'no_overdue_signatures': {
                'description': 'No overdue signatures',
                'column': 'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
                'condition': 'equals_zero'
            },
            'no_inactivated_forms': {
                'description': 'No unexplained inactivated forms',
                'column': 'inactivated_inactivated_form_count',
                'condition': 'equals_zero'
            },
            'no_broken_signatures': {
                'description': 'No broken signatures',
                'column': 'broken_signatures',
                'condition': 'equals_zero'
            }
        }
    
    def _check_condition(self, series: pd.Series, condition: str) -> pd.Series:
        """Evaluate a condition on a series"""
        if condition == 'equals_zero':
            return series.fillna(0) == 0
        elif condition == 'equals_100':
            return series.fillna(0) >= 100
        elif condition == 'greater_than_zero':
            return series.fillna(0) > 0
        return pd.Series(True, index=series.index)
    
    def calculate_tier1_clinical_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Tier 1: Clinical Clean status"""
        
        logger.info("\n" + "=" * 60)
        logger.info("CALCULATING TIER 1: CLINICAL CLEAN (7 HARD BLOCKS)")
        logger.info("=" * 60)
        
        result = df.copy()
        all_pass = pd.Series(True, index=df.index)
        
        for criterion_name, criterion in self.tier1_criteria.items():
            col = criterion['column']
            condition = criterion['condition']
            description = criterion['description']
            
            if col in df.columns:
                passes = self._check_condition(df[col], condition)
                found = True
            else:
                passes = pd.Series(True, index=df.index)
                found = False
            
            result[f'tier1_{criterion_name}'] = passes
            all_pass = all_pass & passes
            
            failures = (~passes).sum()
            pct = failures / len(df) * 100 if len(df) > 0 else 0
            status = "✅" if failures == 0 else "❌"
            found_str = "" if found else " (column not found)"
            logger.info(f"  {status} {description}: {failures:,} failures ({pct:.1f}%){found_str}")
        
        result['tier1_clinical_clean'] = all_pass
        
        # Get blocking reason
        def get_blocking_reason(row):
            for criterion_name in self.tier1_criteria.keys():
                if not row.get(f'tier1_{criterion_name}', True):
                    return criterion_name
            return None
        
        result['tier1_blocking_reason'] = result.apply(get_blocking_reason, axis=1)
        
        clean_count = all_pass.sum()
        clean_pct = clean_count / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"\n  TIER 1 CLINICAL CLEAN: {clean_count:,} / {len(df):,} ({clean_pct:.1f}%)")
        
        return result
    
    def calculate_tier2_operational_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Tier 2: Operational Clean status"""
        
        logger.info("\n" + "=" * 60)
        logger.info("CALCULATING TIER 2: OPERATIONAL CLEAN (7 SOFT BLOCKS)")
        logger.info("=" * 60)
        
        result = df.copy()
        tier1_clean = result.get('tier1_clinical_clean', pd.Series(True, index=df.index))
        all_pass = tier1_clean.copy()
        
        for criterion_name, criterion in self.tier2_criteria.items():
            col = criterion['column']
            condition = criterion['condition']
            description = criterion['description']
            
            if col in df.columns:
                passes = self._check_condition(df[col], condition)
                found = True
            else:
                passes = pd.Series(True, index=df.index)
                found = False
            
            result[f'tier2_{criterion_name}'] = passes
            all_pass = all_pass & passes
            
            # Count failures among Tier 1 clean
            tier1_mask = tier1_clean == True
            failures = (~passes & tier1_mask).sum()
            pct = failures / tier1_mask.sum() * 100 if tier1_mask.sum() > 0 else 0
            status = "✅" if failures == 0 else "⚠️"
            found_str = "" if found else " (column not found)"
            logger.info(f"  {status} {description}: {failures:,} failures ({pct:.1f}% of Tier 1){found_str}")
        
        result['tier2_operational_clean'] = all_pass
        
        def get_soft_blocking_reason(row):
            if not row.get('tier1_clinical_clean', False):
                return 'Not Tier 1 Clean'
            for criterion_name in self.tier2_criteria.keys():
                if not row.get(f'tier2_{criterion_name}', True):
                    return criterion_name
            return None
        
        result['tier2_blocking_reason'] = result.apply(get_soft_blocking_reason, axis=1)
        
        clean_count = all_pass.sum()
        clean_pct = clean_count / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"\n  TIER 2 OPERATIONAL CLEAN: {clean_count:,} / {len(df):,} ({clean_pct:.1f}%)")
        
        return result
    
    def calculate_quick_wins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify Quick Wins - patients close to clean status"""
        
        logger.info("\n" + "=" * 60)
        logger.info("IDENTIFYING QUICK WINS")
        logger.info("=" * 60)
        
        result = df.copy()
        
        # Count failing criteria
        tier1_cols = [f'tier1_{c}' for c in self.tier1_criteria.keys()]
        tier2_cols = [f'tier2_{c}' for c in self.tier2_criteria.keys()]
        
        available_tier1 = [c for c in tier1_cols if c in result.columns]
        available_tier2 = [c for c in tier2_cols if c in result.columns]
        
        result['tier1_failing_count'] = (~result[available_tier1]).sum(axis=1) if available_tier1 else 0
        result['tier2_failing_count'] = (~result[available_tier2]).sum(axis=1) if available_tier2 else 0
        
        # Quick win categories
        result['quick_win_category'] = 'Not Quick Win'
        
        # Already clean
        mask_clean = result.get('tier2_operational_clean', False) == True
        result.loc[mask_clean, 'quick_win_category'] = 'Already Clean'
        
        # Tier 1 clean, 1 issue to Tier 2
        mask_t1_clean = result.get('tier1_clinical_clean', False) == True
        mask_1_to_t2 = mask_t1_clean & (result['tier2_failing_count'] == 1) & ~mask_clean
        result.loc[mask_1_to_t2, 'quick_win_category'] = '1 Issue to Tier 2'
        
        # 1 issue to Tier 1
        mask_1_to_t1 = (result['tier1_failing_count'] == 1) & ~mask_t1_clean
        result.loc[mask_1_to_t1, 'quick_win_category'] = '1 Issue to Tier 1'
        
        # 2 issues to Tier 1
        mask_2_to_t1 = (result['tier1_failing_count'] == 2) & ~mask_t1_clean
        result.loc[mask_2_to_t1, 'quick_win_category'] = '2 Issues to Tier 1'
        
        # Log summary
        for category in ['Already Clean', '1 Issue to Tier 2', '1 Issue to Tier 1', '2 Issues to Tier 1', 'Not Quick Win']:
            count = (result['quick_win_category'] == category).sum()
            pct = count / len(result) * 100 if len(result) > 0 else 0
            if count > 0:
                logger.info(f"  {category}: {count:,} ({pct:.1f}%)")
        
        return result
    
    def calculate_full_clean_patient(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete clean patient pipeline"""
        result = self.calculate_tier1_clinical_clean(df)
        result = self.calculate_tier2_operational_clean(result)
        result = self.calculate_quick_wins(result)
        return result


# =============================================================================
# DB LOCK READY CALCULATOR
# =============================================================================

class DBLockReadyCalculator:
    """Two-Tier DB Lock Ready Assessment"""
    
    def __init__(self):
        self.eligible_statuses = ['Ongoing', 'Completed', 'Discontinued']
    
    def calculate_db_lock_ready(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate DB Lock readiness"""
        
        logger.info("\n" + "=" * 60)
        logger.info("CALCULATING DB LOCK READY STATUS (2-TIER)")
        logger.info("=" * 60)
        
        result = df.copy()
        
        # Check eligibility
        status_col = 'subject_status_clean' if 'subject_status_clean' in df.columns else 'subject_status'
        eligible_mask = result[status_col].isin(self.eligible_statuses)
        
        logger.info(f"\nEligible for DB Lock: {eligible_mask.sum():,} / {len(df):,}")
        
        # Tier 1: Fully Ready = eligible + Tier 2 operational clean + no pending safety
        tier2_clean = result.get('tier2_operational_clean', pd.Series(True, index=df.index))
        
        # Check pending safety items
        sae_dm_pending = get_col(df, 'sae_dm_pending')
        sae_safety_pending = get_col(df, 'sae_safety_pending')
        pending_sigs = get_col(df, 'pending_esignatures')
        
        has_pending = (sae_dm_pending > 0) | (sae_safety_pending > 0) | (pending_sigs > 0)
        
        tier1_ready = eligible_mask & tier2_clean & ~has_pending
        result['db_lock_tier1_ready'] = tier1_ready
        
        tier1_count = tier1_ready.sum()
        tier1_pct = tier1_count / eligible_mask.sum() * 100 if eligible_mask.sum() > 0 else 0
        logger.info(f"\n  TIER 1 - FULLY READY: {tier1_count:,} ({tier1_pct:.1f}% of eligible)")
        
        # Tier 2: Pending = eligible + Tier 1 clinical clean but not Tier 1 ready
        tier1_clean = result.get('tier1_clinical_clean', pd.Series(True, index=df.index))
        tier2_pending = eligible_mask & tier1_clean & ~tier1_ready
        result['db_lock_tier2_pending'] = tier2_pending
        
        tier2_count = tier2_pending.sum()
        tier2_pct = tier2_count / eligible_mask.sum() * 100 if eligible_mask.sum() > 0 else 0
        logger.info(f"  TIER 2 - PENDING: {tier2_count:,} ({tier2_pct:.1f}% of eligible)")
        
        # Not Ready
        not_ready = eligible_mask & ~tier1_clean
        result['db_lock_not_ready'] = not_ready
        
        not_ready_count = not_ready.sum()
        not_ready_pct = not_ready_count / eligible_mask.sum() * 100 if eligible_mask.sum() > 0 else 0
        logger.info(f"  NOT READY: {not_ready_count:,} ({not_ready_pct:.1f}% of eligible)")
        
        # Combined status
        def get_db_lock_status(row):
            if row.get('db_lock_tier1_ready', False):
                return 'Tier 1 - Ready'
            elif row.get('db_lock_tier2_pending', False):
                return 'Tier 2 - Pending'
            elif eligible_mask.loc[row.name]:
                return 'Not Ready'
            else:
                return 'Not Eligible'
        
        result['db_lock_status'] = result.apply(get_db_lock_status, axis=1)
        
        # Effort score
        result['db_lock_effort_score'] = (
            result.get('tier1_failing_count', 0) * 2 + 
            result.get('tier2_failing_count', 0)
        )
        
        return result


# =============================================================================
# SITE PERFORMANCE INDEX
# =============================================================================

class SitePerformanceCalculator:
    """Site Performance Index Calculator"""
    
    def calculate_site_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate site-level performance metrics"""
        
        logger.info("\n" + "=" * 60)
        logger.info("CALCULATING SITE PERFORMANCE INDEX")
        logger.info("=" * 60)
        
        # Aggregation
        agg_dict = {
            'patient_key': 'count',
            'dqi_score': ['mean', 'min', 'max', 'std'],
            'tier1_clinical_clean': 'sum',
            'tier2_operational_clean': 'sum',
            'db_lock_tier1_ready': 'sum',
            'total_queries': 'sum',
            'total_issues_all_sources': 'sum'
        }
        
        # Filter to available columns
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        agg_dict['patient_key'] = 'count'  # Always count
        
        site_df = df.groupby(['study_id', 'site_id']).agg(agg_dict).reset_index()
        
        # Flatten columns
        new_cols = []
        for col in site_df.columns:
            if isinstance(col, tuple):
                new_cols.append('_'.join([str(c) for c in col if c]))
            else:
                new_cols.append(col)
        site_df.columns = new_cols
        
        # Rename
        rename_map = {
            'patient_key_count': 'patient_count',
            'patient_key': 'patient_count',
            'dqi_score_mean': 'site_dqi_mean',
            'dqi_score_min': 'site_dqi_min',
            'dqi_score_max': 'site_dqi_max',
            'dqi_score_std': 'site_dqi_std',
            'tier1_clinical_clean_sum': 'tier1_clean_count',
            'tier2_operational_clean_sum': 'tier2_clean_count',
            'db_lock_tier1_ready_sum': 'db_lock_ready_count',
            'total_queries_sum': 'total_queries',
            'total_issues_all_sources_sum': 'total_issues'
        }
        site_df = site_df.rename(columns={k: v for k, v in rename_map.items() if k in site_df.columns})
        
        # Calculate rates
        if 'patient_count' in site_df.columns:
            for col in ['tier1_clean', 'tier2_clean', 'db_lock_ready']:
                count_col = f'{col}_count'
                if count_col in site_df.columns:
                    site_df[f'{col}_rate'] = (site_df[count_col] / site_df['patient_count'] * 100).round(1)
        
        # Site Performance Index
        spi_components = []
        if 'site_dqi_mean' in site_df.columns:
            spi_components.append(('dqi', site_df['site_dqi_mean'] * 0.4))
        if 'tier1_clean_rate' in site_df.columns:
            spi_components.append(('tier1', site_df['tier1_clean_rate'] * 0.3))
        if 'tier2_clean_rate' in site_df.columns:
            spi_components.append(('tier2', site_df['tier2_clean_rate'] * 0.2))
        if 'db_lock_ready_rate' in site_df.columns:
            spi_components.append(('dblock', site_df['db_lock_ready_rate'] * 0.1))
        
        if spi_components:
            site_df['site_performance_index'] = sum([c[1] for c in spi_components]).clip(0, 100).round(2)
        else:
            site_df['site_performance_index'] = 50.0
        
        # Performance tier
        def get_tier(spi):
            if spi >= 90: return 'Exceptional'
            elif spi >= 80: return 'Strong'
            elif spi >= 70: return 'Adequate'
            elif spi >= 60: return 'Needs Improvement'
            else: return 'At Risk'
        
        site_df['performance_tier'] = site_df['site_performance_index'].apply(get_tier)
        
        # Log summary
        logger.info(f"\n  Total Sites: {len(site_df):,}")
        logger.info(f"  Average SPI: {site_df['site_performance_index'].mean():.1f}")
        
        for tier in ['Exceptional', 'Strong', 'Adequate', 'Needs Improvement', 'At Risk']:
            count = (site_df['performance_tier'] == tier).sum()
            if count > 0:
                pct = count / len(site_df) * 100
                logger.info(f"  {tier}: {count} sites ({pct:.1f}%)")
        
        return site_df


# =============================================================================
# METRICS ENGINE (MAIN)
# =============================================================================

class MetricsEngine:
    """Main Metrics Engine - Orchestrates all calculations"""
    
    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        self.site_df = None
        self.study_df = None
        self.summary = {}
        
        self.dqi_calc = DQICalculator()
        self.clean_calc = CleanPatientCalculator()
        self.dblock_calc = DBLockReadyCalculator()
        self.site_calc = SitePerformanceCalculator()
    
    def load_data(self) -> pd.DataFrame:
        """Load input data from PostgreSQL"""
        logger.info(f"Loading data from PostgreSQL...")
        
        try:
             from src.database.pg_data_service import PostgreSQLDataService
             db_service = PostgreSQLDataService()
             # Assuming 'unified_patient_record' exists in DB
             self.df = db_service.get_patients() 
             logger.info(f"Loaded {len(self.df):,} patients from DB")
        except Exception as e:
             logger.error(f"Failed to load data from DB: {e}")
             self.df = pd.DataFrame()

        return self.df
    
    def run_all_calculations(self) -> pd.DataFrame:
        """Run all metric calculations"""
        
        logger.info("\n" + "=" * 70)
        logger.info("a6on-i - METRICS ENGINE v2.0")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # 1. DQI
        self.df = self.dqi_calc.calculate_full_dqi(self.df)
        
        # 2. Clean Patient
        self.df = self.clean_calc.calculate_full_clean_patient(self.df)
        
        # 3. DB Lock Ready
        self.df = self.dblock_calc.calculate_db_lock_ready(self.df)
        
        # 4. Site Performance
        self.site_df = self.site_calc.calculate_site_metrics(self.df)
        
        # 5. Study metrics
        self.study_df = self._calculate_study_metrics()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 70)
        logger.info("METRICS ENGINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.2f} seconds")
        
        return self.df
    
    def _calculate_study_metrics(self) -> pd.DataFrame:
        """Calculate study-level metrics"""
        
        logger.info("\n" + "=" * 60)
        logger.info("CALCULATING STUDY-LEVEL METRICS")
        logger.info("=" * 60)
        
        agg_dict = {
            'patient_key': 'count',
            'dqi_score': 'mean',
            'tier1_clinical_clean': 'sum',
            'tier2_operational_clean': 'sum',
            'db_lock_tier1_ready': 'sum'
        }
        
        agg_dict = {k: v for k, v in agg_dict.items() if k in self.df.columns}
        agg_dict['patient_key'] = 'count'
        
        study_df = self.df.groupby('study_id').agg(agg_dict).reset_index()
        study_df = study_df.rename(columns={
            'patient_key': 'patient_count',
            'dqi_score': 'study_dqi_mean',
            'tier1_clinical_clean': 'tier1_clean_count',
            'tier2_operational_clean': 'tier2_clean_count',
            'db_lock_tier1_ready': 'db_lock_ready_count'
        })
        
        # Calculate rates
        for col in ['tier1_clean', 'tier2_clean', 'db_lock_ready']:
            count_col = f'{col}_count'
            if count_col in study_df.columns:
                study_df[f'{col}_rate'] = (study_df[count_col] / study_df['patient_count'] * 100).round(1)
        
        logger.info(f"  Studies analyzed: {len(study_df)}")
        
        return study_df
    
    def generate_summary(self) -> Dict:
        """Generate metrics summary"""
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'version': '2.0.0',
            'patient_count': len(self.df),
            
            'dqi': {
                'mean': round(float(self.df['dqi_score'].mean()), 2),
                'median': round(float(self.df['dqi_score'].median()), 2),
                'std': round(float(self.df['dqi_score'].std()), 2),
                'min': round(float(self.df['dqi_score'].min()), 2),
                'max': round(float(self.df['dqi_score'].max()), 2),
                'band_distribution': self.df['dqi_band'].value_counts().to_dict()
            },
            
            'clean_patient': {
                'tier1_count': int(self.df['tier1_clinical_clean'].sum()),
                'tier1_rate': round(float(self.df['tier1_clinical_clean'].mean() * 100), 2),
                'tier2_count': int(self.df['tier2_operational_clean'].sum()),
                'tier2_rate': round(float(self.df['tier2_operational_clean'].mean() * 100), 2)
            },
            
            'db_lock': {
                'tier1_ready': int(self.df['db_lock_tier1_ready'].sum()),
                'tier2_pending': int(self.df['db_lock_tier2_pending'].sum()),
                'not_ready': int(self.df['db_lock_not_ready'].sum()),
                'status_distribution': self.df['db_lock_status'].value_counts().to_dict()
            },
            
            'site_performance': {
                'total_sites': len(self.site_df),
                'mean_spi': round(float(self.site_df['site_performance_index'].mean()), 2),
                'tier_distribution': self.site_df['performance_tier'].value_counts().to_dict()
            },
            
            'quick_wins': self.df['quick_win_category'].value_counts().to_dict()
        }
        
        self.summary = summary
        return summary
    
    def save_outputs(self) -> Dict[str, Path]:
        """Save all outputs to PostgreSQL"""
        
        logger.info("\n" + "=" * 60)
        logger.info("SAVING METRICS OUTPUTS TO POSTGRESQL")
        logger.info("=" * 60)
        
        from src.database.pg_writer import safe_to_postgres
        saved_files = {}
        
        # 1. Patient metrics
        success, msg = safe_to_postgres(self.df, 'patient_metrics', if_exists='replace')
        if success:
            logger.info(f"✅ Saved patient metrics to DB")
        else:
            logger.error(f"❌ Failed to save patient metrics: {msg}")

        # 2. Site metrics
        success, msg = safe_to_postgres(self.site_df, 'site_metrics', if_exists='replace')
        if success:
             logger.info(f"✅ Saved site metrics to DB")
        
        # 3. Study metrics
        success, msg = safe_to_postgres(self.study_df, 'study_metrics', if_exists='replace')
        if success:
             logger.info(f"✅ Saved study metrics to DB")
        
        # 4. Summary JSON (Keep as file for quick access/logs, or save to a metadata table)
        # We will keep the JSON file for now as it's metadata, but valid to keep lightweight files.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.generate_summary()
        summary_path = self.output_dir / 'metrics_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files['summary'] = summary_path
        logger.info(f"✅ Saved summary: {summary_path}")
        
        # 5. DQI distribution
        dqi_dist = self.df.groupby('dqi_band').agg({
            'patient_key': 'count',
            'dqi_score': ['mean', 'min', 'max']
        }).reset_index()
        dqi_dist.columns = ['dqi_band', 'patient_count', 'mean_score', 'min_score', 'max_score']
        dqi_path = self.output_dir / 'dqi_distribution.csv'
        dqi_dist.to_csv(dqi_path, index=False)
        saved_files['dqi_distribution'] = dqi_path
        logger.info(f"✅ Saved DQI distribution: {dqi_path}")
        
        # 6. Quick wins
        quick_win_cols = ['patient_key', 'study_id', 'site_id', 'subject_id', 
                          'dqi_score', 'dqi_band', 'quick_win_category', 
                          'tier1_failing_count', 'tier2_failing_count',
                          'tier1_blocking_reason', 'tier2_blocking_reason']
        available_cols = [c for c in quick_win_cols if c in self.df.columns]
        
        quick_wins = self.df[self.df['quick_win_category'] != 'Already Clean'][available_cols]
        quick_wins = quick_wins[quick_wins['quick_win_category'] != 'Not Quick Win']
        quick_wins = quick_wins.sort_values('tier1_failing_count')
        
        qw_path = self.output_dir / 'quick_wins.csv'
        quick_wins.to_csv(qw_path, index=False)
        saved_files['quick_wins'] = qw_path
        logger.info(f"✅ Saved quick wins: {len(quick_wins):,} patients")
        
        # 7. Blocking reasons summary
        blocking_summary = []
        for col in ['tier1_blocking_reason', 'tier2_blocking_reason']:
            if col in self.df.columns:
                counts = self.df[col].value_counts()
                for reason, count in counts.items():
                    if reason:
                        blocking_summary.append({
                            'tier': col.replace('_blocking_reason', ''),
                            'reason': reason,
                            'count': count,
                            'pct': round(count / len(self.df) * 100, 2)
                        })
        
        blocking_df = pd.DataFrame(blocking_summary)
        blocking_path = self.output_dir / 'blocking_reasons.csv'
        blocking_df.to_csv(blocking_path, index=False)
        saved_files['blocking_reasons'] = blocking_path
        logger.info(f"✅ Saved blocking reasons: {blocking_path}")
        
        # 8. Manifest
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0',
            'input_file': str(self.input_path),
            'patient_count': len(self.df),
            'site_count': len(self.site_df),
            'study_count': len(self.study_df),
            'files_saved': {k: str(v) for k, v in saved_files.items()}
        }
        
        manifest_path = self.output_dir / 'metrics_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        saved_files['manifest'] = manifest_path
        logger.info(f"✅ Saved manifest: {manifest_path}")
        
        return saved_files
    
    def print_summary(self):
        """Print summary"""
        
        print("\n" + "=" * 70)
        print("📊 PHASE 1.5 COMPLETE - METRICS ENGINE v2.0")
        print("=" * 70)
        
        print(f"\n🔢 PATIENT POPULATION: {len(self.df):,}")
        
        # DQI
        print(f"\n📈 DQI (Data Quality Index):")
        print(f"   Mean: {self.df['dqi_score'].mean():.1f}")
        print(f"   Median: {self.df['dqi_score'].median():.1f}")
        print(f"   Range: {self.df['dqi_score'].min():.1f} - {self.df['dqi_score'].max():.1f}")
        
        print(f"\n   DQI Bands:")
        for band in ['Pristine', 'Excellent', 'Good', 'Fair', 'Poor', 'Critical', 'Emergency']:
            count = (self.df['dqi_band'] == band).sum()
            if count > 0:
                pct = count / len(self.df) * 100
                bar = "█" * max(1, int(pct / 2))
                print(f"   {band:12} {count:>6,} ({pct:>5.1f}%) {bar}")
        
        # Clean Patient
        print(f"\n✅ CLEAN PATIENT STATUS:")
        t1 = self.df['tier1_clinical_clean'].sum()
        t1_pct = t1 / len(self.df) * 100
        t2 = self.df['tier2_operational_clean'].sum()
        t2_pct = t2 / len(self.df) * 100
        print(f"   Tier 1 (Clinical Clean): {t1:,} ({t1_pct:.1f}%)")
        print(f"   Tier 2 (Operational Clean): {t2:,} ({t2_pct:.1f}%)")
        
        # Blocking reasons
        print(f"\n   Top Blocking Reasons (Tier 1):")
        t1_reasons = self.df['tier1_blocking_reason'].value_counts().head(5)
        for reason, count in t1_reasons.items():
            if reason:
                print(f"   - {reason}: {count:,}")
        
        # DB Lock
        print(f"\n🔒 DB LOCK READY:")
        for status in ['Tier 1 - Ready', 'Tier 2 - Pending', 'Not Ready', 'Not Eligible']:
            count = (self.df['db_lock_status'] == status).sum()
            pct = count / len(self.df) * 100
            print(f"   {status}: {count:,} ({pct:.1f}%)")
        
        # Quick Wins
        print(f"\n⚡ QUICK WINS:")
        for cat in ['1 Issue to Tier 1', '2 Issues to Tier 1', '1 Issue to Tier 2']:
            count = (self.df['quick_win_category'] == cat).sum()
            if count > 0:
                print(f"   {cat}: {count:,}")
        
        # Sites
        print(f"\n🏥 SITE PERFORMANCE:")
        print(f"   Total Sites: {len(self.site_df):,}")
        print(f"   Average SPI: {self.site_df['site_performance_index'].mean():.1f}")
        
        for tier in ['Exceptional', 'Strong', 'Adequate', 'Needs Improvement', 'At Risk']:
            count = (self.site_df['performance_tier'] == tier).sum()
            if count > 0:
                pct = count / len(self.site_df) * 100
                print(f"   {tier}: {count} ({pct:.1f}%)")
        
        print(f"\n📁 Output Directory: {self.output_dir}")


def main():
    """Main entry point"""
    
    project_root = Path(__file__).parent.parent.parent
    
    # Input now comes from DB
    input_path = Path("DB")
    output_dir = project_root / 'data' / 'processed' / 'metrics'
    
    engine = MetricsEngine(input_path, output_dir)
    engine.load_data()
    engine.run_all_calculations()
    engine.save_outputs()
    engine.print_summary()
    
    return engine


if __name__ == '__main__':
    main()
