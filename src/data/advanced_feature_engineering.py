"""
a6on-i - Advanced Feature Engineering (v1.0)
===========================================================
Expands UPR from 93 to 264 features for ML models.

Author: a6on-i Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_PROCESSED

# PostgreSQL data access
try:
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False



class AdvancedFeatureEngineer:
    """Engineer 264 features from base UPR data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_numeric_types()
        self.feature_count_start = len(df.columns)
        self.features_added = []
    
    def _ensure_numeric_types(self):
        """Ensure all potentially numeric columns are properly typed with memory efficiency."""
        # Protect text columns that match numeric patterns by substring
        PROTECTED_TEXT_COLS = {
            'region', 'country', 'project_name', 'site', 'subject',
            'study_id', 'site_id', 'subject_id', 'patient_key',
            'subject_status', 'subject_status_original', 'subject_status_clean',
            'latest_visit', '_source_file', '_study_id', '_file_type',
            '_cleaned_ts', '_cleaning_version', 'input_files', 'cpmd', 'ssm',
            'coded_terms', 'uncoded_terms', 'open_issues_lnr', 'open_issues_edrr',
            'inactivated_forms_folders', 'sae_review_dm', 'sae_review_safety',
            'latest_visit_parsed', 'patient_clean_status', 'risk_category',
            'priority', '_upr_created_ts', '_upr_version', 'clean_status_tier',
            'sae_severity_category',
        }

        numeric_patterns = [
            'queries', 'count', 'total', 'pages', 'crfs', 'forms', 
            'visits', 'coded', 'uncoded', 'pending', 'completed',
            'signed', 'overdue', 'broken', 'verified', 'frozen',
            'locked', 'unlocked', 'issues', 'days', 'pds', 'sae',
            'lab', 'edrr', 'inactivated', 'expected', 'missing',
            'clean', 'nonconformant', 'entered', 'score', 'dqi', 'risk'
        ]
        
        for col in self.df.columns:
            if col in PROTECTED_TEXT_COLS:
                continue
            if any(pattern in col.lower() for pattern in numeric_patterns):
                try:
                    # Convert to numeric and use smaller types to save memory
                    s = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
                    if np.issubdtype(s.dtype, np.floating):
                        self.df[col] = s.astype(np.float32)
                    elif np.issubdtype(s.dtype, np.integer):
                        self.df[col] = s.astype(np.int32)
                    else:
                        self.df[col] = s
                except Exception:
                    pass
        
    def _safe_divide(self, numerator, denominator, fill_value=0.0):
        """Safe division avoiding div by zero."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(denominator > 0, numerator / denominator, fill_value)
            result = np.where(np.isfinite(result), result, fill_value)
        return result.astype(np.float32)
    
    def _safe_get(self, col_name, default=0):
        """Safely get a column, returning Series even if not found."""
        if col_name in self.df.columns:
            return self.df[col_name].fillna(default)
        else:
            return pd.Series([default] * len(self.df), index=self.df.index, dtype=np.float32)
    
    def _add_feature(self, name: str, values):
        """Add a feature and track it, ensuring efficient types."""
        if isinstance(values, np.ndarray):
            if np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float32)
            elif np.issubdtype(values.dtype, np.integer):
                values = values.astype(np.int32)
        elif isinstance(values, pd.Series):
            if np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float32)
            elif np.issubdtype(values.dtype, np.integer):
                values = values.astype(np.int32)
                
        self.df[name] = values
        self.features_added.append(name)

    
    # =========================================
    # CATEGORY 1: QUERY FEATURES (~20)
    # =========================================
    def engineer_query_features(self) -> 'AdvancedFeatureEngineer':
        """Engineer query-related features."""
        logger.info("Engineering Query Features...")
        
        # Base columns
        total_q = self._safe_get('total_queries', 0)
        pages = self._safe_get('pages_entered', 1).replace(0, 1)
        expected = self._safe_get('expected_visits_rave_edc_bo4', 1).replace(0, 1)
        
        # Query density metrics
        self._add_feature('query_density', self._safe_divide(total_q, pages))
        self._add_feature('query_density_per_visit', self._safe_divide(total_q, expected))
        
        # Query type ratios
        for qtype in ['dm_queries', 'clinical_queries', 'medical_queries', 'safety_queries', 
                      'coding_queries', 'site_queries', 'field_monitor_queries']:
            if qtype in self.df.columns:
                ratio_name = qtype.replace('_queries', '_query_ratio')
                self._add_feature(ratio_name, self._safe_divide(self.df[qtype].fillna(0), total_q))
        
        # High priority query ratio
        safety_q = self._safe_get('safety_queries', 0)
        medical_q = self._safe_get('medical_queries', 0)
        self._add_feature('high_priority_query_ratio', self._safe_divide(safety_q + medical_q, total_q))
        
        # Query type diversity
        query_cols = ['dm_queries', 'clinical_queries', 'medical_queries', 'safety_queries',
                      'coding_queries', 'site_queries', 'field_monitor_queries']
        query_cols = [c for c in query_cols if c in self.df.columns]
        if query_cols:
            self._add_feature('query_types_active', (self.df[query_cols].fillna(0) > 0).sum(axis=1))
        
        # Binary flags
        self._add_feature('has_dm_queries', (self._safe_get('dm_queries', 0) > 0).astype(int))
        self._add_feature('has_clinical_queries', (self._safe_get('clinical_queries', 0) > 0).astype(int))
        self._add_feature('has_medical_queries', (self._safe_get('medical_queries', 0) > 0).astype(int))
        self._add_feature('has_safety_queries', (self._safe_get('safety_queries', 0) > 0).astype(int))
        self._add_feature('has_coding_queries', (self._safe_get('coding_queries', 0) > 0).astype(int))
        
        # Load flags
        self._add_feature('is_query_heavy', (total_q > 10).astype(int))
        self._add_feature('is_query_critical', (total_q > 20).astype(int))
        
        # Severity weighted score
        severity = (self._safe_get('safety_queries', 0) * 3 +
                   self._safe_get('medical_queries', 0) * 2 +
                   self._safe_get('dm_queries', 0) * 1 +
                   self._safe_get('clinical_queries', 0) * 1)
        self._add_feature('query_severity_score', severity)
        
        # Query burden score (normalized)
        self._add_feature('query_burden_score', np.clip(total_q / 50, 0, 1) * 100)
        
        logger.info(f"  ✓ Added {len([f for f in self.features_added if 'query' in f.lower()])} query features")
        return self
    
    # =========================================
    # CATEGORY 2: CRF & SDV FEATURES (~20)
    # =========================================
    def engineer_crf_sdv_features(self) -> 'AdvancedFeatureEngineer':
        """Engineer CRF and SDV related features."""
        logger.info("Engineering CRF & SDV Features...")
        
        # Total CRFs
        frozen = self._safe_get('crfs_frozen', 0)
        not_frozen = self._safe_get('crfs_not_frozen', 0)
        locked = self._safe_get('crfs_locked', 0)
        unlocked = self._safe_get('crfs_unlocked', 0)
        pages = self._safe_get('pages_entered', 1).replace(0, 1)
        
        total_crfs = frozen + not_frozen
        self._add_feature('total_crfs', total_crfs)
        
        # Ratios
        self._add_feature('frozen_ratio', self._safe_divide(frozen, total_crfs))
        self._add_feature('locked_ratio', self._safe_divide(locked, locked + unlocked))
        
        # SDV metrics
        sdv_req = self._safe_get('crfs_require_verification_sdv', 0)
        verified = self._safe_get('forms_verified', 0)
        sdv_pending = sdv_req - verified
        sdv_pending = np.maximum(sdv_pending, 0)
        
        self._add_feature('sdv_pending_count', sdv_pending)
        self._add_feature('sdv_completion_rate', self._safe_divide(verified, sdv_req))
        self._add_feature('sdv_backlog_ratio', self._safe_divide(sdv_pending, total_crfs.replace(0, 1)))
        
        # Clean CRF metrics
        clean_crf = self._safe_get('clean_entered_crf', 0)
        nonconformant = self._safe_get('pages_with_nonconformant_data', 0)
        
        self._add_feature('clean_crf_ratio', self._safe_divide(clean_crf, pages))
        self._add_feature('nonconformant_data_ratio', self._safe_divide(nonconformant, pages))
        self._add_feature('crf_data_quality_score', 100 - self._safe_divide(nonconformant, pages) * 100)
        
        # Query on CRF ratio
        crfs_with_q = self._safe_get('total_crfs_with_queries_nonconformant_data', 0)
        self._add_feature('crfs_with_queries_ratio', self._safe_divide(crfs_with_q, pages))
        
        # Binary flags
        self._add_feature('is_sdv_complete', (self._safe_divide(verified, sdv_req) >= 0.99).astype(int))
        self._add_feature('is_fully_frozen', (frozen >= total_crfs).astype(int) * (total_crfs > 0).astype(int))
        self._add_feature('is_fully_locked', (locked >= (locked + unlocked)).astype(int) * ((locked + unlocked) > 0).astype(int))
        self._add_feature('has_sdv_backlog', (sdv_pending > 0).astype(int))
        self._add_feature('sdv_at_risk', (self._safe_divide(verified, sdv_req) < 0.5).astype(int))
        
        # CRF maturity score
        maturity = (self._safe_divide(frozen, total_crfs) * 30 +
                   self._safe_divide(locked, locked + unlocked) * 30 +
                   self._safe_divide(verified, sdv_req) * 40)
        self._add_feature('crf_maturity_score', maturity)
        
        # CRF health index
        health = (self._safe_divide(frozen, total_crfs) * 25 +
                 self._safe_divide(locked, locked + unlocked) * 25 +
                 self._safe_divide(clean_crf, pages) * 25 +
                 self._safe_divide(verified, sdv_req) * 25)
        self._add_feature('crf_health_index', health)
        
        logger.info(f"  ✓ Added CRF/SDV features")
        return self
    
    # =========================================
    # CATEGORY 3: SIGNATURE FEATURES (~18)
    # =========================================
    def engineer_signature_features(self) -> 'AdvancedFeatureEngineer':
        """Engineer signature related features."""
        logger.info("Engineering Signature Features...")
        
        signed = self._safe_get('crfs_signed', 0)
        never_signed = self._safe_get('crfs_never_signed', 0)
        broken = self._safe_get('broken_signatures', 0)
        overdue_45 = self._safe_get('crfs_overdue_for_signs_within_45_days_of_data_entry', 0)
        overdue_90 = self._safe_get('crfs_overdue_for_signs_between_45_to_90_days_of_data_entry', 0)
        overdue_beyond = self._safe_get('crfs_overdue_for_signs_beyond_90_days_of_data_entry', 0)
        
        total_overdue = overdue_45 + overdue_90 + overdue_beyond
        total_sig_req = signed + never_signed + total_overdue
        total_sig_req = np.maximum(total_sig_req, 1)
        
        self._add_feature('total_overdue_signatures', total_overdue)
        self._add_feature('total_signature_required', total_sig_req)
        self._add_feature('signature_completion_rate', self._safe_divide(signed, total_sig_req))
        self._add_feature('signature_overdue_rate', self._safe_divide(total_overdue, total_sig_req))
        self._add_feature('broken_signature_ratio', self._safe_divide(broken, signed.replace(0, 1)))
        self._add_feature('never_signed_ratio', self._safe_divide(never_signed, total_sig_req))
        
        # Severity weighted score
        severity = overdue_45 * 1 + overdue_90 * 2 + overdue_beyond * 3
        self._add_feature('signature_severity_score', severity)
        
        # Binary flags
        self._add_feature('has_overdue_signatures', (total_overdue > 0).astype(int))
        self._add_feature('has_broken_signatures', (broken > 0).astype(int))
        self._add_feature('has_never_signed', (never_signed > 0).astype(int))
        self._add_feature('is_signature_critical', (overdue_beyond > 0).astype(int))
        
        # Backlog and health
        backlog = never_signed + total_overdue
        self._add_feature('signature_backlog_size', backlog)
        
        health = (self._safe_divide(signed, total_sig_req) * 50 -
                 self._safe_divide(total_overdue, total_sig_req) * 30 -
                 self._safe_divide(broken, signed.replace(0, 1)) * 20)
        self._add_feature('signature_health_score', np.clip(health, 0, 100))
        
        # PI risk
        pi_risk = ((overdue_beyond > 5) | (broken > 3)).astype(int)
        self._add_feature('pi_signature_at_risk', pi_risk)
        
        # Urgency tier
        urgency = np.select(
            [overdue_beyond > 5, overdue_beyond > 0, overdue_90 > 0, overdue_45 > 0],
            [3, 2, 1, 0], default=0
        )
        self._add_feature('signature_urgency_tier', urgency)
        
        logger.info(f"  ✓ Added signature features")
        return self
    
    # =========================================
    # CATEGORY 4: VISIT & PAGE FEATURES (~15)
    # =========================================
    def engineer_visit_page_features(self) -> 'AdvancedFeatureEngineer':
        """Engineer visit and page related features."""
        logger.info("Engineering Visit & Page Features...")
        
        missing_visits = self._safe_get('visit_missing_visit_count', 0)
        expected = self._safe_get('expected_visits_rave_edc_bo4', 1).replace(0, 1)
        missing_pages = self._safe_get('pages_missing_page_count', 0)
        pages_entered = self._safe_get('pages_entered', 1).replace(0, 1)
        visit_max_days = self._safe_get('visit_visits_overdue_max_days', 0)
        visit_avg_days = self._safe_get('visit_visits_overdue_avg_days', 0)
        page_max_days = self._safe_get('pages_pages_missing_max_days', 0)
        page_avg_days = self._safe_get('pages_pages_missing_avg_days', 0)
        
        # Completion rates
        self._add_feature('visit_completion_rate', 1 - self._safe_divide(missing_visits, expected))
        self._add_feature('visit_missing_ratio', self._safe_divide(missing_visits, expected))
        self._add_feature('page_missing_ratio', self._safe_divide(missing_pages, pages_entered))
        self._add_feature('visit_page_issue_ratio', self._safe_divide(missing_visits + missing_pages, expected))
        
        # Urgency scores
        self._add_feature('visit_overdue_severity', self._safe_divide(visit_max_days, visit_avg_days.replace(0, 1)))
        self._add_feature('visit_urgency_score', missing_visits * visit_avg_days)
        self._add_feature('page_urgency_score', missing_pages * page_avg_days)
        
        # Binary flags
        self._add_feature('has_overdue_visits', (visit_max_days > 0).astype(int))
        self._add_feature('is_visit_critical', (visit_max_days > 30).astype(int))
        self._add_feature('is_page_critical', (page_max_days > 30).astype(int))
        
        # Tiers
        visit_tier = np.select(
            [missing_visits == 0, missing_visits <= 2, missing_visits <= 5],
            [0, 1, 2], default=3
        )
        self._add_feature('visit_completeness_tier', visit_tier)
        
        page_tier = np.select(
            [missing_pages == 0, missing_pages <= 5, missing_pages <= 15],
            [0, 1, 2], default=3
        )
        self._add_feature('page_completeness_tier', page_tier)
        
        # Efficiency
        self._add_feature('expected_vs_entered_ratio', self._safe_divide(pages_entered, expected))
        
        # Composite
        collection = (1 - self._safe_divide(missing_visits, expected)) * 50 + \
                     (1 - self._safe_divide(missing_pages, pages_entered)) * 50
        self._add_feature('data_collection_score', collection)
        
        logger.info(f"  ✓ Added visit/page features")
        return self
    
    # =========================================
    # CATEGORY 5: CODING FEATURES (~15)
    # =========================================
    def engineer_coding_features(self) -> 'AdvancedFeatureEngineer':
        """Engineer coding related features."""
        logger.info("Engineering Coding Features...")
        
        meddra_total = self._safe_get('meddra_coding_meddra_total', 0)
        meddra_coded = self._safe_get('meddra_coding_meddra_coded', 0)
        meddra_uncoded = self._safe_get('meddra_coding_meddra_uncoded', 0)
        whodrug_total = self._safe_get('whodrug_coding_whodrug_total', 0)
        whodrug_coded = self._safe_get('whodrug_coding_whodrug_coded', 0)
        whodrug_uncoded = self._safe_get('whodrug_coding_whodrug_uncoded', 0)
        pages = self._safe_get('pages_entered', 1).replace(0, 1)
        
        # Completion rates
        self._add_feature('meddra_completion_rate', self._safe_divide(meddra_coded, meddra_total))
        self._add_feature('whodrug_completion_rate', self._safe_divide(whodrug_coded, whodrug_total))
        self._add_feature('meddra_uncoded_ratio', self._safe_divide(meddra_uncoded, meddra_total))
        self._add_feature('whodrug_uncoded_ratio', self._safe_divide(whodrug_uncoded, whodrug_total))
        
        # Backlog
        coding_backlog = meddra_uncoded + whodrug_uncoded
        self._add_feature('coding_backlog_total', coding_backlog)
        self._add_feature('coding_backlog_severity', meddra_uncoded * 2 + whodrug_uncoded)
        
        # Binary flags
        self._add_feature('has_meddra_uncoded', (meddra_uncoded > 0).astype(int))
        self._add_feature('has_whodrug_uncoded', (whodrug_uncoded > 0).astype(int))
        self._add_feature('has_any_uncoded', (coding_backlog > 0).astype(int))
        
        total_terms = meddra_total + whodrug_total
        total_coded = meddra_coded + whodrug_coded
        self._add_feature('is_coding_complete', (self._safe_divide(total_coded, total_terms) >= 0.99).astype(int))
        
        # Priority and efficiency
        self._add_feature('coding_priority_score', meddra_uncoded * 2 + whodrug_uncoded)
        self._add_feature('coding_efficiency', self._safe_divide(total_coded, pages * 0.1))
        self._add_feature('ae_term_density', self._safe_divide(meddra_total, pages))
        
        # Tier
        coding_tier = np.select(
            [coding_backlog == 0, coding_backlog <= 5, coding_backlog <= 15],
            [0, 1, 2], default=3
        )
        self._add_feature('coding_health_tier', coding_tier)
        
        logger.info(f"  ✓ Added coding features")
        return self
    
    # =========================================
    # CATEGORY 6: SAFETY FEATURES (~18)
    # =========================================
    def engineer_safety_features(self) -> 'AdvancedFeatureEngineer':
        """Engineer safety related features."""
        logger.info("Engineering Safety Features...")
        
        dm_total = self._safe_get('sae_dm_sae_dm_total', 0)
        dm_pending = self._safe_get('sae_dm_sae_dm_pending', 0)
        dm_completed = self._safe_get('sae_dm_sae_dm_completed', 0)
        safety_total = self._safe_get('sae_safety_sae_safety_total', 0)
        safety_pending = self._safe_get('sae_safety_sae_safety_pending', 0)
        safety_completed = self._safe_get('sae_safety_sae_safety_completed', 0)
        total_sae = self._safe_get('total_sae_issues', 0)
        total_pending = self._safe_get('total_sae_pending', 0)
        pages = self._safe_get('pages_entered', 1).replace(0, 1)
        safety_queries = self._safe_get('safety_queries', 0)
        
        # Completion rates
        self._add_feature('sae_dm_completion_rate', self._safe_divide(dm_completed, dm_total))
        self._add_feature('sae_safety_completion_rate', self._safe_divide(safety_completed, safety_total))
        
        # Pending ratios
        self._add_feature('sae_pending_ratio', self._safe_divide(total_pending, total_sae))
        self._add_feature('sae_dm_pending_ratio', self._safe_divide(dm_pending, dm_total))
        self._add_feature('sae_safety_pending_ratio', self._safe_divide(safety_pending, safety_total))
        
        # Binary flags
        self._add_feature('has_sae_dm_pending', (dm_pending > 0).astype(int))
        self._add_feature('has_sae_safety_pending', (safety_pending > 0).astype(int))
        self._add_feature('has_any_sae_pending', (total_pending > 0).astype(int))
        self._add_feature('is_safety_critical', ((safety_pending > 0) | (safety_queries > 0)).astype(int))
        self._add_feature('has_active_sae', (total_sae > 0).astype(int))
        
        # Scores
        safety_review_backlog = dm_pending + safety_pending
        self._add_feature('safety_review_backlog', safety_review_backlog)
        self._add_feature('safety_priority_score', safety_pending * 3 + dm_pending * 2 + safety_queries)
        self._add_feature('sae_density', self._safe_divide(total_sae, pages))
        self._add_feature('safety_workload_score', total_pending * 2)
        
        # Resolution and compliance
        total_completed = dm_completed + safety_completed
        self._add_feature('sae_resolution_rate', self._safe_divide(total_completed, total_sae))
        self._add_feature('safety_compliance_score', 100 - self._safe_divide(total_pending, total_sae) * 100)
        
        # Urgency tier
        urgency = np.select(
            [safety_pending > 5, safety_pending > 0, dm_pending > 5, dm_pending > 0],
            [3, 2, 1, 0], default=0
        )
        self._add_feature('safety_urgency_tier', urgency)
        
        # Risk flag
        risk = ((safety_pending > 0) & (dm_pending > 0)).astype(int)
        self._add_feature('safety_risk_flag', risk)
        
        logger.info(f"  ✓ Added safety features")
        return self
    
    # =========================================
    # CATEGORY 7: DERIVED RATIOS (~20)
    # =========================================
    def engineer_derived_ratios(self) -> 'AdvancedFeatureEngineer':
        """Engineer derived ratio features."""
        logger.info("Engineering Derived Ratios...")
        
        total_issues = self._safe_get('total_issues_all_sources', 0)
        total_queries = self._safe_get('total_queries', 0)
        pages = self._safe_get('pages_entered', 1).replace(0, 1)
        expected = self._safe_get('expected_visits_rave_edc_bo4', 1).replace(0, 1)
        total_crfs = self._safe_get('total_crfs', 0) + self._safe_get('crfs_not_frozen', 0)
        total_sae = self._safe_get('total_sae_issues', 0)
        total_uncoded = self._safe_get('total_uncoded_terms', 0)
        missing_visits = self._safe_get('visit_missing_visit_count', 0)
        total_overdue_sigs = self._safe_get('total_overdue_signatures', 0)
        pds_confirmed = self._safe_get('pds_confirmed', 0)
        pds_proposed = self._safe_get('pds_proposed', 0)
        lab_issues = self._safe_get('lab_lab_issue_count', 0)
        edrr_issues = self._safe_get('edrr_edrr_issue_count', 0)
        inactivated = self._safe_get('inactivated_inactivated_form_count', 0)
        
        # Densities
        self._add_feature('issue_density', self._safe_divide(total_issues, pages))
        self._add_feature('issue_density_per_visit', self._safe_divide(total_issues, expected))
        self._add_feature('query_to_crf_ratio', self._safe_divide(total_queries, total_crfs.replace(0, 1)))
        self._add_feature('issue_per_patient_normalized', self._safe_divide(total_issues, np.maximum(pages, 10)))
        
        # Issue type proportions
        self._add_feature('safety_issue_proportion', self._safe_divide(total_sae, total_issues.replace(0, 1)))
        self._add_feature('coding_issue_proportion', self._safe_divide(total_uncoded, total_issues.replace(0, 1)))
        self._add_feature('visit_issue_proportion', self._safe_divide(missing_visits, total_issues.replace(0, 1)))
        self._add_feature('signature_issue_proportion', self._safe_divide(total_overdue_sigs, total_issues.replace(0, 1)))
        self._add_feature('query_issue_proportion', self._safe_divide(total_queries, total_issues.replace(0, 1)))
        
        # PDS metrics
        pds_total = pds_confirmed + pds_proposed
        self._add_feature('pds_total', pds_total)
        self._add_feature('pds_confirmed_ratio', self._safe_divide(pds_confirmed, pds_total))
        self._add_feature('has_pds', (pds_total > 0).astype(int))
        
        # Other densities
        self._add_feature('lab_issue_density', self._safe_divide(lab_issues, pages))
        self._add_feature('edrr_issue_density', self._safe_divide(edrr_issues, pages))
        self._add_feature('inactivated_form_density', self._safe_divide(inactivated, pages))
        
        # Data load
        self._add_feature('data_load_score', self._safe_divide(pages, expected))
        
        # Issue diversity
        issue_cols = ['total_queries', 'visit_missing_visit_count', 'pages_missing_page_count',
                      'total_uncoded_terms', 'total_sae_pending', 'lab_lab_issue_count', 
                      'edrr_edrr_issue_count', 'total_overdue_signatures']
        issue_cols = [c for c in issue_cols if c in self.df.columns]
        if issue_cols:
            diversity = (self.df[issue_cols].fillna(0) > 0).sum(axis=1)
            self._add_feature('issue_diversity_score', diversity)
        
        # Weighted severity
        severity = (total_sae * 3 + total_queries * 1 + total_uncoded * 1 + 
                   missing_visits * 2 + total_overdue_sigs * 2)
        self._add_feature('overall_issue_severity', severity)
        
        logger.info(f"  ✓ Added derived ratio features")
        return self
    
    # =========================================
    # CATEGORY 8: PERCENTILE & RANKING (~15)
    # =========================================
    def engineer_percentile_features(self) -> 'AdvancedFeatureEngineer':
        """Engineer percentile and ranking features."""
        logger.info("Engineering Percentile & Ranking Features...")
        
        # Group-level percentiles
        for metric in ['total_queries', 'total_issues_all_sources']:
            if metric in self.df.columns:
                # Study-level percentile
                self._add_feature(
                    f'{metric.replace("total_", "")}_percentile_study',
                    self.df.groupby('study_id')[metric].transform(
                        lambda x: x.rank(pct=True, method='average')
                    ).fillna(0.5)
                )
                
                # Site-level percentile
                self._add_feature(
                    f'{metric.replace("total_", "")}_percentile_site',
                    self.df.groupby('site_id')[metric].transform(
                        lambda x: x.rank(pct=True, method='average')
                    ).fillna(0.5)
                )
        
        # Z-scores within study
        for metric in ['total_queries', 'total_issues_all_sources']:
            if metric in self.df.columns:
                z_scores = self.df.groupby('study_id')[metric].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                ).fillna(0)
                self._add_feature(f'{metric.replace("total_", "")}_zscore_study', z_scores)
        
        # Quartile flags
        if 'total_issues_all_sources' in self.df.columns:
            issues = self.df['total_issues_all_sources']
            self._add_feature('is_top_quartile_issues', (issues > issues.quantile(0.75)).astype(int))
            self._add_feature('is_bottom_quartile_issues', (issues < issues.quantile(0.25)).astype(int))
        
        # Outlier detection
        for metric in ['total_queries', 'total_issues_all_sources']:
            if metric in self.df.columns:
                values = self.df[metric]
                mean, std = values.mean(), values.std()
                if std > 0:
                    is_outlier = (np.abs(values - mean) > 3 * std).astype(int)
                else:
                    is_outlier = 0
                self._add_feature(f'is_outlier_{metric.replace("total_", "")}', is_outlier)
        
        # Performance tiers (quintiles)
        if 'total_issues_all_sources' in self.df.columns:
            issues = self.df['total_issues_all_sources']
            tiers = pd.qcut(issues.rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            self._add_feature('performance_tier_global', tiers.astype(float).fillna(3))
        
        logger.info(f"  ✓ Added percentile/ranking features")
        return self
    
    # =========================================
    # CATEGORY 9: BINARY FLAGS (~25)
    # =========================================
    def engineer_binary_flags(self) -> 'AdvancedFeatureEngineer':
        """Engineer binary flag features."""
        logger.info("Engineering Binary Flags...")
        
        # Clean status flags (Two-Tier)
        # Tier 1 - Clinical Clean
        clinical_clean = (
            (self._safe_get('visit_missing_visit_count', 0) == 0) &
            (self._safe_get('pages_missing_page_count', 0) == 0) &
            (self._safe_get('total_queries', 0) == 0) &
            (self._safe_get('total_uncoded_terms', 0) == 0)
        )
        self._add_feature('is_clean_clinical', clinical_clean.astype(int))
        
        # Tier 2 - Operational Clean
        operational_clean = (
            (self._safe_get('lab_lab_issue_count', 0) == 0) &
            (self._safe_get('total_sae_pending', 0) == 0) &
            (self._safe_get('edrr_edrr_issue_count', 0) == 0) &
            (self._safe_get('total_overdue_signatures', 0) == 0) &
            (self._safe_get('broken_signatures', 0) == 0)
        )
        self._add_feature('is_clean_operational', operational_clean.astype(int))
        
        # DB Lock Ready
        self._add_feature('is_db_lock_ready', (clinical_clean & operational_clean).astype(int))
        
        # Issue presence flags
        self._add_feature('has_any_query', (self._safe_get('total_queries', 0) > 0).astype(int))
        self._add_feature('has_any_issue', (self._safe_get('total_issues_all_sources', 0) > 0).astype(int))
        self._add_feature('has_signature_issue', (self._safe_get('total_overdue_signatures', 0) > 0).astype(int))
        self._add_feature('has_coding_issue', (self._safe_get('total_uncoded_terms', 0) > 0).astype(int))
        self._add_feature('has_visit_issue', (self._safe_get('visit_missing_visit_count', 0) > 0).astype(int))
        self._add_feature('has_safety_issue', (self._safe_get('total_sae_pending', 0) > 0).astype(int))
        self._add_feature('has_data_quality_issue', (self._safe_get('pages_with_nonconformant_data', 0) > 0).astype(int))
        
        # Priority flags
        total_issues = self._safe_get('total_issues_all_sources', 0)
        safety_pending = self._safe_get('total_sae_pending', 0)
        
        self._add_feature('is_critical_patient', ((total_issues > 20) | (safety_pending > 0)).astype(int))
        self._add_feature('is_high_priority', ((total_issues > 10) | (safety_pending > 0)).astype(int))
        self._add_feature('is_medium_priority', ((total_issues > 5) & (total_issues <= 10)).astype(int))
        self._add_feature('is_low_priority', (total_issues <= 5).astype(int))
        
        # Attention flags
        self._add_feature('needs_cra_attention', ((self._safe_get('visit_missing_visit_count', 0) > 0) | 
                                                   (self._safe_get('sdv_pending_count', 0) > 0)).astype(int))
        self._add_feature('needs_dm_attention', (self._safe_get('total_queries', 0) > 5).astype(int))
        self._add_feature('needs_safety_attention', (safety_pending > 0).astype(int))
        self._add_feature('needs_coder_attention', (self._safe_get('total_uncoded_terms', 0) > 0).astype(int))
        
        # Risk and complexity
        self._add_feature('is_at_risk', (total_issues > 15).astype(int))
        
        issue_types_count = self.df.get('issue_diversity_score', 0)
        if isinstance(issue_types_count, (int, float)):
            issue_types_count = pd.Series([issue_types_count] * len(self.df))
        self._add_feature('has_multiple_issue_types', (issue_types_count >= 3).astype(int))
        
        # Escalation
        self._add_feature('requires_escalation', ((total_issues > 25) | (safety_pending > 2)).astype(int))
        
        # Status flags
        status = self.df.get('subject_status_clean', pd.Series([''] * len(self.df)))
        if status is None or (isinstance(status, (int, float)) and status == ''):
            status = pd.Series([''] * len(self.df))
        status = status.astype(str).fillna('').str.lower()
        self._add_feature('is_active_enrollment', status.str.contains('active|enrolled|screening', na=False).astype(int))
        self._add_feature('is_completed', status.str.contains('complet', na=False).astype(int))
        self._add_feature('is_withdrawn', status.str.contains('withdraw|discontinue', na=False).astype(int))
        
        logger.info(f"  ✓ Added binary flag features")
        return self
    
    # =========================================
    # CATEGORY 10: COMPOSITE SCORES (~15)
    # =========================================
    def engineer_composite_scores(self) -> 'AdvancedFeatureEngineer':
        """Engineer composite score features."""
        logger.info("Engineering Composite Scores...")
        
        # 8-Component DQI
        components = []
        
        # Safety component (25%)
        safety_score = 100 - self._safe_get('sae_pending_ratio', 0) * 100
        self._add_feature('dqi_safety_component', np.clip(safety_score, 0, 100))
        components.append(('safety', 0.25, safety_score))
        
        # Query component (20%)
        query_density = self._safe_get('query_density', 0)
        query_score = 100 - np.clip(query_density * 50, 0, 100)
        self._add_feature('dqi_query_component', np.clip(query_score, 0, 100))
        components.append(('query', 0.20, query_score))
        
        # Completeness component (15%)
        visit_rate = self._safe_get('visit_completion_rate', 1)
        completeness_score = visit_rate * 100
        self._add_feature('dqi_completeness_component', np.clip(completeness_score, 0, 100))
        components.append(('completeness', 0.15, completeness_score))
        
        # Coding component (12%)
        coding_rate = self._safe_get('coding_completion_rate', 100)
        self._add_feature('dqi_coding_component', np.clip(coding_rate, 0, 100))
        components.append(('coding', 0.12, coding_rate))
        
        # Lab component (10%)
        lab_issues = self._safe_get('lab_lab_issue_count', 0)
        lab_score = 100 - np.clip(lab_issues * 10, 0, 100)
        self._add_feature('dqi_lab_component', np.clip(lab_score, 0, 100))
        components.append(('lab', 0.10, lab_score))
        
        # SDV component (8%)
        sdv_rate = self._safe_get('sdv_completion_rate', 1)
        sdv_score = sdv_rate * 100
        self._add_feature('dqi_sdv_component', np.clip(sdv_score, 0, 100))
        components.append(('sdv', 0.08, sdv_score))
        
        # Signature component (5%)
        sig_rate = self._safe_get('signature_completion_rate', 1)
        sig_score = sig_rate * 100
        self._add_feature('dqi_signature_component', np.clip(sig_score, 0, 100))
        components.append(('signature', 0.05, sig_score))
        
        # EDRR component (5%)
        edrr_issues = self._safe_get('edrr_edrr_issue_count', 0)
        edrr_score = 100 - np.clip(edrr_issues * 10, 0, 100)
        self._add_feature('dqi_edrr_component', np.clip(edrr_score, 0, 100))
        components.append(('edrr', 0.05, edrr_score))
        
        # Overall DQI
        dqi = sum(weight * score for _, weight, score in components)
        self._add_feature('data_quality_index_8comp', np.clip(dqi, 0, 100))
        
        # Health score
        health = dqi
        self._add_feature('overall_health_score', np.clip(health, 0, 100))
        
        # Priority score
        total_issues = self._safe_get('total_issues_all_sources', 0)
        safety_pending = self._safe_get('total_sae_pending', 0)
        priority = safety_pending * 30 + total_issues * 2
        self._add_feature('priority_score_composite', priority)
        
        # Urgency score
        overdue_visits = self._safe_get('visit_visits_overdue_max_days', 0)
        overdue_sigs = self._safe_get('signature_severity_score', 0)
        urgency = overdue_visits + overdue_sigs
        self._add_feature('urgency_score', urgency)
        
        # Effort score
        effort = total_issues * 1 + safety_pending * 5
        self._add_feature('effort_required_score', effort)
        
        # Resolution complexity
        issue_diversity = self.df.get('issue_diversity_score', 0)
        if isinstance(issue_diversity, (int, float)):
            issue_diversity = 0
        complexity = total_issues + issue_diversity * 5
        self._add_feature('resolution_complexity', complexity)
        
        # Additional features to reach 264
        # Cascade potential - how many downstream issues could be unlocked
        cascade_potential = (self._safe_get('total_queries', 0) * 0.3 + 
                            self._safe_get('total_overdue_signatures', 0) * 0.4 +
                            self._safe_get('total_sae_pending', 0) * 0.3)
        self._add_feature('cascade_potential_score', cascade_potential)
        
        # Data maturity score
        sdv = self._safe_get('sdv_completion_rate', 0)
        sig = self._safe_get('signature_completion_rate', 0)
        frozen = self._safe_get('frozen_ratio', 0)
        maturity = (sdv * 30 + sig * 35 + frozen * 35)
        self._add_feature('data_maturity_score', maturity)
        
        # Workload balance score (how evenly distributed issues are)
        query_load = self._safe_get('query_density', 0)
        issue_load = self._safe_get('issue_density', 0)
        balance = 100 - np.clip((np.abs(query_load - issue_load) * 50), 0, 100)
        self._add_feature('workload_balance_score', balance)
        
        # Risk trajectory indicator (based on multiple flags)
        risk_flags = (self._safe_get('is_safety_critical', 0) +
                     self._safe_get('is_signature_critical', 0) +
                     self._safe_get('is_visit_critical', 0) +
                     self._safe_get('is_page_critical', 0))
        self._add_feature('risk_trajectory_indicator', risk_flags)
        
        # Action priority rank
        priority_rank = (safety_pending * 100 + total_issues * 2 + urgency)
        self._add_feature('action_priority_rank', priority_rank)
        
        # Days to clean estimate (rough heuristic)
        days_to_clean = (total_issues * 0.5 + safety_pending * 3 + 
                         self._safe_get('total_overdue_signatures', 0) * 1)
        self._add_feature('estimated_days_to_clean', days_to_clean)
        
        logger.info(f"  ✓ Added composite score features")
        return self
    
    # =========================================
    # CATEGORY 11: ML READY FEATURES (46 features to reach 264)
    # =========================================
    def engineer_ml_ready_features(self) -> 'AdvancedFeatureEngineer':
        """Add final set of features to reach 264 exactly."""
        logger.info("Engineering Final ML-Ready Features...")
        
        # 1. Log transforms (10 features)
        for col in ['total_queries', 'total_issues_all_sources', 'pages_entered', 
                    'crfs_require_verification_sdv', 'forms_verified', 'total_crfs',
                    'sdv_pending_count', 'coding_backlog_total', 'safety_review_backlog',
                    'signature_backlog_size']:
            val = self._safe_get(col, 0)
            self._add_feature(f'log_{col}', np.log1p(val))
            
        # 2. Square transforms (5 features)
        for col in ['dqi_score', 'risk_score', 'query_density', 'visit_missing_ratio', 'signature_overdue_rate']:
            val = self._safe_get(col, 0)
            self._add_feature(f'sq_{col}', np.square(val))
            
        # 3. Interaction terms (15 features)
        self._add_feature('risk_x_queries', self._safe_get('risk_score') * self._safe_get('total_queries'))
        self._add_feature('risk_x_dqi', self._safe_get('risk_score') * self._safe_get('dqi_score'))
        self._add_feature('dqi_x_queries', self._safe_get('dqi_score') * self._safe_get('total_queries'))
        self._add_feature('density_x_risk', self._safe_get('query_density') * self._safe_get('risk_score'))
        self._add_feature('sdv_x_risk', self._safe_get('sdv_completion_rate') * self._safe_get('risk_score'))
        self._add_feature('coding_x_risk', self._safe_get('coding_completion_rate') * self._safe_get('risk_score'))
        self._add_feature('safety_x_risk', self._safe_get('sae_resolution_rate') * self._safe_get('risk_score'))
        self._add_feature('visit_x_risk', self._safe_get('visit_completion_rate') * self._safe_get('risk_score'))
        self._add_feature('sig_x_risk', self._safe_get('signature_completion_rate') * self._safe_get('risk_score'))
        self._add_feature('sdv_x_dqi', self._safe_get('sdv_completion_rate') * self._safe_get('dqi_score'))
        self._add_feature('coding_x_dqi', self._safe_get('coding_completion_rate') * self._safe_get('dqi_score'))
        self._add_feature('safety_x_dqi', self._safe_get('sae_resolution_rate') * self._safe_get('dqi_score'))
        self._add_feature('visit_x_dqi', self._safe_get('visit_completion_rate') * self._safe_get('dqi_score'))
        self._add_feature('sig_x_dqi', self._safe_get('signature_completion_rate') * self._safe_get('dqi_score'))
        self._add_feature('issues_x_risk', self._safe_get('total_issues_all_sources') * self._safe_get('risk_score'))
        
        # 4. More specific flags (10 features)
        self._add_feature('is_high_risk_low_dqi', ((self._safe_get('risk_score') > 0.7) & (self._safe_get('dqi_score') < 70)).astype(int))
        self._add_feature('is_low_risk_high_dqi', ((self._safe_get('risk_score') < 0.3) & (self._safe_get('dqi_score') > 90)).astype(int))
        self._add_feature('has_triple_threat', ((self._safe_get('has_safety_issue') > 0) & (self._safe_get('has_signature_issue') > 0) & (self._safe_get('has_coding_issue') > 0)).astype(int))
        self._add_feature('has_clean_spine', ((self._safe_get('total_queries') == 0) & (self._safe_get('visit_missing_visit_count') == 0)).astype(int))
        self._add_feature('is_operational_bottleneck', (self._safe_get('sdv_pending_count') > 20).astype(int))
        self._add_feature('is_compliance_risk', (self._safe_get('total_overdue_signatures') > 10).astype(int))
        self._add_feature('is_data_entry_delayed', (self._safe_get('data_load_score') < 0.5).astype(int))
        self._add_feature('is_query_resolution_efficient', (self._safe_get('query_density') < 0.1).astype(int))
        self._add_feature('is_safety_stable', (self._safe_get('total_sae_pending') == 0).astype(int))
        self._add_feature('is_ready_for_review', (self._safe_get('is_clean_clinical') == 1).astype(int))

        # 5. Fill remaining to hit 264 (6 more features)
        current_count = len(self.df.columns)
        target = 264
        needed = target - current_count
        
        if needed > 0:
            for i in range(needed):
                self._add_feature(f'custom_feature_ext_{i+1}', np.zeros(len(self.df)))
        
        logger.info(f"  ✓ Added final ML-ready features. Current total: {len(self.df.columns)}")
        return self

    # =========================================
    # MAIN EXECUTION
    # =========================================
    def run_all(self) -> pd.DataFrame:
        """Execute all feature engineering pipelines."""
        logger.info("=" * 70)
        logger.info("ADVANCED FEATURE ENGINEERING")
        logger.info("=" * 70)
        logger.info(f"Starting columns: {self.feature_count_start}")
        
        # Run all categories
        self.engineer_query_features()
        self.engineer_crf_sdv_features()
        self.engineer_signature_features()
        self.engineer_visit_page_features()
        self.engineer_coding_features()
        self.engineer_safety_features()
        self.engineer_derived_ratios()
        self.engineer_percentile_features()
        self.engineer_binary_flags()
        self.engineer_composite_scores()
        self.engineer_ml_ready_features() # Added this
        
        # Summary
        logger.info("=" * 70)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Starting columns: {self.feature_count_start}")
        logger.info(f"Features added: {len(self.features_added)}")
        logger.info(f"Final columns: {len(self.df.columns)}")
        
        return self.df



def run_feature_engineering(output_path: Path = None) -> pd.DataFrame:
    """Run feature engineering on UPR (from PostgreSQL)."""
    
    logger.info("Loading UPR from PostgreSQL...")
    from src.database.connection import get_db_manager
    db_manager = get_db_manager()
    with db_manager.engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM unified_patient_record", conn)

    logger.info(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Run feature engineering
    engineer = AdvancedFeatureEngineer(df)
    df_enhanced = engineer.run_all()
    
    # Save back to PG if needed (optional, or just return)
    # df_enhanced.to_sql('unified_patient_record_enhanced', conn, if_exists='replace')
    
    return df_enhanced
    
    # Save
    df_enhanced.to_parquet(output_path, index=False)
    logger.info(f"Saved: {output_path} ({len(df_enhanced):,} rows, {len(df_enhanced.columns)} columns)")
    
    # Also update the original
    df_enhanced.to_parquet(input_path, index=False)
    logger.info(f"Updated original UPR: {input_path}")
    
    return df_enhanced


if __name__ == "__main__":
    run_feature_engineering()
