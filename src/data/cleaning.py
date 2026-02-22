"""
a6on-i - Data Cleaning & Standardization Engine (v2.0 FIXED)
==========================================================================
Properly handles CPID EDC Metrics with unnamed columns.

Author: a6on-i Team
Version: 2.0.0
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import traceback
import warnings

import pandas as pd
import numpy as np
from loguru import logger

warnings.filterwarnings('ignore', category=UserWarning)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_PROCESSED, LOGS_DIR


# ============================================
# CONSTANTS
# ============================================

CLEANED_DATA_DIR = DATA_PROCESSED / "cleaned"
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# CPID EDC Metrics column mapping (unnamed_X -> meaningful names)
# Based on the data description provided
CPID_COLUMN_MAPPING = {
    'unnamed_0': 'project_name',
    'unnamed_1': 'region', 
    'unnamed_2': 'country',
    'unnamed_3': 'site',
    'unnamed_4': 'subject',
    'unnamed_5': 'latest_visit',
    'unnamed_6': 'subject_status',
    'unnamed_7': 'input_files',
    'unnamed_8': 'cpmd',
    'unnamed_9': 'ssm',
    'unnamed_10': 'missing_visits',
    'unnamed_11': 'missing_pages',
    'unnamed_12': 'coded_terms',
    'unnamed_13': 'uncoded_terms',
    'unnamed_14': 'open_issues_lnr',
    'unnamed_15': 'open_issues_edrr',
    'unnamed_16': 'inactivated_forms_folders',
    'unnamed_17': 'sae_review_dm',
    'unnamed_18': 'sae_review_safety',
}

SUBJECT_STATUS_MAPPING = {
    'screen failure': 'Screen Failure',
    'screen fail': 'Screen Failure',
    'screening failure': 'Screen Failure',
    'discontinued': 'Discontinued',
    'discontinue': 'Discontinued',
    'withdrawn': 'Discontinued',
    'dropout': 'Discontinued',
    'drop out': 'Discontinued',
    'completed': 'Completed',
    'complete': 'Completed',
    'finished': 'Completed',
    'ongoing': 'Ongoing',
    'active': 'Ongoing',
    'enrolled': 'Ongoing',
    'in progress': 'Ongoing',
    'on study': 'Ongoing',
    'on treatment': 'Ongoing',
    'survival': 'Ongoing',
    'survival follow-up': 'Ongoing',
    'follow-up': 'Ongoing',
    'follow up': 'Ongoing',
    'screening': 'Screening',
}

NUMERIC_COLUMNS = [
    'missing_visits', 'missing_pages', 'coded_terms', 'uncoded_terms',
    'open_issues_lnr', 'open_issues_edrr', 'inactivated_forms_folders',
    'sae_review_dm', 'sae_review_safety', 'expected_visits_rave_edc_bo4',
    'pages_entered', 'pages_with_nonconformant_data',
    'total_crfs_with_queries_nonconformant_data',
    'total_crfs_without_queries_nonconformant_data',
    'dm_queries', 'clinical_queries', 'medical_queries', 'site_queries',
    'field_monitor_queries', 'coding_queries', 'safety_queries', 'total_queries',
    'crfs_require_verification_sdv', 'forms_verified',
    'crfs_frozen', 'crfs_not_frozen', 'crfs_locked', 'crfs_unlocked',
    'pds_confirmed', 'pds_proposed', 'crfs_signed', 'crfs_never_signed',
    'broken_signatures',
    'crfs_overdue_for_signs_within_45_days_of_data_entry',
    'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
    'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
]


# ============================================
# LOGGING
# ============================================

def setup_logger() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"cleaning_{timestamp}.log"
    
    logger.remove()
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", level="DEBUG", rotation="10 MB")
    logger.add(lambda msg: print(msg, end=""), format="<level>{level:<8}</level> | {message}\n", level="INFO", colorize=True)
    
    return log_file


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class CleaningStats:
    table_name: str
    input_rows: int
    output_rows: int
    duplicates_removed: int = 0
    nulls_filled: Dict[str, int] = field(default_factory=dict)
    type_conversions: Dict[str, str] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0


@dataclass
class CleaningManifest:
    run_id: str
    start_time: str
    end_time: str = ""
    status: str = "running"
    schema_version: str = "2.0.0"
    tables_processed: int = 0
    total_input_rows: int = 0
    total_output_rows: int = 0
    total_duplicates_removed: int = 0
    table_stats: Dict[str, Dict] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').fillna(default)


def standardize_study_id(value: Any) -> str:
    if pd.isna(value) or str(value).lower() in ['none', 'nan', '']:
        return 'Unknown'
    value_str = str(value).strip()
    match = re.search(r'(\d+)', value_str)
    if match:
        return f"Study_{match.group(1)}"
    return value_str


def standardize_site_id(value: Any) -> Optional[str]:
    if pd.isna(value) or str(value).lower() in ['none', 'nan', '']:
        return None
    value_str = str(value).strip()
    if not value_str:
        return None
    # Remove 'Site' prefix if present
    clean = re.sub(r'^site[\s_-]*', '', value_str, flags=re.IGNORECASE)
    if clean:
        return f"Site_{clean}"
    return None


def standardize_subject_id(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    value_str = str(value).strip()
    if value_str.lower() in ['none', 'nan', '', 'null']:
        return None
    # Remove 'Subject' prefix if present
    clean = re.sub(r'^(subject|subj)[\s_-]*', '', value_str, flags=re.IGNORECASE)
    if clean:
        return f"Subject_{clean}"
    return None


def standardize_subject_status(value: Any) -> str:
    if pd.isna(value) or str(value).lower() in ['none', 'nan', '']:
        return 'Unknown'
    value_lower = str(value).lower().strip()
    
    # Direct lookup
    if value_lower in SUBJECT_STATUS_MAPPING:
        return SUBJECT_STATUS_MAPPING[value_lower]
    
    # Partial matching
    for pattern, canonical in SUBJECT_STATUS_MAPPING.items():
        if pattern in value_lower:
            return canonical
    
    return 'Unknown'


def create_patient_key(study_id: str, site_id: Optional[str], subject_id: Optional[str]) -> str:
    site = site_id if site_id else 'Unknown'
    subject = subject_id if subject_id else 'Unknown'
    return f"{study_id}|{site}|{subject}"


def get_column_if_exists(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
        # Case insensitive
        for c in df.columns:
            if c.lower() == col.lower():
                return c
    return None


def coalesce_columns(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    result = pd.Series([None] * len(df), index=df.index)
    for col in candidates:
        if col in df.columns:
            mask = result.isna() | (result.astype(str).isin(['None', '', 'nan']))
            result = result.where(~mask, df[col])
    return result


# ============================================
# CPID EDC METRICS CLEANER (FIXED)
# ============================================

class CPIDEDCMetricsCleaner:
    """Clean CPID EDC Metrics with proper column mapping."""
    
    JUNK_PATTERNS = ['responsible', 'lf for action', 'site/cra', 'coder', 
                     'safety team', 'investigator', 'cse/cdd', 'cdmd']
    
    def __init__(self):
        self.stats = CleaningStats(table_name='cpid_edc_metrics', input_rows=0, output_rows=0)
    
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningStats]:
        start_time = datetime.now()
        self.stats.input_rows = len(df)
        
        logger.info(f"Cleaning CPID EDC Metrics: {len(df):,} rows")
        logger.info(f"  Original columns: {list(df.columns)[:15]}...")
        
        # Step 1: Map unnamed columns
        df = self._map_columns(df)
        
        # Step 2: Remove junk rows
        df = self._remove_junk_rows(df)
        
        # Step 3: Remove invalid subjects
        df = self._remove_invalid_subjects(df)
        
        # Step 4: Standardize IDs
        df = self._standardize_ids(df)
        
        # Step 5: Standardize status
        df = self._standardize_status(df)
        
        # Step 6: Convert numeric
        df = self._convert_numeric(df)
        
        # Step 7: Create patient key
        df = self._create_patient_key(df)
        
        # Step 8: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Add metadata
        df['_cleaned_ts'] = datetime.now().isoformat()
        df['_cleaning_version'] = '2.0.0'
        
        self.stats.output_rows = len(df)
        self.stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"  ✓ Cleaned: {self.stats.input_rows:,} -> {self.stats.output_rows:,} rows")
        logger.info(f"  ✓ Unique patient_keys: {df['patient_key'].nunique():,}")
        
        return df, self.stats
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map unnamed columns to meaningful names."""
        df = df.copy()
        
        # Apply mapping
        rename_map = {}
        for old_name, new_name in CPID_COLUMN_MAPPING.items():
            if old_name in df.columns:
                rename_map[old_name] = new_name
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"  Mapped {len(rename_map)} columns")
        
        logger.info(f"  Columns after mapping: {list(df.columns)[:15]}...")
        
        return df
    
    def _remove_junk_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        
        # Check first few columns for junk patterns
        check_cols = ['project_name', 'region', 'country', 'site', 'subject']
        check_cols = [c for c in check_cols if c in df.columns]
        
        if not check_cols:
            check_cols = list(df.columns)[:5]
        
        def is_junk(row):
            for col in check_cols:
                val = row.get(col)
                if pd.notna(val):
                    val_lower = str(val).lower().strip()
                    if any(p in val_lower for p in self.JUNK_PATTERNS):
                        return True
            return False
        
        mask = ~df.apply(is_junk, axis=1)
        df = df[mask].reset_index(drop=True)
        
        removed = initial - len(df)
        if removed > 0:
            logger.info(f"  Removed {removed:,} junk rows")
        
        return df
    
    def _remove_invalid_subjects(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        
        if 'subject' in df.columns:
            logger.info(f"  Subject column sample: {df['subject'].head(5).tolist()}")
            
            valid = (
                df['subject'].notna() & 
                (df['subject'].astype(str) != 'None') &
                (df['subject'].astype(str) != '') &
                (df['subject'].astype(str) != 'nan') &
                (df['subject'].astype(str).str.len() > 0)
            )
            df = df[valid].reset_index(drop=True)
        
        removed = initial - len(df)
        if removed > 0:
            logger.info(f"  Removed {removed:,} rows without subject")
        
        return df
    
    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Study ID from metadata
        if '_study_id' in df.columns:
            df['study_id'] = df['_study_id'].apply(standardize_study_id)
        elif 'project_name' in df.columns:
            df['study_id'] = df['project_name'].apply(standardize_study_id)
        else:
            df['study_id'] = 'Unknown'
        
        # Site ID
        if 'site' in df.columns:
            df['site_id'] = df['site'].apply(standardize_site_id)
        else:
            df['site_id'] = None
        
        # Subject ID
        if 'subject' in df.columns:
            df['subject_id'] = df['subject'].apply(standardize_subject_id)
        else:
            df['subject_id'] = None
        
        logger.info(f"  Study ID sample: {df['study_id'].head(3).tolist()}")
        logger.info(f"  Site ID sample: {df['site_id'].head(3).tolist()}")
        logger.info(f"  Subject ID sample: {df['subject_id'].head(3).tolist()}")
        
        valid_subjects = df['subject_id'].notna().sum()
        logger.info(f"  Valid subject IDs: {valid_subjects:,}/{len(df):,}")
        
        return df
    
    def _standardize_status(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'subject_status' in df.columns:
            logger.info(f"  Status column sample: {df['subject_status'].head(5).tolist()}")
            df['subject_status_original'] = df['subject_status']
            df['subject_status_clean'] = df['subject_status'].apply(standardize_subject_status)
            
            dist = df['subject_status_clean'].value_counts()
            logger.info("  Subject status distribution:")
            for status, count in dist.items():
                logger.info(f"    {status}: {count:,}")
        else:
            df['subject_status_clean'] = 'Unknown'
        
        return df
    
    def _convert_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        count = 0
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = safe_numeric(df[col], 0.0)
                count += 1
        logger.debug(f"  Converted {count} numeric columns")
        return df
    
    def _create_patient_key(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['patient_key'] = df.apply(
            lambda r: create_patient_key(
                str(r.get('study_id', 'Unknown')),
                r.get('site_id'),
                r.get('subject_id')
            ),
            axis=1
        )
        logger.info(f"  Patient key sample: {df['patient_key'].head(5).tolist()}")
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        df = df.drop_duplicates(keep='first')
        self.stats.duplicates_removed = initial - len(df)
        if self.stats.duplicates_removed > 0:
            logger.info(f"  Removed {self.stats.duplicates_removed:,} duplicates")
        return df.reset_index(drop=True)


# ============================================
# DETAIL TABLE AGGREGATORS
# ============================================

class DetailTableAggregator:
    def __init__(self):
        self.stats: Dict[str, CleaningStats] = {}
    
    def aggregate_visit_projection(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningStats]:
        stats = CleaningStats(table_name='visit_projection', input_rows=len(df), output_rows=0)
        logger.info(f"Aggregating visit_projection: {len(df):,} rows")
        
        df = df.copy()
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column_if_exists(df, ['subject', 'subject_id'])
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        site_col = get_column_if_exists(df, ['site', 'site_id', 'site_number'])
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        days_col = get_column_if_exists(df, ['days_outstanding', 'days_outstanding_1'])
        df['days_outstanding'] = safe_numeric(df[days_col], 0) if days_col else 0
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        visit_col = get_column_if_exists(df, ['visit', 'visit_name']) or 'patient_key'
        
        agg = df.groupby('patient_key').agg(
            study_id=('study_id', 'first'),
            site_id=('site_id', 'first'),
            subject_id=('subject_id', 'first'),
            missing_visit_count=(visit_col, 'count'),
            visits_overdue_max_days=('days_outstanding', 'max'),
            visits_overdue_avg_days=('days_outstanding', 'mean')
        ).reset_index()
        
        stats.output_rows = len(agg)
        logger.info(f"  ✓ Aggregated to {len(agg):,} patients")
        return agg, stats
    
    def aggregate_missing_lab_ranges(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningStats]:
        stats = CleaningStats(table_name='missing_lab_ranges', input_rows=len(df), output_rows=0)
        logger.info(f"Aggregating missing_lab_ranges: {len(df):,} rows")
        
        df = df.copy()
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column_if_exists(df, ['subject', 'subject_id', 'patient_id'])
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        site_col = get_column_if_exists(df, ['site', 'site_id', 'site_number', 'sitenumber'])
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        agg = df.groupby('patient_key').agg(
            study_id=('study_id', 'first'),
            site_id=('site_id', 'first'),
            subject_id=('subject_id', 'first'),
            lab_issue_count=('patient_key', 'count')
        ).reset_index()
        
        stats.output_rows = len(agg)
        logger.info(f"  ✓ Aggregated to {len(agg):,} patients")
        return agg, stats
    
    def aggregate_sae_dashboard(self, df: pd.DataFrame, sae_type: str) -> Tuple[pd.DataFrame, CleaningStats]:
        table_name = f'sae_dashboard_{sae_type.lower()}'
        stats = CleaningStats(table_name=table_name, input_rows=len(df), output_rows=0)
        logger.info(f"Aggregating {table_name}: {len(df):,} rows")
        
        df = df.copy()
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        patient_col = get_column_if_exists(df, ['patient_id', 'patientid', 'subject', 'subject_id'])
        df['subject_id'] = df[patient_col].apply(standardize_subject_id) if patient_col else None
        
        site_col = get_column_if_exists(df, ['site', 'site_id', 'siteid'])
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        
        if len(df) == 0:
            agg = pd.DataFrame(columns=['patient_key', 'study_id', 'site_id', 'subject_id',
                                        f'sae_{sae_type.lower()}_total', f'sae_{sae_type.lower()}_pending',
                                        f'sae_{sae_type.lower()}_completed'])
            stats.output_rows = 0
            return agg, stats
        
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        review_col = get_column_if_exists(df, ['review_status', 'reviewstatus', 'status'])
        if review_col:
            df['is_pending'] = df[review_col].str.contains('Pending', case=False, na=False)
            df['is_completed'] = df[review_col].str.contains('Completed|Complete', case=False, na=False)
        else:
            df['is_pending'] = False
            df['is_completed'] = False
        
        agg = df.groupby('patient_key').agg(
            study_id=('study_id', 'first'),
            site_id=('site_id', 'first'),
            subject_id=('subject_id', 'first'),
            **{f'sae_{sae_type.lower()}_total': ('patient_key', 'count'),
               f'sae_{sae_type.lower()}_pending': ('is_pending', 'sum'),
               f'sae_{sae_type.lower()}_completed': ('is_completed', 'sum')}
        ).reset_index()
        
        stats.output_rows = len(agg)
        logger.info(f"  ✓ Aggregated to {len(agg):,} patients")
        return agg, stats
    
    def aggregate_inactivated_forms(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningStats]:
        stats = CleaningStats(table_name='inactivated_forms', input_rows=len(df), output_rows=0)
        logger.info(f"Aggregating inactivated_forms: {len(df):,} rows")
        
        df = df.copy()
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column_if_exists(df, ['subject', 'subject_id', 'patient_id'])
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        site_col = get_column_if_exists(df, ['site', 'site_id', 'study_site_number', 'sitenumber'])
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        form_col = get_column_if_exists(df, ['form', 'formname', 'form_name']) or 'patient_key'
        
        agg = df.groupby('patient_key').agg(
            study_id=('study_id', 'first'),
            site_id=('site_id', 'first'),
            subject_id=('subject_id', 'first'),
            inactivated_form_count=('patient_key', 'count'),
            inactivated_unique_forms=(form_col, 'nunique')
        ).reset_index()
        
        stats.output_rows = len(agg)
        logger.info(f"  ✓ Aggregated to {len(agg):,} patients")
        return agg, stats
    
    def aggregate_missing_pages(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningStats]:
        stats = CleaningStats(table_name='missing_pages', input_rows=len(df), output_rows=0)
        logger.info(f"Aggregating missing_pages: {len(df):,} rows")
        
        df = df.copy()
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_candidates = ['subjectname', 'subject_name', 'subject', 'subject_id']
        df['subject_raw'] = coalesce_columns(df, subject_candidates)
        df['subject_id'] = df['subject_raw'].apply(standardize_subject_id)
        
        site_candidates = ['sitenumber', 'site_number', 'site', 'site_id']
        df['site_raw'] = coalesce_columns(df, site_candidates)
        df['site_id'] = df['site_raw'].apply(standardize_site_id)
        
        days_candidates = ['no_days_page_missing', 'of_days_missing', 'days_missing']
        df['days_missing_raw'] = coalesce_columns(df, days_candidates)
        df['days_missing'] = safe_numeric(df['days_missing_raw'], 0)
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        agg = df.groupby('patient_key').agg(
            study_id=('study_id', 'first'),
            site_id=('site_id', 'first'),
            subject_id=('subject_id', 'first'),
            missing_page_count=('patient_key', 'count'),
            pages_missing_max_days=('days_missing', 'max'),
            pages_missing_avg_days=('days_missing', 'mean')
        ).reset_index()
        
        stats.output_rows = len(agg)
        logger.info(f"  ✓ Aggregated to {len(agg):,} patients")
        return agg, stats
    
    def aggregate_compiled_edrr(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningStats]:
        stats = CleaningStats(table_name='compiled_edrr', input_rows=len(df), output_rows=0)
        logger.info(f"Aggregating compiled_edrr: {len(df):,} rows")
        
        df = df.copy()
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column_if_exists(df, ['subject', 'subject_id', 'patient_id'])
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], 'Unknown', r['subject_id']), axis=1)
        
        count_col = get_column_if_exists(df, ['total_open_issue_count_per_subject', 'issue_count'])
        df['edrr_issue_count'] = safe_numeric(df[count_col], 0) if count_col else 1
        
        agg = df.groupby('patient_key').agg(
            study_id=('study_id', 'first'),
            subject_id=('subject_id', 'first'),
            edrr_issue_count=('edrr_issue_count', 'sum')
        ).reset_index()
        agg['site_id'] = 'Unknown'
        
        stats.output_rows = len(agg)
        logger.info(f"  ✓ Aggregated to {len(agg):,} patients")
        return agg, stats
    
    def aggregate_coding(self, df: pd.DataFrame, coding_type: str) -> Tuple[pd.DataFrame, CleaningStats]:
        table_name = f'coding_{coding_type.lower()}'
        stats = CleaningStats(table_name=table_name, input_rows=len(df), output_rows=0)
        logger.info(f"Aggregating {table_name}: {len(df):,} rows")
        
        df = df.copy()
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column_if_exists(df, ['subject', 'subject_id', 'patient_id'])
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], 'Unknown', r['subject_id']), axis=1)
        
        status_col = get_column_if_exists(df, ['coding_status', 'status', 'codingstatus'])
        if status_col:
            df['is_coded'] = df[status_col].str.contains('Coded', case=False, na=False) & ~df[status_col].str.contains('UnCoded|Un-Coded|Not Coded', case=False, na=False)
            df['is_uncoded'] = df[status_col].str.contains('UnCoded|Un-Coded|Not Coded', case=False, na=False)
        else:
            df['is_coded'] = False
            df['is_uncoded'] = True
        
        agg = df.groupby('patient_key').agg(
            study_id=('study_id', 'first'),
            subject_id=('subject_id', 'first'),
            **{f'coding_{coding_type.lower()}_total': ('patient_key', 'count'),
               f'coding_{coding_type.lower()}_coded': ('is_coded', 'sum'),
               f'coding_{coding_type.lower()}_uncoded': ('is_uncoded', 'sum')}
        ).reset_index()
        agg['site_id'] = 'Unknown'
        
        stats.output_rows = len(agg)
        logger.info(f"  ✓ Aggregated to {len(agg):,} patients")
        return agg, stats


# ============================================
# MAIN ENGINE
# ============================================

class DataCleaningEngine:
    def __init__(self, input_dir: Path = None, output_dir: Path = None):
        self.input_dir = input_dir or DATA_PROCESSED
        self.output_dir = output_dir or CLEANED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = setup_logger()
        self.start_time = datetime.now()
        
        self.manifest = CleaningManifest(
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now().isoformat()
        )
        
        self.cpid_cleaner = CPIDEDCMetricsCleaner()
        self.aggregator = DetailTableAggregator()
    
    def run(self) -> CleaningManifest:
        logger.info("=" * 70)
        logger.info("a6on-i - DATA CLEANING (v3.0 SQL)")
        logger.info("=" * 70)
        
        try:
            from src.database.pg_data_service import PostgreSQLDataService
            from src.database.pg_writer import safe_to_postgres
            db_service = PostgreSQLDataService()
            
            # Step 1: Clean CPID
            logger.info("STEP 1: CLEANING CPID EDC METRICS (from SQL)")
            df_cpid = db_service.execute_query("SELECT * FROM cpid_edc_metrics")
            
            if df_cpid is not None and not df_cpid.empty:
                df_cpid_clean, cpid_stats = self.cpid_cleaner.clean(df_cpid)
                # Save back to SQL as cleaned version
                safe_to_postgres(df_cpid_clean, "cpid_edc_metrics_clean")
                # Also save to parquet for compatibility with some scripts
                df_cpid_clean.to_parquet(self.output_dir / "cpid_edc_metrics.parquet", index=False)
                
                self.manifest.table_stats['cpid_edc_metrics'] = asdict(cpid_stats)
                self.manifest.total_input_rows += cpid_stats.input_rows
                self.manifest.total_output_rows += cpid_stats.output_rows
                self.manifest.tables_processed += 1
            
            # Step 2: Aggregate detail tables
            logger.info("\nSTEP 2: AGGREGATING DETAIL TABLES (from SQL)")
            
            detail_tables = [
                ('visit_projection', self.aggregator.aggregate_visit_projection),
                ('missing_lab_ranges', self.aggregator.aggregate_missing_lab_ranges),
                ('sae_dashboard_dm', lambda df: self.aggregator.aggregate_sae_dashboard(df, 'DM')),
                ('sae_dashboard_safety', lambda df: self.aggregator.aggregate_sae_dashboard(df, 'Safety')),
                ('inactivated_forms', self.aggregator.aggregate_inactivated_forms),
                ('missing_pages', self.aggregator.aggregate_missing_pages),
                ('compiled_edrr', self.aggregator.aggregate_compiled_edrr),
                ('coding_meddra', lambda df: self.aggregator.aggregate_coding(df, 'meddra')),
                ('coding_whodrug', lambda df: self.aggregator.aggregate_coding(df, 'whodrug')),
            ]
            
            for table_name, agg_func in detail_tables:
                try:
                    df = db_service.execute_query(f"SELECT * FROM {table_name}")
                    if df is None or df.empty:
                        continue
                        
                    # Add _study_id if missing (it's in the raw SQL table but maybe named differently)
                    if '_study_id' not in df.columns:
                        df['_study_id'] = df['study_id'] if 'study_id' in df.columns else 'Unknown'
                        
                    df_agg, stats = agg_func(df)
                    
                    # Save back to SQL
                    safe_to_postgres(df_agg, f"{table_name}_agg")
                    # Also save to parquet
                    df_agg.to_parquet(self.output_dir / f"{table_name}_agg.parquet", index=False)
                    
                    self.manifest.table_stats[stats.table_name] = asdict(stats)
                    self.manifest.total_input_rows += stats.input_rows
                    self.manifest.total_output_rows += stats.output_rows
                    self.manifest.tables_processed += 1
                except Exception as e:
                    logger.error(f"  ✗ Error cleaning {table_name}: {e}")
                    
            self.manifest.status = "completed"
            self.manifest.end_time = datetime.now().isoformat()
            
        except Exception as e:
            self.manifest.status = "failed"
            self.manifest.errors.append({'error': str(e)})
            logger.error(traceback.format_exc())
        
        # Save manifest
        manifest_path = self.output_dir / "cleaning_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(convert_to_serializable(self.manifest.to_dict()), f, indent=2)
        
        self._print_summary()
        return self.manifest

    
    def _print_summary(self):
        logger.info("\n" + "=" * 70)
        logger.info("CLEANING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Status: {self.manifest.status}")
        logger.info(f"Tables: {self.manifest.tables_processed}")
        logger.info(f"Input: {self.manifest.total_input_rows:,}")
        logger.info(f"Output: {self.manifest.total_output_rows:,}")
        
        if self.manifest.errors:
            logger.error(f"\nErrors: {len(self.manifest.errors)}")
        else:
            logger.info("\n✅ NO ERRORS!")


def main():
    engine = DataCleaningEngine()
    manifest = engine.run()
    
    if manifest.status == "completed" and not manifest.errors:
        print("\n" + "=" * 70)
        print("✅ CLEANING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    return 1


if __name__ == "__main__":
    import sys

# PostgreSQL data access
try:
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

    sys.exit(main())
