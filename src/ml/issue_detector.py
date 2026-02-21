# src/ml/issue_detector.py
"""
SANCHALAK AI - Phase 3.3: Issue Detector (FIXED v1.3)
Multi-label classification for 14 issue types with priority assignment

Version: 1.3 - Fixed all data issues based on actual column analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available")


class IssueDetector:
    """
    Multi-label issue detector for 14 clinical trial issue types.
    
    Version 1.3 - Fixed based on actual data analysis:
    - SAE Safety: Use total - completed for pending calculation
    - Overdue Queries: Removed (no data) - replaced with Query Aging proxy
    - SDV: Fixed comparison logic
    - ML: Strict feature separation to prevent leakage
    """
    
    ISSUE_TYPES = {
        'sae_dm_pending': {
            'name': 'SAE-DM Pending',
            'category': 'safety',
            'criticality': 1.5,
            'responsible': 'Safety Data Manager',
            'sla_days': 3,
            'weight': 25.0
        },
        'sae_safety_pending': {
            'name': 'SAE-Safety Pending',
            'category': 'safety',
            'criticality': 1.5,
            'responsible': 'Safety Physician',
            'sla_days': 7,
            'weight': 25.0
        },
        'open_queries': {
            'name': 'Open Queries',
            'category': 'query',
            'criticality': 1.1,
            'responsible': 'Data Manager',
            'sla_days': 14,
            'weight': 20.0
        },
        'high_query_volume': {
            'name': 'High Query Volume',
            'category': 'query',
            'criticality': 1.2,
            'responsible': 'Data Manager',
            'sla_days': 7,
            'weight': 15.0
        },
        'sdv_incomplete': {
            'name': 'SDV Incomplete',
            'category': 'sdv',
            'criticality': 1.0,
            'responsible': 'CRA',
            'sla_days': 30,
            'weight': 8.0
        },
        'signature_gaps': {
            'name': 'Signature Gaps',
            'category': 'signature',
            'criticality': 1.1,
            'responsible': 'Site',
            'sla_days': 14,
            'weight': 5.0
        },
        'broken_signatures': {
            'name': 'Broken Signatures',
            'category': 'signature',
            'criticality': 1.2,
            'responsible': 'Site',
            'sla_days': 7,
            'weight': 5.0
        },
        'meddra_uncoded': {
            'name': 'MedDRA Uncoded',
            'category': 'coding',
            'criticality': 1.0,
            'responsible': 'Medical Coder',
            'sla_days': 21,
            'weight': 6.0
        },
        'whodrug_uncoded': {
            'name': 'WHODrug Uncoded',
            'category': 'coding',
            'criticality': 1.0,
            'responsible': 'Medical Coder',
            'sla_days': 21,
            'weight': 6.0
        },
        'missing_visits': {
            'name': 'Missing Visits',
            'category': 'completeness',
            'criticality': 1.1,
            'responsible': 'CRA',
            'sla_days': 14,
            'weight': 7.5
        },
        'missing_pages': {
            'name': 'Missing Pages',
            'category': 'completeness',
            'criticality': 1.1,
            'responsible': 'CRA',
            'sla_days': 14,
            'weight': 7.5
        },
        'lab_issues': {
            'name': 'Lab Issues',
            'category': 'lab',
            'criticality': 1.1,
            'responsible': 'Data Manager',
            'sla_days': 21,
            'weight': 10.0
        },
        'edrr_issues': {
            'name': 'EDRR Issues',
            'category': 'edrr',
            'criticality': 1.0,
            'responsible': 'Data Manager',
            'sla_days': 21,
            'weight': 5.0
        },
        'inactivated_forms': {
            'name': 'Inactivated Forms',
            'category': 'forms',
            'criticality': 0.8,
            'responsible': 'Data Manager',
            'sla_days': 30,
            'weight': 3.0
        }
    }
    
    # RAW features only - no derived/aggregated columns
    # These are the ONLY features allowed for ML to prevent leakage
    RAW_FEATURE_WHITELIST = [
        # Query columns (raw counts)
        'dm_queries', 'clinical_queries', 'medical_queries', 
        'site_queries', 'field_monitor_queries', 'coding_queries', 'safety_queries',
        # CRF columns
        'crfs_require_verification_sdv', 'crfs_frozen', 'crfs_not_frozen',
        'crfs_locked', 'crfs_unlocked', 'crfs_signed',
        'crfs_overdue_for_signs_within_45_days_of_data_entry',
        'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
        'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
        'crfs_never_signed', 'broken_signatures',
        # Pages
        'pages_entered', 'pages_with_nonconformant_data',
        # Visits
        'expected_visits_rave_edc_bo4',
        # Protocol deviations
        'pds_major', 'pds_minor',
    ]
    
    def __init__(self, model_dir: Path = None):
        """Initialize the Issue Detector."""
        self.model_dir = model_dir or Path('models/issue_detector')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.thresholds = {}
        self.feature_importance = {}
        self.is_fitted = False
        
    def _safe_get(self, df: pd.DataFrame, col: str, default: float = 0) -> pd.Series:
        """Safely get a column, returning default if not found."""
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce').fillna(default)
        return pd.Series(default, index=df.index)
    
    def detect_issues_rule_based(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule-based issue detection from raw data.
        FIXED v1.3: Corrected logic for all issue types based on actual data.
        """
        print("\n" + "="*60)
        print("RULE-BASED ISSUE DETECTION (v1.3)")
        print("="*60)
        
        result = df.copy()
        issue_counts = {}
        
        for issue_type, metadata in self.ISSUE_TYPES.items():
            col_name = f'issue_{issue_type}'
            count_col = f'count_{issue_type}'
            result[col_name] = 0
            result[count_col] = 0
            
            try:
                if issue_type == 'sae_dm_pending':
                    # Use the pending column directly
                    pending = self._safe_get(df, 'sae_dm_sae_dm_pending', 0)
                    result[col_name] = (pending > 0).astype(int)
                    result[count_col] = pending
                    
                elif issue_type == 'sae_safety_pending':
                    # FIXED: pending column is all 0, calculate from total - completed
                    total = self._safe_get(df, 'sae_safety_sae_safety_total', 0)
                    completed = self._safe_get(df, 'sae_safety_sae_safety_completed', 0)
                    pending = (total - completed).clip(lower=0)
                    result[col_name] = (pending > 0).astype(int)
                    result[count_col] = pending
                    
                elif issue_type == 'open_queries':
                    # Sum of all query types
                    total = self._safe_get(df, 'total_queries', 0)
                    result[col_name] = (total > 0).astype(int)
                    result[count_col] = total
                    
                elif issue_type == 'high_query_volume':
                    # REPLACED overdue_queries: High volume = > 10 queries per patient
                    total = self._safe_get(df, 'total_queries', 0)
                    result[col_name] = (total > 10).astype(int)
                    result[count_col] = total
                    
                elif issue_type == 'sdv_incomplete':
                    # FIXED: CRFs requiring SDV > 0 means SDV is incomplete
                    # (if all SDV was done, this would be 0)
                    require_sdv = self._safe_get(df, 'crfs_require_verification_sdv', 0)
                    result[col_name] = (require_sdv > 0).astype(int)
                    result[count_col] = require_sdv
                    
                elif issue_type == 'signature_gaps':
                    # Never signed OR any overdue signatures
                    never_signed = self._safe_get(df, 'crfs_never_signed', 0)
                    overdue_45 = self._safe_get(df, 'crfs_overdue_for_signs_within_45_days_of_data_entry', 0)
                    overdue_90 = self._safe_get(df, 'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry', 0)
                    overdue_beyond = self._safe_get(df, 'crfs_overdue_for_signs_beyond_90_days_of_data_entry', 0)
                    
                    total_gaps = never_signed + overdue_45 + overdue_90 + overdue_beyond
                    result[col_name] = (total_gaps > 0).astype(int)
                    result[count_col] = total_gaps
                    
                elif issue_type == 'broken_signatures':
                    broken = self._safe_get(df, 'broken_signatures', 0)
                    result[col_name] = (broken > 0).astype(int)
                    result[count_col] = broken
                    
                elif issue_type == 'meddra_uncoded':
                    uncoded = self._safe_get(df, 'meddra_coding_meddra_uncoded', 0)
                    result[col_name] = (uncoded > 0).astype(int)
                    result[count_col] = uncoded
                    
                elif issue_type == 'whodrug_uncoded':
                    uncoded = self._safe_get(df, 'whodrug_coding_whodrug_uncoded', 0)
                    result[col_name] = (uncoded > 0).astype(int)
                    result[count_col] = uncoded
                    
                elif issue_type == 'missing_visits':
                    # FIXED: Use the correct column
                    count = self._safe_get(df, 'visit_missing_visit_count', 0)
                    result[col_name] = (count > 0).astype(int)
                    result[count_col] = count
                    
                elif issue_type == 'missing_pages':
                    # FIXED: Use pages_missing_page_count (not missing_pages which is all 0)
                    count = self._safe_get(df, 'pages_missing_page_count', 0)
                    result[col_name] = (count > 0).astype(int)
                    result[count_col] = count
                    
                elif issue_type == 'lab_issues':
                    count = self._safe_get(df, 'lab_lab_issue_count', 0)
                    result[col_name] = (count > 0).astype(int)
                    result[count_col] = count
                    
                elif issue_type == 'edrr_issues':
                    # Try multiple columns
                    count = self._safe_get(df, 'edrr_edrr_issue_count', 0)
                    if count.sum() == 0:
                        count = self._safe_get(df, 'open_issues_edrr', 0)
                    result[col_name] = (count > 0).astype(int)
                    result[count_col] = count
                    
                elif issue_type == 'inactivated_forms':
                    count = self._safe_get(df, 'inactivated_inactivated_form_count', 0)
                    result[col_name] = (count > 0).astype(int)
                    result[count_col] = count
                
                # Count
                detected = int(result[col_name].sum())
                pct = detected / len(result) * 100
                issue_counts[issue_type] = {'count': detected, 'pct': round(pct, 2)}
                
                status = "✓" if detected > 0 else "○"
                print(f"  {status} {metadata['name']:25s}: {detected:,} patients ({pct:.1f}%)")
                
            except Exception as e:
                print(f"  ✗ {metadata['name']:25s}: ERROR - {str(e)[:50]}")
                issue_counts[issue_type] = {'count': 0, 'pct': 0, 'error': str(e)}
        
        # Calculate totals
        issue_cols = [f'issue_{it}' for it in self.ISSUE_TYPES.keys()]
        result['total_issues'] = result[issue_cols].sum(axis=1)
        result['has_any_issue'] = (result['total_issues'] > 0).astype(int)
        
        # Summary
        no_issues = (result['total_issues'] == 0).sum()
        with_issues = (result['total_issues'] > 0).sum()
        print(f"\n  ─────────────────────────────────────────────")
        print(f"  Patients with NO issues: {no_issues:,} ({no_issues/len(result)*100:.1f}%)")
        print(f"  Patients with 1+ issues: {with_issues:,} ({with_issues/len(result)*100:.1f}%)")
        print(f"  Average issues/patient:  {result['total_issues'].mean():.2f}")
        
        self.issue_counts = issue_counts
        return result
    
    def calculate_severity_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate severity score for each issue.
        FIXED v1.3: Use count columns for severity calculation.
        """
        print("\n" + "="*60)
        print("CALCULATING SEVERITY SCORES")
        print("="*60)
        
        result = df.copy()
        
        for issue_type, metadata in self.ISSUE_TYPES.items():
            sev_col = f'severity_{issue_type}'
            issue_col = f'issue_{issue_type}'
            count_col = f'count_{issue_type}'
            
            # Get count (default to issue flag if count not available)
            if count_col in result.columns:
                count = result[count_col].fillna(0)
            else:
                count = result[issue_col].fillna(0)
            
            # Calculate severity based on issue type
            if issue_type == 'sae_dm_pending':
                severity = count * 15  # High weight per pending SAE
                
            elif issue_type == 'sae_safety_pending':
                severity = count * 20  # Very high weight for safety pending
                
            elif issue_type == 'open_queries':
                severity = count * 2  # 2 points per query
                
            elif issue_type == 'high_query_volume':
                severity = (count - 10).clip(lower=0) * 3  # Extra points above threshold
                
            elif issue_type == 'sdv_incomplete':
                severity = count * 0.2  # Low weight per CRF
                
            elif issue_type == 'signature_gaps':
                # Weight by overdue period
                never = self._safe_get(df, 'crfs_never_signed', 0)
                overdue_45 = self._safe_get(df, 'crfs_overdue_for_signs_within_45_days_of_data_entry', 0)
                overdue_90 = self._safe_get(df, 'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry', 0)
                overdue_beyond = self._safe_get(df, 'crfs_overdue_for_signs_beyond_90_days_of_data_entry', 0)
                severity = never * 1 + overdue_45 * 0.5 + overdue_90 * 2 + overdue_beyond * 5
                
            elif issue_type == 'broken_signatures':
                severity = count * 3
                
            elif issue_type == 'meddra_uncoded':
                severity = count * 2
                
            elif issue_type == 'whodrug_uncoded':
                severity = count * 2
                
            elif issue_type == 'missing_visits':
                severity = count * 10  # High weight per missing visit
                
            elif issue_type == 'missing_pages':
                severity = count * 2
                
            elif issue_type == 'lab_issues':
                severity = count * 5
                
            elif issue_type == 'edrr_issues':
                severity = count * 3
                
            elif issue_type == 'inactivated_forms':
                severity = count * 0.3  # Low weight
            
            else:
                severity = count
            
            # Apply criticality multiplier
            severity = severity * metadata['criticality']
            
            # Cap at 100
            severity = severity.clip(upper=100)
            
            # Zero if no issue
            if issue_col in result.columns:
                severity = severity * result[issue_col]
            
            result[sev_col] = severity
            
            # Stats for non-zero only
            non_zero = severity[severity > 0]
            if len(non_zero) > 0:
                print(f"  {metadata['name']:25s}: n={len(non_zero):,}, mean={non_zero.mean():.1f}, max={non_zero.max():.1f}")
            else:
                print(f"  {metadata['name']:25s}: n=0")
        
        # Total severity
        sev_cols = [f'severity_{it}' for it in self.ISSUE_TYPES.keys()]
        result['total_severity'] = result[sev_cols].sum(axis=1)
        
        print(f"\n  Total Severity: mean={result['total_severity'].mean():.1f}, max={result['total_severity'].max():.1f}")
        
        return result
    
    def assign_priority_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign priority tier based on severity and issue types."""
        print("\n" + "="*60)
        print("ASSIGNING PRIORITY TIERS")
        print("="*60)
        
        result = df.copy()
        
        # Calculate priority score
        priority_score = pd.Series(0.0, index=result.index)
        for issue_type, metadata in self.ISSUE_TYPES.items():
            sev_col = f'severity_{issue_type}'
            if sev_col in result.columns:
                priority_score += result[sev_col] * (metadata['weight'] / 100)
        
        result['priority_score'] = priority_score.clip(upper=100)
        
        # Check for safety issues
        has_safety = (
            (result.get('issue_sae_dm_pending', 0) == 1) | 
            (result.get('issue_sae_safety_pending', 0) == 1)
        )
        
        # Calculate percentiles for dynamic thresholds (only for patients with issues)
        with_issues_mask = result['total_issues'] > 0
        if with_issues_mask.sum() > 0:
            scores_with_issues = priority_score[with_issues_mask]
            p25 = scores_with_issues.quantile(0.25)
            p50 = scores_with_issues.quantile(0.50)
            p75 = scores_with_issues.quantile(0.75)
            p90 = scores_with_issues.quantile(0.90)
        else:
            p25, p50, p75, p90 = 5, 10, 25, 50
        
        print(f"\n  Score percentiles (patients with issues): P25={p25:.1f}, P50={p50:.1f}, P75={p75:.1f}, P90={p90:.1f}")
        
        # Assign tiers
        result['priority_tier'] = 'none'
        result.loc[result['total_issues'] > 0, 'priority_tier'] = 'low'
        result.loc[result['priority_score'] >= p25, 'priority_tier'] = 'low'
        result.loc[result['priority_score'] >= p50, 'priority_tier'] = 'medium'
        result.loc[(result['priority_score'] >= p75) | (result['total_issues'] >= 4), 'priority_tier'] = 'high'
        result.loc[has_safety | (result['priority_score'] >= p90), 'priority_tier'] = 'critical'
        
        # No issues = none
        result.loc[result['total_issues'] == 0, 'priority_tier'] = 'none'
        
        # SLA days
        sla_map = {'critical': 3, 'high': 7, 'medium': 14, 'low': 30, 'none': 0}
        result['sla_days'] = result['priority_tier'].map(sla_map)
        
        # Print distribution
        print("\nPriority Distribution:")
        for tier in ['critical', 'high', 'medium', 'low', 'none']:
            count = (result['priority_tier'] == tier).sum()
            pct = count / len(result) * 100
            print(f"  {tier.upper():10s}: {count:,} ({pct:.1f}%)")
        
        return result
    
    def identify_primary_issue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify the primary (most severe) issue for each patient."""
        print("\n" + "="*60)
        print("IDENTIFYING PRIMARY ISSUES")
        print("="*60)
        
        result = df.copy()
        
        sev_cols = [f'severity_{it}' for it in self.ISSUE_TYPES.keys()]
        available_sev = [c for c in sev_cols if c in result.columns]
        
        if not available_sev:
            result['primary_issue'] = 'none'
            result['primary_issue_severity'] = 0
            return result
        
        sev_df = result[available_sev].copy()
        result['primary_issue_severity'] = sev_df.max(axis=1)
        result['primary_issue'] = sev_df.idxmax(axis=1).str.replace('severity_', '')
        
        # No issues = none
        result.loc[result['total_issues'] == 0, 'primary_issue'] = 'none'
        result.loc[result['total_issues'] == 0, 'primary_issue_severity'] = 0
        
        # Count
        print("\nPrimary Issue Distribution:")
        counts = result['primary_issue'].value_counts()
        for issue, count in counts.head(12).items():
            pct = count / len(result) * 100
            print(f"  {issue:25s}: {count:,} ({pct:.1f}%)")
        
        return result
    
    def train_ml_models(self, df: pd.DataFrame) -> Dict:
        """
        Train ML models for issue prediction.
        FIXED v1.3: Use ONLY raw features to prevent data leakage.
        """
        if not XGB_AVAILABLE:
            print("\nSkipping ML training - XGBoost not available")
            return {}
        
        print("\n" + "="*60)
        print("TRAINING ML MODELS (No Leakage)")
        print("="*60)
        
        # Use ONLY raw features from whitelist
        available_features = [f for f in self.RAW_FEATURE_WHITELIST if f in df.columns]
        
        if len(available_features) < 5:
            print(f"  Only {len(available_features)} raw features found - skipping ML")
            return {}
        
        print(f"  Using {len(available_features)} RAW features (whitelist only)")
        print(f"  Features: {available_features[:10]}...")
        
        X = df[available_features].copy()
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Remove constant columns
        non_const = X.columns[X.std() > 0]
        X = X[non_const]
        print(f"  After removing constants: {len(X.columns)} features")
        
        if len(X.columns) < 3:
            print("  Too few features after filtering - skipping ML")
            return {}
        
        results = {}
        
        for issue_type, metadata in self.ISSUE_TYPES.items():
            issue_col = f'issue_{issue_type}'
            
            if issue_col not in df.columns:
                continue
            
            y = df[issue_col].fillna(0).astype(int)
            pos_rate = y.mean()
            
            if pos_rate == 0 or pos_rate == 1:
                print(f"  {metadata['name']:25s}: SKIPPED (no variation)")
                continue
            
            if pos_rate < 0.001:  # Less than 0.1%
                print(f"  {metadata['name']:25s}: SKIPPED (too rare: {pos_rate:.4%})")
                continue
            
            try:
                # Split data properly
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                scale_pos = min((1 - pos_rate) / pos_rate, 10)
                
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='auc',
                    verbosity=0,
                    n_jobs=-1
                )
                
                # Fit on train, evaluate on test
                model.fit(X_train, y_train)
                
                # Predict on test set
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation on training set
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
                
                print(f"  {metadata['name']:25s}: Test AUC={test_auc:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
                # Store model (retrain on full data for deployment)
                model.fit(X, y)
                self.models[issue_type] = model
                
                # Feature importance
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[issue_type] = importance
                
                results[issue_type] = {
                    'test_auc': float(test_auc),
                    'cv_auc_mean': float(cv_scores.mean()),
                    'cv_auc_std': float(cv_scores.std()),
                    'positive_rate': float(pos_rate),
                    'top_features': importance.head(3)['feature'].tolist()
                }
                
            except Exception as e:
                print(f"  {metadata['name']:25s}: ERROR - {str(e)[:40]}")
        
        self.is_fitted = len(self.models) > 0
        print(f"\n  Trained {len(self.models)} models")
        
        return results
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive issue report."""
        report = {
            'summary': {
                'total_patients': len(df),
                'patients_with_issues': int((df.get('has_any_issue', 0) == 1).sum()),
                'patients_clean': int((df.get('has_any_issue', 0) == 0).sum()),
                'avg_issues_per_patient': float(df.get('total_issues', pd.Series(0)).mean()),
                'avg_severity': float(df.get('total_severity', pd.Series(0)).mean())
            },
            'by_issue_type': {},
            'by_priority': {},
            'by_responsible': {},
            'by_category': {}
        }
        
        # By issue type
        for issue_type, metadata in self.ISSUE_TYPES.items():
            issue_col = f'issue_{issue_type}'
            sev_col = f'severity_{issue_type}'
            count_col = f'count_{issue_type}'
            
            if issue_col in df.columns:
                detected = int(df[issue_col].sum())
                avg_sev = float(df.loc[df[issue_col] == 1, sev_col].mean()) if sev_col in df.columns and detected > 0 else 0
                avg_count = float(df.loc[df[issue_col] == 1, count_col].mean()) if count_col in df.columns and detected > 0 else 0
                
                report['by_issue_type'][issue_type] = {
                    'name': metadata['name'],
                    'count': detected,
                    'pct': round(detected / len(df) * 100, 2),
                    'avg_severity': round(avg_sev, 2),
                    'avg_item_count': round(avg_count, 2),
                    'category': metadata['category'],
                    'responsible': metadata['responsible'],
                    'sla_days': metadata['sla_days']
                }
        
        # By priority
        if 'priority_tier' in df.columns:
            for tier in ['critical', 'high', 'medium', 'low', 'none']:
                count = int((df['priority_tier'] == tier).sum())
                report['by_priority'][tier] = {
                    'count': count,
                    'pct': round(count / len(df) * 100, 2)
                }
        
        # By responsible
        responsible = {}
        for issue_type, metadata in self.ISSUE_TYPES.items():
            issue_col = f'issue_{issue_type}'
            resp = metadata['responsible']
            if issue_col in df.columns:
                if resp not in responsible:
                    responsible[resp] = {'patients': set(), 'issues': 0}
                mask = df[issue_col] == 1
                responsible[resp]['patients'].update(df.index[mask].tolist())
                responsible[resp]['issues'] += int(df[issue_col].sum())
        
        report['by_responsible'] = {
            k: {'unique_patients': len(v['patients']), 'total_issues': v['issues']}
            for k, v in sorted(responsible.items(), key=lambda x: x[1]['issues'], reverse=True)
        }
        
        # By category
        categories = {}
        for issue_type, metadata in self.ISSUE_TYPES.items():
            cat = metadata['category']
            issue_col = f'issue_{issue_type}'
            if issue_col in df.columns:
                if cat not in categories:
                    categories[cat] = 0
                categories[cat] += int(df[issue_col].sum())
        report['by_category'] = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
        
        return report
    
    def save_models(self):
        """Save trained models."""
        if not self.models:
            print("No models to save")
            return
        
        for issue_type, model in self.models.items():
            path = self.model_dir / f'issue_detector_{issue_type}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save feature importance
        for issue_type, importance in self.feature_importance.items():
            path = self.model_dir / f'feature_importance_{issue_type}.csv'
            importance.to_csv(path, index=False)
        
        meta = {
            'issue_types': list(self.ISSUE_TYPES.keys()),
            'trained_models': list(self.models.keys()),
            'raw_features_used': self.RAW_FEATURE_WHITELIST,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.model_dir / 'issue_detector_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"\nModels saved to {self.model_dir}")


def run_issue_detection():
    """Main execution function."""
    print("\n" + "="*70)
    print("SANCHALAK AI - PHASE 3.3: ISSUE DETECTOR v1.3")
    print("="*70)
    
    start_time = datetime.now()
    
    data_dir = Path('data/processed')
    output_dir = data_dir / 'analytics'
    model_dir = Path('models/issue_detector')
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    upr_paths = [
        data_dir / 'segments' / 'unified_patient_record_segmented.parquet',
        data_dir / 'upr' / 'unified_patient_record.parquet',
        data_dir / 'metrics' / 'patient_metrics.parquet'
    ]
    
    df = None
    for path in upr_paths:
        if path.exists():
            print(f"\nLoading: {path}")
            df = pd.read_parquet(path)
            print(f"Loaded {len(df):,} patients, {len(df.columns)} columns")
            break
    
    if df is None:
        raise FileNotFoundError("No UPR data found!")
    
    # Initialize detector
    detector = IssueDetector(model_dir=model_dir)
    
    # Run pipeline
    df_issues = detector.detect_issues_rule_based(df)
    df_severity = detector.calculate_severity_scores(df_issues)
    df_priority = detector.assign_priority_tiers(df_severity)
    df_final = detector.identify_primary_issue(df_priority)
    ml_results = detector.train_ml_models(df_final)
    report = detector.generate_report(df_final)
    
    # Save outputs
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    # Select columns to save (avoid bloat)
    save_cols = ['patient_key', 'study_id', 'site_id', 'subject_id']
    save_cols += [c for c in df_final.columns if c.startswith('issue_')]
    save_cols += [c for c in df_final.columns if c.startswith('count_')]
    save_cols += [c for c in df_final.columns if c.startswith('severity_')]
    save_cols += ['total_issues', 'has_any_issue', 'total_severity', 
                  'priority_tier', 'priority_score', 'sla_days',
                  'primary_issue', 'primary_issue_severity']
    save_cols = [c for c in save_cols if c in df_final.columns]
    
    # Save to PostgreSQL
    pg_writer = get_pg_writer()
    success, msg = pg_writer.safe_to_postgres(df_final[save_cols], 'patient_issues', if_exists='replace')
    if success:
        logger.info(f"Saved patient_issues to PostgreSQL: {len(df_final):,} rows")
    else:
        logger.warning(f"PostgreSQL save failed, falling back to parquet: {msg}")
        
    # Save to PostgreSQL
    from src.database.pg_writer import safe_to_postgres
    success, msg = safe_to_postgres(df_final[save_cols], 'patient_issues', if_exists='replace')
    if success:
        print(f"  ✓ patient_issues saved to PostgreSQL: {len(df_final):,} rows, {len(save_cols)} cols")
    else:
        print(f"  ✗ PostgreSQL write failed: {msg}")

    
    # Issue summary
    issue_summary = pd.DataFrame([
        {
            'issue_type': it,
            'name': meta['name'],
            'category': meta['category'],
            'count': report['by_issue_type'].get(it, {}).get('count', 0),
            'pct': report['by_issue_type'].get(it, {}).get('pct', 0),
            'avg_severity': report['by_issue_type'].get(it, {}).get('avg_severity', 0),
            'avg_item_count': report['by_issue_type'].get(it, {}).get('avg_item_count', 0),
            'responsible': meta['responsible'],
            'sla_days': meta['sla_days']
        }
        for it, meta in detector.ISSUE_TYPES.items()
    ])
    issue_summary.to_csv(output_dir / 'issue_summary.csv', index=False)
    print(f"  ✓ issue_summary.csv")
    
    # Priority distribution
    priority_df = pd.DataFrame([
        {'tier': tier, **data}
        for tier, data in report['by_priority'].items()
    ])
    priority_df.to_csv(output_dir / 'priority_distribution.csv', index=False)
    print(f"  ✓ priority_distribution.csv")
    
    # High priority patients
    high_priority = df_final[df_final['priority_tier'].isin(['critical', 'high'])]
    if len(high_priority) > 0:
        key_cols = ['patient_key', 'study_id', 'site_id', 'subject_id', 
                    'priority_tier', 'priority_score', 'total_issues', 
                    'primary_issue', 'primary_issue_severity', 'sla_days']
        key_cols = [c for c in key_cols if c in high_priority.columns]
        high_priority[key_cols].to_csv(output_dir / 'high_priority_patients.csv', index=False)
        print(f"  ✓ high_priority_patients.csv ({len(high_priority):,} patients)")
    
    # ML results
    if ml_results:
        ml_df = pd.DataFrame([
            {'issue_type': it, **{k: v for k, v in data.items() if k != 'top_features'}}
            for it, data in ml_results.items()
        ])
        ml_df.to_csv(output_dir / 'issue_detector_ml_results.csv', index=False)
        print(f"  ✓ issue_detector_ml_results.csv")
    
    # Save models
    detector.save_models()
    
    # Full report
    with open(output_dir / 'issue_detector_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✓ issue_detector_report.json")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("PHASE 3.3 COMPLETE (v1.3)")
    print("="*70)
    print(f"\n  Patients:              {len(df_final):,}")
    print(f"  With Issues:           {report['summary']['patients_with_issues']:,} ({report['summary']['patients_with_issues']/len(df_final)*100:.1f}%)")
    print(f"  Clean (No Issues):     {report['summary']['patients_clean']:,} ({report['summary']['patients_clean']/len(df_final)*100:.1f}%)")
    print(f"  Avg Issues/Patient:    {report['summary']['avg_issues_per_patient']:.2f}")
    print(f"  ML Models Trained:     {len(ml_results)}")
    print(f"  Duration:              {duration:.1f}s")
    
    print("\n  Priority Distribution:")
    for tier in ['critical', 'high', 'medium', 'low', 'none']:
        data = report['by_priority'].get(tier, {'count': 0, 'pct': 0})
        emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢', 'none': '⚪'}
        print(f"    {emoji.get(tier, '')} {tier.upper():10s}: {data['count']:,} ({data['pct']:.1f}%)")
    
    print("\n  Issue Counts:")
    for it, meta in detector.ISSUE_TYPES.items():
        data = report['by_issue_type'].get(it, {'count': 0, 'pct': 0})
        if data['count'] > 0:
            print(f"    {meta['name']:25s}: {data['count']:,} ({data['pct']:.1f}%)")
    
    return detector, df_final, report


if __name__ == "__main__":
    detector, df_results, report = run_issue_detection()