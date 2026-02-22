"""
a6on-i - DB Lock Ready Engine v2.1 (FIXED)
Phase 2.3: Database Lock Readiness Assessment

FIXES:
- Handle nan% for studies with 0 eligible patients
- Show underlying blockers instead of just tier2_clean
- Fix days-to-ready categories
- Improve blocker analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class DBLockStatus(Enum):
    """DB Lock Status Categories"""
    READY = ("Ready", "🟢", "Can be locked immediately")
    PENDING = ("Pending", "🟡", "Minor items remaining - lock within days")
    NOT_READY = ("Not Ready", "🟠", "Significant work needed")
    BLOCKED = ("Blocked", "🔴", "Critical blockers present")
    NOT_ELIGIBLE = ("Not Eligible", "⚪", "Excluded from DB Lock")


class BlockerSeverity(Enum):
    """Blocker Severity Levels"""
    CRITICAL = ("Critical", 1, "Must resolve before any lock activity")
    HIGH = ("High", 2, "Blocks lock, needs immediate attention")
    MEDIUM = ("Medium", 3, "Should resolve before lock")
    LOW = ("Low", 4, "Can potentially lock with documented exception")


@dataclass
class DBLockCriterion:
    """DB Lock criterion definition"""
    name: str
    description: str
    column: str
    condition: str
    severity: BlockerSeverity
    resolution_days: float
    responsible: str
    can_exception: bool


@dataclass
class DBLockConfig:
    """DB Lock Configuration"""
    
    eligible_statuses: List[str] = field(default_factory=lambda: [
        'Ongoing', 'Completed', 'Discontinued'
    ])
    
    excluded_statuses: List[str] = field(default_factory=lambda: [
        'Screen Failure', 'Screening', 'Unknown'
    ])
    
    # All criteria for DB Lock (unified list for better blocker analysis)
    all_criteria: Dict[str, DBLockCriterion] = field(default_factory=lambda: {
        # Critical blockers
        'no_sae_dm_pending': DBLockCriterion(
            name='No SAE DM Pending',
            description='No pending SAE data management',
            column='sae_dm_sae_dm_pending',
            condition='equals_zero',
            severity=BlockerSeverity.CRITICAL,
            resolution_days=2.0,
            responsible='Safety Data Manager',
            can_exception=False
        ),
        'no_sae_safety_pending': DBLockCriterion(
            name='No SAE Safety Pending',
            description='No pending safety narratives',
            column='sae_safety_sae_safety_pending',
            condition='equals_zero',
            severity=BlockerSeverity.CRITICAL,
            resolution_days=3.0,
            responsible='Safety Physician',
            can_exception=False
        ),
        # High blockers
        'no_open_queries': DBLockCriterion(
            name='No Open Queries',
            description='All queries resolved',
            column='total_queries',
            condition='equals_zero',
            severity=BlockerSeverity.HIGH,
            resolution_days=0.5,
            responsible='Data Manager',
            can_exception=False
        ),
        'signatures_complete': DBLockCriterion(
            name='Signatures Complete',
            description='No CRFs awaiting signature',
            column='crfs_never_signed',
            condition='equals_zero',
            severity=BlockerSeverity.HIGH,
            resolution_days=1.0,
            responsible='Site',
            can_exception=False
        ),
        'sdv_complete': DBLockCriterion(
            name='SDV Complete',
            description='Source data verification complete',
            column='crfs_require_verification_sdv',
            condition='equals_zero',
            severity=BlockerSeverity.HIGH,
            resolution_days=0.25,
            responsible='CRA',
            can_exception=False
        ),
        'no_missing_visits': DBLockCriterion(
            name='No Missing Visits',
            description='All expected visits entered',
            column='visit_missing_visit_count',
            condition='equals_zero',
            severity=BlockerSeverity.HIGH,
            resolution_days=2.0,
            responsible='CRA',
            can_exception=False
        ),
        'no_missing_pages': DBLockCriterion(
            name='No Missing Pages',
            description='All CRF pages entered',
            column='pages_missing_page_count',
            condition='equals_zero',
            severity=BlockerSeverity.HIGH,
            resolution_days=1.5,
            responsible='CRA',
            can_exception=False
        ),
        # Medium blockers
        'coding_complete': DBLockCriterion(
            name='Coding Complete',
            description='All terms coded',
            column='total_uncoded_terms',
            condition='equals_zero',
            severity=BlockerSeverity.MEDIUM,
            resolution_days=0.5,
            responsible='Medical Coder',
            can_exception=True
        ),
        'no_broken_signatures': DBLockCriterion(
            name='No Broken Signatures',
            description='No invalidated signatures',
            column='broken_signatures',
            condition='equals_zero',
            severity=BlockerSeverity.MEDIUM,
            resolution_days=1.0,
            responsible='Site',
            can_exception=True
        ),
        'no_overdue_signatures': DBLockCriterion(
            name='No Overdue Signatures',
            description='No signatures overdue >90 days',
            column='crfs_overdue_for_signs_beyond_90_days_of_data_entry',
            condition='equals_zero',
            severity=BlockerSeverity.MEDIUM,
            resolution_days=1.0,
            responsible='Site',
            can_exception=True
        ),
        'no_edrr_issues': DBLockCriterion(
            name='No EDRR Issues',
            description='External data reconciled',
            column='edrr_edrr_issue_count',
            condition='equals_zero',
            severity=BlockerSeverity.MEDIUM,
            resolution_days=2.0,
            responsible='Data Manager',
            can_exception=True
        ),
        # Low blockers
        'no_inactivated': DBLockCriterion(
            name='No Inactivations',
            description='All inactivations documented',
            column='inactivated_inactivated_form_count',
            condition='equals_zero',
            severity=BlockerSeverity.LOW,
            resolution_days=0.5,
            responsible='Data Manager',
            can_exception=True
        ),
        'no_lab_issues': DBLockCriterion(
            name='No Lab Issues',
            description='Lab names and ranges resolved',
            column='lab_lab_issue_count',
            condition='equals_zero',
            severity=BlockerSeverity.LOW,
            resolution_days=1.0,
            responsible='Data Manager',
            can_exception=True
        )
    })
    
    work_hours_per_day: float = 8.0
    estimation_buffer: float = 1.2


# =============================================================================
# DB LOCK READY CALCULATOR
# =============================================================================

class DBLockReadyCalculator:
    """Enhanced DB Lock Ready Assessment Engine v2.1"""
    
    def __init__(self, config: DBLockConfig = None):
        self.config = config or DBLockConfig()
        self.eligibility_stats = {}
        self.readiness_stats = {}
        self.blocker_stats = {}
        
    def _get_col(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Safely get column"""
        if col in df.columns:
            return df[col].fillna(0)
        return pd.Series(0, index=df.index)
    
    def _check_condition(self, series: pd.Series, condition: str) -> pd.Series:
        """Evaluate condition"""
        if condition == 'equals_zero':
            return series == 0
        elif condition == 'is_true':
            return series == True
        elif condition == 'is_false':
            return series == False
        return pd.Series(True, index=series.index)
    
    def check_eligibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check DB Lock eligibility"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING DB LOCK ELIGIBILITY")
        logger.info("=" * 70)
        
        result = df.copy()
        status_col = 'subject_status_clean' if 'subject_status_clean' in df.columns else 'subject_status'
        
        eligible_mask = result[status_col].isin(self.config.eligible_statuses)
        excluded_mask = result[status_col].isin(self.config.excluded_statuses)
        
        result['dblock_eligible'] = eligible_mask
        result['dblock_exclusion_reason'] = None
        result.loc[excluded_mask, 'dblock_exclusion_reason'] = result.loc[excluded_mask, status_col]
        
        total = len(df)
        eligible = eligible_mask.sum()
        excluded = excluded_mask.sum()
        
        self.eligibility_stats = {
            'total': total,
            'eligible': int(eligible),
            'eligible_pct': round(eligible / total * 100, 2) if total > 0 else 0,
            'excluded': int(excluded)
        }
        
        logger.info(f"\n  ELIGIBILITY SUMMARY:")
        logger.info(f"    Total Patients: {total:,}")
        logger.info(f"    ✅ Eligible: {eligible:,} ({eligible/total*100:.1f}%)")
        logger.info(f"    ❌ Excluded: {excluded:,} ({excluded/total*100:.1f}%)")
        
        logger.info(f"\n  EXCLUSION BREAKDOWN:")
        for status in self.config.excluded_statuses:
            count = (result[status_col] == status).sum()
            if count > 0:
                logger.info(f"    {status}: {count:,} ({count/total*100:.1f}%)")
        
        return result
    
    def check_all_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check all DB Lock criteria and identify blockers"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING ALL DB LOCK CRITERIA")
        logger.info("=" * 70)
        
        result = df.copy()
        eligible_mask = result['dblock_eligible'] == True
        
        self.blocker_stats = {}
        
        for criterion_id, criterion in self.config.all_criteria.items():
            col_data = self._get_col(df, criterion.column)
            passes = self._check_condition(col_data, criterion.condition)
            
            # Store pass/fail for each criterion
            result[f'dblock_{criterion_id}'] = passes
            result[f'dblock_{criterion_id}_count'] = col_data
            
            # Count failures among eligible
            failures = (~passes & eligible_mask).sum()
            issue_count = int(col_data[~passes & eligible_mask].sum())
            
            self.blocker_stats[criterion_id] = {
                'name': criterion.name,
                'severity': criterion.severity.value[0],
                'priority': criterion.severity.value[1],
                'failures': int(failures),
                'issue_count': issue_count,
                'responsible': criterion.responsible,
                'can_exception': criterion.can_exception
            }
            
            status = "✅" if failures == 0 else "❌"
            if failures > 0:
                logger.info(f"\n  {status} {criterion.name} [{criterion.severity.value[0]}]")
                logger.info(f"      Failures: {failures:,} patients ({failures/eligible_mask.sum()*100:.1f}%)")
                logger.info(f"      Issues: {issue_count:,}")
                logger.info(f"      Responsible: {criterion.responsible}")
        
        return result
    
    def calculate_readiness_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate readiness tiers"""
        logger.info("\n" + "=" * 70)
        logger.info("CALCULATING READINESS TIERS")
        logger.info("=" * 70)
        
        result = df.copy()
        eligible_mask = result['dblock_eligible'] == True
        
        # Get all criterion columns
        criterion_cols = [f'dblock_{c}' for c in self.config.all_criteria.keys()]
        available_cols = [c for c in criterion_cols if c in result.columns]
        
        # Tier 1 Ready: ALL criteria pass
        all_pass = eligible_mask.copy()
        for col in available_cols:
            all_pass = all_pass & result[col]
        
        result['dblock_tier1_ready'] = all_pass
        tier1_count = all_pass.sum()
        tier1_pct = tier1_count / eligible_mask.sum() * 100 if eligible_mask.sum() > 0 else 0
        
        logger.info(f"\n  TIER 1 (READY): {tier1_count:,} ({tier1_pct:.1f}% of eligible)")
        
        # Get critical and high severity criteria
        critical_high_cols = []
        medium_low_cols = []
        
        for criterion_id, criterion in self.config.all_criteria.items():
            col = f'dblock_{criterion_id}'
            if col in result.columns:
                if criterion.severity in [BlockerSeverity.CRITICAL, BlockerSeverity.HIGH]:
                    critical_high_cols.append(col)
                else:
                    medium_low_cols.append(col)
        
        # Tier 2 Pending: All critical/high pass, only medium/low fail
        critical_high_pass = eligible_mask.copy()
        for col in critical_high_cols:
            critical_high_pass = critical_high_pass & result[col]
        
        tier2_pending = critical_high_pass & ~all_pass
        result['dblock_tier2_pending'] = tier2_pending
        tier2_count = tier2_pending.sum()
        tier2_pct = tier2_count / eligible_mask.sum() * 100 if eligible_mask.sum() > 0 else 0
        
        logger.info(f"  TIER 2 (PENDING): {tier2_count:,} ({tier2_pct:.1f}% of eligible)")
        
        # Not Ready: Eligible but not Tier 1 or Tier 2
        not_ready = eligible_mask & ~all_pass & ~tier2_pending
        result['dblock_not_ready'] = not_ready
        not_ready_count = not_ready.sum()
        not_ready_pct = not_ready_count / eligible_mask.sum() * 100 if eligible_mask.sum() > 0 else 0
        
        logger.info(f"  NOT READY: {not_ready_count:,} ({not_ready_pct:.1f}% of eligible)")
        
        self.readiness_stats = {
            'tier1_ready': int(tier1_count),
            'tier1_pct': round(tier1_pct, 2),
            'tier2_pending': int(tier2_count),
            'tier2_pct': round(tier2_pct, 2),
            'not_ready': int(not_ready_count),
            'not_ready_pct': round(not_ready_pct, 2)
        }
        
        return result
    
    def assign_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign DB Lock status"""
        logger.info("\n" + "=" * 70)
        logger.info("ASSIGNING DB LOCK STATUS")
        logger.info("=" * 70)
        
        result = df.copy()
        
        def get_status(row):
            if not row.get('dblock_eligible', False):
                return 'Not Eligible'
            elif row.get('dblock_tier1_ready', False):
                return 'Ready'
            elif row.get('dblock_tier2_pending', False):
                return 'Pending'
            else:
                return 'Blocked'
        
        def get_icon(status):
            icons = {
                'Ready': '🟢',
                'Pending': '🟡',
                'Blocked': '🔴',
                'Not Eligible': '⚪'
            }
            return icons.get(status, '⚪')
        
        result['dblock_status'] = result.apply(get_status, axis=1)
        result['dblock_status_icon'] = result['dblock_status'].apply(get_icon)
        
        # Log distribution
        logger.info(f"\n  STATUS DISTRIBUTION:")
        for status in ['Ready', 'Pending', 'Blocked', 'Not Eligible']:
            count = (result['dblock_status'] == status).sum()
            pct = count / len(result) * 100
            icon = get_icon(status)
            logger.info(f"    {icon} {status}: {count:,} ({pct:.1f}%)")
        
        return result
    
    def identify_blockers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify blockers for each patient"""
        logger.info("\n" + "=" * 70)
        logger.info("IDENTIFYING BLOCKERS")
        logger.info("=" * 70)
        
        result = df.copy()
        
        # Sort criteria by priority
        sorted_criteria = sorted(
            self.config.all_criteria.items(),
            key=lambda x: x[1].severity.value[1]
        )
        
        def get_blockers(row):
            """Get list of failing criteria"""
            blockers = []
            for criterion_id, criterion in sorted_criteria:
                col = f'dblock_{criterion_id}'
                if col in row.index and not row[col]:
                    blockers.append({
                        'id': criterion_id,
                        'name': criterion.name,
                        'severity': criterion.severity.value[0],
                        'priority': criterion.severity.value[1],
                        'responsible': criterion.responsible
                    })
            return blockers
        
        def get_primary_blocker(row):
            blockers = get_blockers(row)
            return blockers[0]['id'] if blockers else None
        
        def get_primary_blocker_name(row):
            blockers = get_blockers(row)
            return blockers[0]['name'] if blockers else None
        
        def get_blocker_count(row):
            return len(get_blockers(row))
        
        def get_critical_count(row):
            blockers = get_blockers(row)
            return sum(1 for b in blockers if b['severity'] == 'Critical')
        
        def format_blockers(row):
            blockers = get_blockers(row)
            if not blockers:
                return 'None'
            return '; '.join([f"[{b['severity'][0]}] {b['name']}" for b in blockers[:5]])
        
        result['dblock_primary_blocker'] = result.apply(get_primary_blocker, axis=1)
        result['dblock_primary_blocker_name'] = result.apply(get_primary_blocker_name, axis=1)
        result['dblock_blocker_count'] = result.apply(get_blocker_count, axis=1)
        result['dblock_critical_count'] = result.apply(get_critical_count, axis=1)
        result['dblock_blockers_list'] = result.apply(format_blockers, axis=1)
        
        # Blocker analysis
        eligible_mask = result['dblock_eligible'] == True
        not_ready_mask = ~result['dblock_tier1_ready'] & eligible_mask
        
        logger.info(f"\n  TOP BLOCKERS (among not-ready eligible patients):")
        blocker_counts = result[not_ready_mask]['dblock_primary_blocker_name'].value_counts().head(10)
        for blocker, count in blocker_counts.items():
            if blocker:
                pct = count / not_ready_mask.sum() * 100
                logger.info(f"    {blocker}: {count:,} ({pct:.1f}%)")
        
        return result
    
    def estimate_days_to_ready(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate days to ready"""
        logger.info("\n" + "=" * 70)
        logger.info("ESTIMATING DAYS TO READY")
        logger.info("=" * 70)
        
        result = df.copy()
        
        def calculate_days(row):
            if row.get('dblock_tier1_ready', False):
                return 0.0
            if not row.get('dblock_eligible', False):
                return np.nan
            
            total_days = 0.0
            
            for criterion_id, criterion in self.config.all_criteria.items():
                col = f'dblock_{criterion_id}'
                if col in row.index and not row[col]:
                    count_col = f'dblock_{criterion_id}_count'
                    issue_count = row.get(count_col, 1) if count_col in row.index else 1
                    issue_count = max(1, min(issue_count, 100))  # Cap at 100
                    
                    # Days based on issue count (with diminishing returns)
                    if issue_count <= 5:
                        days = criterion.resolution_days * issue_count
                    else:
                        days = criterion.resolution_days * 5 + criterion.resolution_days * 0.5 * (issue_count - 5)
                    
                    total_days += min(days, 30)  # Cap at 30 days per criterion
            
            total_days *= self.config.estimation_buffer
            return round(min(total_days, 90), 1)  # Cap at 90 days total
        
        result['dblock_days_to_ready'] = result.apply(calculate_days, axis=1)
        
        # Categories
        def days_category(days):
            if pd.isna(days):
                return 'Not Eligible'
            elif days == 0:
                return 'Ready Now'
            elif days <= 3:
                return '1-3 Days'
            elif days <= 7:
                return '4-7 Days'
            elif days <= 14:
                return '1-2 Weeks'
            elif days <= 30:
                return '2-4 Weeks'
            else:
                return '>4 Weeks'
        
        result['dblock_days_category'] = result['dblock_days_to_ready'].apply(days_category)
        
        # Statistics
        eligible_mask = result['dblock_eligible'] == True
        logger.info(f"\n  DAYS TO READY DISTRIBUTION (Eligible patients):")
        for category in ['Ready Now', '1-3 Days', '4-7 Days', '1-2 Weeks', '2-4 Weeks', '>4 Weeks']:
            count = (result[eligible_mask]['dblock_days_category'] == category).sum()
            pct = count / eligible_mask.sum() * 100 if eligible_mask.sum() > 0 else 0
            if count > 0:
                logger.info(f"    {category}: {count:,} ({pct:.1f}%)")
        
        not_ready_days = result[eligible_mask & ~result['dblock_tier1_ready']]['dblock_days_to_ready']
        if len(not_ready_days) > 0:
            logger.info(f"\n  Average days to ready (not ready): {not_ready_days.mean():.1f}")
            logger.info(f"  Median days to ready (not ready): {not_ready_days.median():.1f}")
        
        return result
    
    def calculate_study_projections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate study-level projections"""
        logger.info("\n" + "=" * 70)
        logger.info("STUDY-LEVEL PROJECTIONS")
        logger.info("=" * 70)
        
        agg_dict = {
            'patient_key': 'count',
            'dblock_eligible': 'sum',
            'dblock_tier1_ready': 'sum',
            'dblock_tier2_pending': 'sum',
            'dblock_days_to_ready': ['mean', 'median', 'max']
        }
        
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        agg_dict['patient_key'] = 'count'
        
        study_df = df.groupby('study_id').agg(agg_dict).reset_index()
        
        # Flatten columns
        new_cols = []
        for col in study_df.columns:
            if isinstance(col, tuple):
                new_cols.append('_'.join([str(c) for c in col if c]))
            else:
                new_cols.append(col)
        study_df.columns = new_cols
        
        study_df = study_df.rename(columns={
            'patient_key_count': 'total_patients',
            'patient_key': 'total_patients',
            'dblock_eligible_sum': 'eligible',
            'dblock_tier1_ready_sum': 'ready',
            'dblock_tier2_pending_sum': 'pending',
            'dblock_days_to_ready_mean': 'avg_days',
            'dblock_days_to_ready_median': 'median_days',
            'dblock_days_to_ready_max': 'max_days'
        })
        
        # Calculate rates (handle division by zero)
        study_df['ready_rate'] = np.where(
            study_df['eligible'] > 0,
            (study_df['ready'] / study_df['eligible'] * 100).round(1),
            np.nan
        )
        
        # Study tier
        def get_tier(row):
            if pd.isna(row['ready_rate']) or row['eligible'] == 0:
                return 'No Eligible Patients'
            elif row['ready_rate'] >= 95:
                return 'Lock Ready'
            elif row['ready_rate'] >= 80:
                return 'Near Ready'
            elif row['ready_rate'] >= 50:
                return 'In Progress'
            else:
                return 'Not Ready'
        
        study_df['lock_tier'] = study_df.apply(get_tier, axis=1)
        
        # Estimated lock date
        study_df['est_lock_date'] = pd.Timestamp.now() + pd.to_timedelta(
            study_df['max_days'].fillna(0), unit='D'
        )
        
        logger.info(f"\n  STUDY READINESS:")
        for _, row in study_df.iterrows():
            eligible = row.get('eligible', 0)
            ready_rate = row.get('ready_rate', 0)
            tier = row.get('lock_tier', 'Unknown')
            
            if eligible > 0:
                logger.info(f"    {row['study_id']}: {ready_rate:.1f}% ready, {int(eligible)} eligible ({tier})")
            else:
                logger.info(f"    {row['study_id']}: No eligible patients")
        
        return study_df
    
    def calculate_site_projections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate site-level projections"""
        logger.info("\n" + "=" * 70)
        logger.info("SITE-LEVEL PROJECTIONS")
        logger.info("=" * 70)
        
        agg_dict = {
            'patient_key': 'count',
            'dblock_eligible': 'sum',
            'dblock_tier1_ready': 'sum',
            'dblock_days_to_ready': 'mean',
            'dblock_blocker_count': 'sum'
        }
        
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        agg_dict['patient_key'] = 'count'
        
        site_df = df.groupby(['study_id', 'site_id']).agg(agg_dict).reset_index()
        
        site_df = site_df.rename(columns={
            'patient_key': 'total_patients',
            'dblock_eligible': 'eligible',
            'dblock_tier1_ready': 'ready',
            'dblock_days_to_ready': 'avg_days',
            'dblock_blocker_count': 'total_blockers'
        })
        
        # Ready rate
        site_df['ready_rate'] = np.where(
            site_df['eligible'] > 0,
            (site_df['ready'] / site_df['eligible'] * 100).round(1),
            100.0  # 100% if no eligible (nothing to do)
        )
        
        # Priority score (lower = higher priority to fix)
        site_df['priority_score'] = 100 - site_df['ready_rate']
        
        logger.info(f"\n  Sites analyzed: {len(site_df):,}")
        
        sites_with_eligible = site_df[site_df['eligible'] > 0]
        if len(sites_with_eligible) > 0:
            logger.info(f"  Sites with eligible patients: {len(sites_with_eligible):,}")
            logger.info(f"  Average ready rate: {sites_with_eligible['ready_rate'].mean():.1f}%")
        
        return site_df
    
    def calculate_full_db_lock_ready(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run complete DB Lock Ready calculation"""
        logger.info("\n" + "=" * 70)
        logger.info("a6on-i - DB LOCK READY ENGINE v2.1")
        logger.info("=" * 70)
        logger.info(f"Patients: {len(df):,}")
        
        start_time = datetime.now()
        
        result = self.check_eligibility(df)
        result = self.check_all_criteria(result)
        result = self.calculate_readiness_tiers(result)
        result = self.assign_status(result)
        result = self.identify_blockers(result)
        result = self.estimate_days_to_ready(result)
        
        study_df = self.calculate_study_projections(result)
        site_df = self.calculate_site_projections(result)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 70)
        logger.info("DB LOCK READY COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.2f} seconds")
        
        return result, study_df, site_df


# =============================================================================
# MAIN ENGINE
# =============================================================================

class DBLockReadyEngine:
    """Main DB Lock Ready Engine"""
    
    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        self.study_df = None
        self.site_df = None
        self.calculator = DBLockReadyCalculator()
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(self.df):,} patients with {len(self.df.columns)} columns")
        return self.df
    
    def run(self) -> pd.DataFrame:
        self.df, self.study_df, self.site_df = self.calculator.calculate_full_db_lock_ready(self.df)
        return self.df
    
    def save_outputs(self) -> Dict[str, Path]:
        logger.info("\n" + "=" * 60)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 60)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # Save to PostgreSQL
        pg_writer = get_pg_writer()
        success, msg = pg_writer.safe_to_postgres(self.df, 'patient_dblock_status', if_exists='replace')
        if success:
            logger.info(f"✅ Saved patient_dblock_status to PostgreSQL: {len(self.df):,} rows")
        else:
            logger.warning(f"PostgreSQL failed, using parquet: {msg}")
            patient_path = self.output_dir / 'patient_dblock_status.parquet'
            self.df.to_parquet(patient_path, index=False)
        
        # Study
        study_path = self.output_dir / 'study_dblock_projection.csv'
        self.study_df.to_csv(study_path, index=False)
        saved_files['study'] = study_path
        logger.info(f"✅ Study projections: {study_path}")
        
        # Site
        site_path = self.output_dir / 'site_dblock_projection.csv'
        self.site_df.to_csv(site_path, index=False)
        saved_files['site'] = site_path
        logger.info(f"✅ Site projections: {site_path}")
        
        # Ready patients
        ready_cols = ['patient_key', 'study_id', 'site_id', 'subject_id', 'dblock_status']
        available = [c for c in ready_cols if c in self.df.columns]
        ready_df = self.df[self.df['dblock_tier1_ready']][available]
        ready_path = self.output_dir / 'dblock_ready_patients.csv'
        ready_df.to_csv(ready_path, index=False)
        logger.info(f"✅ Ready patients: {len(ready_df):,}")
        
        # Blocked patients
        blocked_cols = ['patient_key', 'study_id', 'site_id', 'subject_id',
                        'dblock_status', 'dblock_primary_blocker_name', 
                        'dblock_blocker_count', 'dblock_days_to_ready',
                        'dblock_days_category', 'dblock_blockers_list']
        available = [c for c in blocked_cols if c in self.df.columns]
        
        blocked_df = self.df[~self.df['dblock_tier1_ready'] & self.df['dblock_eligible']][available]
        blocked_df = blocked_df.sort_values('dblock_days_to_ready')
        blocked_path = self.output_dir / 'dblock_blocked_patients.csv'
        blocked_df.to_csv(blocked_path, index=False)
        logger.info(f"✅ Blocked patients: {len(blocked_df):,}")
        
        # Summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'version': '2.1.0',
            'patient_count': len(self.df),
            'eligibility': self.calculator.eligibility_stats,
            'readiness': self.calculator.readiness_stats,
            'blockers': self.calculator.blocker_stats,
            'status_distribution': self.df['dblock_status'].value_counts().to_dict(),
            'days_distribution': self.df['dblock_days_category'].value_counts().to_dict()
        }
        
        summary_path = self.output_dir / 'dblock_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"✅ Summary: {summary_path}")
        
        return saved_files
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 PHASE 2.3 COMPLETE - DB LOCK READY ENGINE v2.1")
        print("=" * 70)
        
        print(f"\n🔢 PATIENTS: {len(self.df):,}")
        
        eligible = self.df['dblock_eligible'].sum()
        print(f"\n📋 ELIGIBILITY: {eligible:,} / {len(self.df):,} ({eligible/len(self.df)*100:.1f}%)")
        
        print(f"\n🔒 DB LOCK STATUS:")
        for status in ['Ready', 'Pending', 'Blocked', 'Not Eligible']:
            count = (self.df['dblock_status'] == status).sum()
            pct = count / len(self.df) * 100
            icons = {'Ready': '🟢', 'Pending': '🟡', 'Blocked': '🔴', 'Not Eligible': '⚪'}
            print(f"   {icons[status]} {status}: {count:,} ({pct:.1f}%)")
        
        print(f"\n📅 DAYS TO READY (Eligible):")
        for cat in ['Ready Now', '1-3 Days', '4-7 Days', '1-2 Weeks', '2-4 Weeks', '>4 Weeks']:
            count = (self.df[self.df['dblock_eligible']]['dblock_days_category'] == cat).sum()
            if count > 0:
                pct = count / eligible * 100
                print(f"   {cat}: {count:,} ({pct:.1f}%)")
        
        print(f"\n🚫 TOP BLOCKERS:")
        not_ready = self.df[~self.df['dblock_tier1_ready'] & self.df['dblock_eligible']]
        blockers = not_ready['dblock_primary_blocker_name'].value_counts().head(5)
        for blocker, count in blockers.items():
            if blocker:
                print(f"   {blocker}: {count:,}")
        
        print(f"\n📚 STUDY READINESS:")
        for _, row in self.study_df.iterrows():
            eligible = row.get('eligible', 0)
            if eligible > 0:
                ready_rate = row.get('ready_rate', 0)
                tier = row.get('lock_tier', 'Unknown')
                print(f"   {row['study_id']}: {ready_rate:.0f}% ready ({tier})")
        
        print(f"\n📁 Output: {self.output_dir}")


def main():
    project_root = Path(__file__).parent.parent.parent
    
    input_path = project_root / 'data' / 'processed' / 'analytics' / 'patient_clean_status.parquet'
    
    if not input_path.exists():
        input_path = project_root / 'data' / 'processed' / 'analytics' / 'patient_dqi_enhanced.parquet'
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return
    
    output_dir = project_root / 'data' / 'processed' / 'analytics'
    
    engine = DBLockReadyEngine(input_path, output_dir)
    engine.load_data()
    engine.run()
    engine.save_outputs()
    engine.print_summary()
    
    return engine


if __name__ == '__main__':
    main()
