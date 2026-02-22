"""
a6on-i - Enhanced Clean Patient Calculator v2.0
Phase 2.2: Two-Tier Clean Patient with Effort Scoring

Features:
- Tier 1: Clinical Clean (7 hard blocks)
- Tier 2: Operational Clean (7 soft blocks)  
- Quick Wins identification with priority ranking
- Effort-to-clean calculation (hours/days estimate)
- Blocking reason analysis
- Path-to-clean recommendations
- Site-level clean patient summary
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
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

class BlockType(Enum):
    """Block type classification"""
    HARD = ("Hard Block", "Must be resolved for Clinical Clean")
    SOFT = ("Soft Block", "Should be resolved for Operational Clean")
    
    @property
    def name_str(self): return self.value[0]
    @property
    def description(self): return self.value[1]


@dataclass
class CleanCriterion:
    """Definition of a single clean patient criterion"""
    name: str
    description: str
    column: str
    condition: str  # 'equals_zero', 'equals_100', 'is_true', 'is_false'
    block_type: BlockType
    effort_hours_per_issue: float  # Estimated hours to resolve per issue
    priority: int  # 1=highest priority to resolve
    responsible_role: str  # Who typically resolves this
    resolution_action: str  # What action to take


@dataclass
class CleanPatientConfig:
    """Complete Clean Patient Configuration"""
    
    # Tier 1: Clinical Clean (7 Hard Blocks)
    tier1_criteria: Dict[str, CleanCriterion] = field(default_factory=lambda: {
        'no_missing_visits': CleanCriterion(
            name='No Missing Visits',
            description='All expected visits have been entered',
            column='visit_missing_visit_count',
            condition='equals_zero',
            block_type=BlockType.HARD,
            effort_hours_per_issue=2.0,
            priority=2,
            responsible_role='CRA',
            resolution_action='Follow up with site on missing visit data'
        ),
        'no_missing_pages': CleanCriterion(
            name='No Missing Pages',
            description='All CRF pages have been entered',
            column='pages_missing_page_count',
            condition='equals_zero',
            block_type=BlockType.HARD,
            effort_hours_per_issue=1.5,
            priority=2,
            responsible_role='CRA',
            resolution_action='Request site to complete missing CRF pages'
        ),
        'no_open_queries': CleanCriterion(
            name='No Open Queries',
            description='All data queries have been resolved',
            column='total_queries',
            condition='equals_zero',
            block_type=BlockType.HARD,
            effort_hours_per_issue=0.5,
            priority=1,
            responsible_role='Data Manager',
            resolution_action='Review and close open queries with site'
        ),
        'sdv_complete': CleanCriterion(
            name='SDV Complete',
            description='Source Data Verification is 100% complete',
            column='crfs_require_verification_sdv',
            condition='equals_zero',
            block_type=BlockType.HARD,
            effort_hours_per_issue=0.25,
            priority=3,
            responsible_role='CRA',
            resolution_action='Complete SDV during next monitoring visit'
        ),
        'signatures_complete': CleanCriterion(
            name='Signatures Complete',
            description='All required signatures have been obtained',
            column='crfs_never_signed',
            condition='equals_zero',
            block_type=BlockType.HARD,
            effort_hours_per_issue=0.1,
            priority=4,
            responsible_role='Site',
            resolution_action='Request PI/Sub-I to sign pending CRFs'
        ),
        'meddra_coded': CleanCriterion(
            name='MedDRA Coding Complete',
            description='All adverse events coded in MedDRA',
            column='meddra_coding_meddra_uncoded',
            condition='equals_zero',
            block_type=BlockType.HARD,
            effort_hours_per_issue=0.15,
            priority=5,
            responsible_role='Medical Coder',
            resolution_action='Complete MedDRA coding for uncoded terms'
        ),
        'whodrug_coded': CleanCriterion(
            name='WHODrug Coding Complete',
            description='All medications coded in WHODrug',
            column='whodrug_coding_whodrug_uncoded',
            condition='equals_zero',
            block_type=BlockType.HARD,
            effort_hours_per_issue=0.15,
            priority=5,
            responsible_role='Medical Coder',
            resolution_action='Complete WHODrug coding for uncoded terms'
        )
    })
    
    # Tier 2: Operational Clean (7 Soft Blocks)
    tier2_criteria: Dict[str, CleanCriterion] = field(default_factory=lambda: {
        'no_lab_issues': CleanCriterion(
            name='No Lab Issues',
            description='No missing lab names or ranges',
            column='lab_lab_issue_count',
            condition='equals_zero',
            block_type=BlockType.SOFT,
            effort_hours_per_issue=0.5,
            priority=6,
            responsible_role='Data Manager',
            resolution_action='Resolve lab name/range discrepancies'
        ),
        'no_sae_dm_pending': CleanCriterion(
            name='No SAE DM Pending',
            description='No SAE data management issues pending',
            column='sae_dm_sae_dm_pending',
            condition='equals_zero',
            block_type=BlockType.SOFT,
            effort_hours_per_issue=2.0,
            priority=1,
            responsible_role='Safety Data Manager',
            resolution_action='Complete SAE data reconciliation'
        ),
        'no_sae_safety_pending': CleanCriterion(
            name='No SAE Safety Pending',
            description='No SAE safety review pending',
            column='sae_safety_sae_safety_pending',
            condition='equals_zero',
            block_type=BlockType.SOFT,
            effort_hours_per_issue=3.0,
            priority=1,
            responsible_role='Safety Physician',
            resolution_action='Complete safety narrative and causality'
        ),
        'no_edrr_issues': CleanCriterion(
            name='No EDRR Issues',
            description='No external data reconciliation issues',
            column='edrr_edrr_issue_count',
            condition='equals_zero',
            block_type=BlockType.SOFT,
            effort_hours_per_issue=1.0,
            priority=7,
            responsible_role='Data Manager',
            resolution_action='Reconcile with external data sources'
        ),
        'no_overdue_signatures': CleanCriterion(
            name='No Overdue Signatures',
            description='No signatures overdue >90 days',
            column='crfs_overdue_for_signs_beyond_90_days_of_data_entry',
            condition='equals_zero',
            block_type=BlockType.SOFT,
            effort_hours_per_issue=0.25,
            priority=4,
            responsible_role='Site',
            resolution_action='Escalate overdue signatures to PI'
        ),
        'no_inactivated_forms': CleanCriterion(
            name='No Inactivated Forms',
            description='No unexplained inactivated forms',
            column='inactivated_inactivated_form_count',
            condition='equals_zero',
            block_type=BlockType.SOFT,
            effort_hours_per_issue=0.5,
            priority=8,
            responsible_role='Data Manager',
            resolution_action='Review and document inactivation reasons'
        ),
        'no_broken_signatures': CleanCriterion(
            name='No Broken Signatures',
            description='No broken/invalidated signatures',
            column='broken_signatures',
            condition='equals_zero',
            block_type=BlockType.SOFT,
            effort_hours_per_issue=0.5,
            priority=3,
            responsible_role='Site',
            resolution_action='Re-sign CRFs with broken signatures'
        )
    })
    
    # Quick Win thresholds
    quick_win_max_issues: int = 3  # Max issues to be considered quick win
    quick_win_max_hours: float = 4.0  # Max hours to be considered quick win


# =============================================================================
# ENHANCED CLEAN PATIENT CALCULATOR
# =============================================================================

class EnhancedCleanPatientCalculator:
    """
    Enhanced Two-Tier Clean Patient Calculator
    
    Features:
    - 14 criteria across 2 tiers
    - Effort estimation (hours/days)
    - Priority-based recommendations
    - Quick win identification
    - Path-to-clean roadmap
    """
    
    def __init__(self, config: CleanPatientConfig = None):
        self.config = config or CleanPatientConfig()
        self.tier1_results = {}
        self.tier2_results = {}
        self.summary_stats = {}
        
    def _get_col(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Safely get column or return zeros"""
        if col in df.columns:
            return df[col].fillna(0)
        return pd.Series(0, index=df.index)
    
    def _check_condition(self, series: pd.Series, condition: str) -> pd.Series:
        """Evaluate a condition"""
        if condition == 'equals_zero':
            return series == 0
        elif condition == 'equals_100':
            return series >= 100
        elif condition == 'greater_than_zero':
            return series > 0
        elif condition == 'is_true':
            return series == True
        elif condition == 'is_false':
            return series == False
        return pd.Series(True, index=series.index)
    
    def calculate_tier1_clinical_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Tier 1: Clinical Clean (7 Hard Blocks)
        """
        logger.info("\n" + "=" * 70)
        logger.info("TIER 1: CLINICAL CLEAN (7 HARD BLOCKS)")
        logger.info("=" * 70)
        
        result = df.copy()
        all_pass = pd.Series(True, index=df.index)
        tier1_issues_count = pd.Series(0, index=df.index)
        tier1_effort_hours = pd.Series(0.0, index=df.index)
        blocking_reasons = []
        
        for criterion_id, criterion in self.config.tier1_criteria.items():
            col_data = self._get_col(df, criterion.column)
            passes = self._check_condition(col_data, criterion.condition)
            
            # Store pass/fail
            result[f't1_{criterion_id}'] = passes
            result[f't1_{criterion_id}_count'] = col_data
            
            # Calculate effort for failures
            issue_count = col_data.copy()
            issue_count[passes] = 0  # No effort needed if passing
            effort = issue_count * criterion.effort_hours_per_issue
            result[f't1_{criterion_id}_effort_hrs'] = effort
            
            # Accumulate
            all_pass = all_pass & passes
            tier1_issues_count += (~passes).astype(int)
            tier1_effort_hours += effort
            
            # Statistics
            failures = (~passes).sum()
            total_issues = int(col_data[~passes].sum())
            total_effort = effort.sum()
            
            self.tier1_results[criterion_id] = {
                'criterion': criterion,
                'failures': failures,
                'total_issues': total_issues,
                'total_effort_hours': total_effort
            }
            
            status = "✅" if failures == 0 else "❌"
            logger.info(f"\n  {status} {criterion.name}")
            logger.info(f"      Column: {criterion.column}")
            logger.info(f"      Failures: {failures:,} patients")
            if failures > 0:
                logger.info(f"      Total issues: {total_issues:,}")
                logger.info(f"      Est. effort: {total_effort:,.1f} hours")
                logger.info(f"      Responsible: {criterion.responsible_role}")
                blocking_reasons.append((criterion_id, failures, criterion.priority))
        
        # Final Tier 1 status
        result['tier1_clean'] = all_pass
        result['tier1_failing_count'] = tier1_issues_count
        result['tier1_effort_hours'] = tier1_effort_hours
        result['tier1_effort_days'] = tier1_effort_hours / 8  # 8-hour workday
        
        # Primary blocking reason (highest priority issue)
        def get_primary_block(row):
            blocks = []
            for criterion_id, criterion in self.config.tier1_criteria.items():
                if not row.get(f't1_{criterion_id}', True):
                    blocks.append((criterion_id, criterion.priority))
            if not blocks:
                return None
            # Return highest priority (lowest number)
            return min(blocks, key=lambda x: x[1])[0]
        
        result['tier1_primary_block'] = result.apply(get_primary_block, axis=1)
        
        # Summary
        clean_count = all_pass.sum()
        clean_pct = clean_count / len(df) * 100
        
        self.summary_stats['tier1'] = {
            'clean_count': int(clean_count),
            'clean_pct': round(clean_pct, 2),
            'total_effort_hours': float(tier1_effort_hours.sum()),
            'avg_effort_per_patient': float(tier1_effort_hours[~all_pass].mean()) if (~all_pass).sum() > 0 else 0
        }
        
        logger.info(f"\n  {'─' * 50}")
        logger.info(f"  TIER 1 CLINICAL CLEAN: {clean_count:,} / {len(df):,} ({clean_pct:.1f}%)")
        logger.info(f"  Total Effort to Clean All: {tier1_effort_hours.sum():,.0f} hours ({tier1_effort_hours.sum()/8:,.0f} days)")
        
        return result
    
    def calculate_tier2_operational_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Tier 2: Operational Clean (7 Soft Blocks)
        Requires Tier 1 to be clean first
        """
        logger.info("\n" + "=" * 70)
        logger.info("TIER 2: OPERATIONAL CLEAN (7 SOFT BLOCKS)")
        logger.info("=" * 70)
        
        result = df.copy()
        
        # Must be Tier 1 clean first
        tier1_clean = result.get('tier1_clean', pd.Series(True, index=df.index))
        
        all_pass = tier1_clean.copy()
        tier2_issues_count = pd.Series(0, index=df.index)
        tier2_effort_hours = pd.Series(0.0, index=df.index)
        
        for criterion_id, criterion in self.config.tier2_criteria.items():
            col_data = self._get_col(df, criterion.column)
            passes = self._check_condition(col_data, criterion.condition)
            
            result[f't2_{criterion_id}'] = passes
            result[f't2_{criterion_id}_count'] = col_data
            
            # Calculate effort
            issue_count = col_data.copy()
            issue_count[passes] = 0
            effort = issue_count * criterion.effort_hours_per_issue
            result[f't2_{criterion_id}_effort_hrs'] = effort
            
            # Only count failures among Tier 1 clean
            all_pass = all_pass & passes
            tier2_issues_count += (~passes).astype(int)
            tier2_effort_hours += effort
            
            # Statistics (among Tier 1 clean only)
            t1_clean_mask = tier1_clean == True
            failures_in_t1 = (~passes & t1_clean_mask).sum()
            total_issues = int(col_data[~passes & t1_clean_mask].sum())
            
            self.tier2_results[criterion_id] = {
                'criterion': criterion,
                'failures': int((~passes).sum()),
                'failures_in_tier1': failures_in_t1,
                'total_issues': total_issues
            }
            
            status = "✅" if failures_in_t1 == 0 else "⚠️"
            logger.info(f"\n  {status} {criterion.name}")
            logger.info(f"      Column: {criterion.column}")
            logger.info(f"      Failures (in T1 clean): {failures_in_t1:,}")
            if failures_in_t1 > 0:
                logger.info(f"      Responsible: {criterion.responsible_role}")
        
        # Final Tier 2 status
        result['tier2_clean'] = all_pass
        result['tier2_failing_count'] = tier2_issues_count
        result['tier2_effort_hours'] = tier2_effort_hours
        result['tier2_effort_days'] = tier2_effort_hours / 8
        
        # Primary blocking reason for Tier 2
        def get_t2_primary_block(row):
            if not row.get('tier1_clean', False):
                return 'Not Tier 1 Clean'
            blocks = []
            for criterion_id, criterion in self.config.tier2_criteria.items():
                if not row.get(f't2_{criterion_id}', True):
                    blocks.append((criterion_id, criterion.priority))
            if not blocks:
                return None
            return min(blocks, key=lambda x: x[1])[0]
        
        result['tier2_primary_block'] = result.apply(get_t2_primary_block, axis=1)
        
        # Combined effort (Tier 1 + Tier 2)
        result['total_effort_hours'] = result['tier1_effort_hours'] + result['tier2_effort_hours']
        result['total_effort_days'] = result['total_effort_hours'] / 8
        
        # Summary
        clean_count = all_pass.sum()
        clean_pct = clean_count / len(df) * 100
        
        self.summary_stats['tier2'] = {
            'clean_count': int(clean_count),
            'clean_pct': round(clean_pct, 2),
            'total_effort_hours': float(tier2_effort_hours.sum())
        }
        
        logger.info(f"\n  {'─' * 50}")
        logger.info(f"  TIER 2 OPERATIONAL CLEAN: {clean_count:,} / {len(df):,} ({clean_pct:.1f}%)")
        
        return result
    
    def identify_quick_wins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify Quick Wins - patients close to clean with low effort
        """
        logger.info("\n" + "=" * 70)
        logger.info("IDENTIFYING QUICK WINS")
        logger.info("=" * 70)
        
        result = df.copy()
        
        # Quick Win Categories
        result['quick_win_category'] = 'Not Quick Win'
        result['quick_win_priority'] = 999
        
        # Category 1: Already Clean (Tier 2)
        mask_clean = result['tier2_clean'] == True
        result.loc[mask_clean, 'quick_win_category'] = 'Already Clean'
        result.loc[mask_clean, 'quick_win_priority'] = 0
        
        # Category 2: Tier 1 Clean, 1 issue to Tier 2
        mask_t1_clean = result['tier1_clean'] == True
        mask_1_to_t2 = mask_t1_clean & (result['tier2_failing_count'] == 1) & ~mask_clean
        result.loc[mask_1_to_t2, 'quick_win_category'] = '1 Issue to Tier 2'
        result.loc[mask_1_to_t2, 'quick_win_priority'] = 1
        
        # Category 3: 1 issue to Tier 1
        mask_1_to_t1 = (result['tier1_failing_count'] == 1) & ~mask_t1_clean
        result.loc[mask_1_to_t1, 'quick_win_category'] = '1 Issue to Tier 1'
        result.loc[mask_1_to_t1, 'quick_win_priority'] = 2
        
        # Category 4: 2 issues to Tier 1
        mask_2_to_t1 = (result['tier1_failing_count'] == 2) & ~mask_t1_clean
        result.loc[mask_2_to_t1, 'quick_win_category'] = '2 Issues to Tier 1'
        result.loc[mask_2_to_t1, 'quick_win_priority'] = 3
        
        # Category 5: Low effort (< 4 hours to clean)
        mask_low_effort = (
            (result['total_effort_hours'] > 0) & 
            (result['total_effort_hours'] <= self.config.quick_win_max_hours) &
            ~mask_clean & ~mask_1_to_t2 & ~mask_1_to_t1 & ~mask_2_to_t1
        )
        result.loc[mask_low_effort, 'quick_win_category'] = 'Low Effort (<4 hrs)'
        result.loc[mask_low_effort, 'quick_win_priority'] = 4
        
        # Quick win flag
        result['is_quick_win'] = result['quick_win_category'] != 'Not Quick Win'
        
        # Log summary
        logger.info(f"\n  QUICK WIN DISTRIBUTION:")
        for category in ['Already Clean', '1 Issue to Tier 2', '1 Issue to Tier 1', 
                         '2 Issues to Tier 1', 'Low Effort (<4 hrs)', 'Not Quick Win']:
            count = (result['quick_win_category'] == category).sum()
            pct = count / len(result) * 100
            icon = "✅" if category == 'Already Clean' else "⚡" if 'Issue' in category or 'Low' in category else "⏳"
            logger.info(f"    {icon} {category}: {count:,} ({pct:.1f}%)")
        
        self.summary_stats['quick_wins'] = {
            'already_clean': int(mask_clean.sum()),
            '1_issue_to_t2': int(mask_1_to_t2.sum()),
            '1_issue_to_t1': int(mask_1_to_t1.sum()),
            '2_issues_to_t1': int(mask_2_to_t1.sum()),
            'low_effort': int(mask_low_effort.sum()),
            'not_quick_win': int((result['quick_win_category'] == 'Not Quick Win').sum())
        }
        
        return result
    
    def calculate_path_to_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate path-to-clean recommendations for each patient
        """
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING PATH-TO-CLEAN RECOMMENDATIONS")
        logger.info("=" * 70)
        
        result = df.copy()
        
        def get_path_to_clean(row):
            """Generate prioritized action list"""
            actions = []
            
            # Tier 1 issues first (by priority)
            t1_issues = []
            for criterion_id, criterion in self.config.tier1_criteria.items():
                if not row.get(f't1_{criterion_id}', True):
                    issue_count = row.get(f't1_{criterion_id}_count', 0)
                    effort = row.get(f't1_{criterion_id}_effort_hrs', 0)
                    t1_issues.append({
                        'tier': 1,
                        'criterion': criterion_id,
                        'name': criterion.name,
                        'priority': criterion.priority,
                        'issues': issue_count,
                        'effort_hrs': effort,
                        'responsible': criterion.responsible_role,
                        'action': criterion.resolution_action
                    })
            
            # Sort by priority
            t1_issues.sort(key=lambda x: x['priority'])
            actions.extend(t1_issues)
            
            # Tier 2 issues (only if Tier 1 is clean or we show all)
            t2_issues = []
            for criterion_id, criterion in self.config.tier2_criteria.items():
                if not row.get(f't2_{criterion_id}', True):
                    issue_count = row.get(f't2_{criterion_id}_count', 0)
                    effort = row.get(f't2_{criterion_id}_effort_hrs', 0)
                    t2_issues.append({
                        'tier': 2,
                        'criterion': criterion_id,
                        'name': criterion.name,
                        'priority': criterion.priority,
                        'issues': issue_count,
                        'effort_hrs': effort,
                        'responsible': criterion.responsible_role,
                        'action': criterion.resolution_action
                    })
            
            t2_issues.sort(key=lambda x: x['priority'])
            actions.extend(t2_issues)
            
            return actions
        
        # Generate path for each patient (store as JSON string for now)
        def format_path(row):
            actions = get_path_to_clean(row)
            if not actions:
                return 'Clean - No actions needed'
            
            formatted = []
            for i, action in enumerate(actions[:5], 1):  # Top 5 actions
                formatted.append(f"{i}. [{action['responsible']}] {action['name']}: {action['action']}")
            
            return ' | '.join(formatted)
        
        def get_next_action(row):
            actions = get_path_to_clean(row)
            if not actions:
                return None
            return actions[0]['action']
        
        def get_next_responsible(row):
            actions = get_path_to_clean(row)
            if not actions:
                return None
            return actions[0]['responsible']
        
        result['path_to_clean'] = result.apply(format_path, axis=1)
        result['next_action'] = result.apply(get_next_action, axis=1)
        result['next_responsible'] = result.apply(get_next_responsible, axis=1)
        
        # Count by responsible role
        logger.info(f"\n  NEXT ACTION BY ROLE:")
        role_counts = result['next_responsible'].value_counts()
        for role, count in role_counts.items():
            if role:
                logger.info(f"    {role}: {count:,} patients")
        
        return result
    
    def calculate_site_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate site-level clean patient summary
        """
        logger.info("\n" + "=" * 70)
        logger.info("CALCULATING SITE-LEVEL SUMMARY")
        logger.info("=" * 70)
        
        # Aggregation
        agg_dict = {
            'patient_key': 'count',
            'tier1_clean': 'sum',
            'tier2_clean': 'sum',
            'tier1_effort_hours': 'sum',
            'tier2_effort_hours': 'sum',
            'total_effort_hours': 'sum',
            'is_quick_win': 'sum'
        }
        
        # Filter to available columns
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        site_df = df.groupby(['study_id', 'site_id']).agg(agg_dict).reset_index()
        
        # Rename
        site_df = site_df.rename(columns={
            'patient_key': 'patient_count',
            'tier1_clean': 'tier1_clean_count',
            'tier2_clean': 'tier2_clean_count',
            'is_quick_win': 'quick_win_count'
        })
        
        # Calculate rates
        site_df['tier1_clean_rate'] = (site_df['tier1_clean_count'] / site_df['patient_count'] * 100).round(1)
        site_df['tier2_clean_rate'] = (site_df['tier2_clean_count'] / site_df['patient_count'] * 100).round(1)
        site_df['quick_win_rate'] = (site_df['quick_win_count'] / site_df['patient_count'] * 100).round(1)
        
        # Calculate effort per patient
        not_clean = site_df['patient_count'] - site_df['tier2_clean_count']
        site_df['avg_effort_per_patient'] = np.where(
            not_clean > 0,
            (site_df['total_effort_hours'] / not_clean).round(2),
            0
        )
        
        # Site readiness score
        site_df['site_readiness_score'] = (
            site_df['tier1_clean_rate'] * 0.4 +
            site_df['tier2_clean_rate'] * 0.4 +
            site_df['quick_win_rate'] * 0.2
        ).round(1)
        
        # Readiness tier
        def get_readiness_tier(score):
            if score >= 90: return 'Excellent'
            elif score >= 75: return 'Good'
            elif score >= 60: return 'Fair'
            elif score >= 40: return 'Needs Work'
            else: return 'At Risk'
        
        site_df['readiness_tier'] = site_df['site_readiness_score'].apply(get_readiness_tier)
        
        logger.info(f"\n  Sites analyzed: {len(site_df):,}")
        logger.info(f"  Average T1 Clean Rate: {site_df['tier1_clean_rate'].mean():.1f}%")
        logger.info(f"  Average T2 Clean Rate: {site_df['tier2_clean_rate'].mean():.1f}%")
        
        return site_df
    
    def calculate_full_clean_patient(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete enhanced clean patient calculation
        """
        logger.info("\n" + "=" * 70)
        logger.info("a6on-i - ENHANCED CLEAN PATIENT v2.0")
        logger.info("=" * 70)
        logger.info(f"Patients: {len(df):,}")
        
        start_time = datetime.now()
        
        # Step 1: Tier 1
        result = self.calculate_tier1_clinical_clean(df)
        
        # Step 2: Tier 2
        result = self.calculate_tier2_operational_clean(result)
        
        # Step 3: Quick Wins
        result = self.identify_quick_wins(result)
        
        # Step 4: Path to Clean
        result = self.calculate_path_to_clean(result)
        
        # Step 5: Site Summary
        site_df = self.calculate_site_summary(result)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 70)
        logger.info("CLEAN PATIENT CALCULATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.2f} seconds")
        
        return result, site_df
    
    def generate_blocking_summary(self) -> pd.DataFrame:
        """Generate summary of all blocking reasons"""
        
        blocking_data = []
        
        # Tier 1
        for criterion_id, data in self.tier1_results.items():
            criterion = data['criterion']
            blocking_data.append({
                'tier': 1,
                'criterion_id': criterion_id,
                'name': criterion.name,
                'block_type': 'Hard Block',
                'failures': data['failures'],
                'total_issues': data['total_issues'],
                'effort_hours': data['total_effort_hours'],
                'priority': criterion.priority,
                'responsible': criterion.responsible_role,
                'action': criterion.resolution_action
            })
        
        # Tier 2
        for criterion_id, data in self.tier2_results.items():
            criterion = data['criterion']
            blocking_data.append({
                'tier': 2,
                'criterion_id': criterion_id,
                'name': criterion.name,
                'block_type': 'Soft Block',
                'failures': data['failures'],
                'total_issues': data['total_issues'],
                'effort_hours': 0,  # Not calculated for Tier 2 summary
                'priority': criterion.priority,
                'responsible': criterion.responsible_role,
                'action': criterion.resolution_action
            })
        
        return pd.DataFrame(blocking_data).sort_values(['tier', 'priority'])


# =============================================================================
# MAIN ENGINE
# =============================================================================

class EnhancedCleanPatientEngine:
    """Main engine for Enhanced Clean Patient calculation"""
    
    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        self.site_df = None
        self.calculator = EnhancedCleanPatientCalculator()
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(self.df):,} patients with {len(self.df.columns)} columns")
        return self.df
    
    def run(self) -> pd.DataFrame:
        self.df, self.site_df = self.calculator.calculate_full_clean_patient(self.df)
        return self.df
    
    def save_outputs(self) -> Dict[str, Path]:
        logger.info("\n" + "=" * 60)
        logger.info("SAVING ENHANCED CLEAN PATIENT OUTPUTS")
        logger.info("=" * 60)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # Save to PostgreSQL
        pg_writer = get_pg_writer()
        
        # Patient-level clean status
        success, msg = pg_writer.safe_to_postgres(self.df, 'patient_clean_status', if_exists='replace')
        if success:
            logger.info(f"✅ Saved patient_clean_status to PostgreSQL: {len(self.df):,} rows")
        else:
            logger.warning(f"PostgreSQL save failed, using parquet: {msg}")
            patient_path = self.output_dir / 'patient_clean_status.parquet'
            self.df.to_parquet(patient_path, index=False)
        
        # Site-level summary
        if self.site_df is not None:
            success, msg = pg_writer.safe_to_postgres(self.site_df, 'site_clean_summary', if_exists='replace')
            if success:
                logger.info(f"✅ Saved site_clean_summary to PostgreSQL: {len(self.site_df):,} rows")
            else:
                site_path = self.output_dir / 'site_clean_summary.parquet'
                self.site_df.to_parquet(site_path, index=False)
        
        # 3. Blocking reasons summary
        blocking_df = self.calculator.generate_blocking_summary()
        blocking_path = self.output_dir / 'blocking_reasons_summary.csv'
        blocking_df.to_csv(blocking_path, index=False)
        saved_files['blocking_summary'] = blocking_path
        logger.info(f"✅ Saved blocking summary: {blocking_path}")
        
        # 4. Quick wins list
        quick_wins_cols = ['patient_key', 'study_id', 'site_id', 'subject_id',
                           'quick_win_category', 'quick_win_priority',
                           'tier1_failing_count', 'tier2_failing_count',
                           'total_effort_hours', 'next_action', 'next_responsible']
        available_cols = [c for c in quick_wins_cols if c in self.df.columns]
        
        quick_wins = self.df[self.df['is_quick_win'] & (self.df['quick_win_category'] != 'Already Clean')]
        quick_wins = quick_wins[available_cols].sort_values(['quick_win_priority', 'total_effort_hours'])
        
        qw_path = self.output_dir / 'quick_wins_actionable.csv'
        quick_wins.to_csv(qw_path, index=False)
        saved_files['quick_wins'] = qw_path
        logger.info(f"✅ Saved quick wins: {len(quick_wins):,} patients")
        
        # 5. Not clean patients with path
        not_clean_cols = ['patient_key', 'study_id', 'site_id', 'subject_id',
                          'tier1_clean', 'tier2_clean', 'tier1_primary_block', 'tier2_primary_block',
                          'total_effort_hours', 'total_effort_days', 'path_to_clean']
        available_cols = [c for c in not_clean_cols if c in self.df.columns]
        
        not_clean = self.df[~self.df['tier2_clean']][available_cols].sort_values('total_effort_hours')
        nc_path = self.output_dir / 'not_clean_with_path.csv'
        not_clean.head(5000).to_csv(nc_path, index=False)  # Top 5000 by effort
        saved_files['not_clean'] = nc_path
        logger.info(f"✅ Saved not clean patients: {len(not_clean):,}")
        
        # 6. Summary JSON
        summary = {
            'generated_at': datetime.now().isoformat(),
            'version': '2.0.0',
            'patient_count': len(self.df),
            'tier1': self.calculator.summary_stats.get('tier1', {}),
            'tier2': self.calculator.summary_stats.get('tier2', {}),
            'quick_wins': self.calculator.summary_stats.get('quick_wins', {}),
            'effort_summary': {
                'total_hours_to_clean_all': float(self.df['total_effort_hours'].sum()),
                'total_days_to_clean_all': float(self.df['total_effort_hours'].sum() / 8),
                'avg_hours_per_not_clean': float(self.df[~self.df['tier2_clean']]['total_effort_hours'].mean())
            }
        }
        
        summary_path = self.output_dir / 'clean_patient_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files['summary'] = summary_path
        logger.info(f"✅ Saved summary: {summary_path}")
        
        return saved_files
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 PHASE 2.2 COMPLETE - ENHANCED CLEAN PATIENT v2.0")
        print("=" * 70)
        
        print(f"\n🔢 PATIENTS ANALYZED: {len(self.df):,}")
        
        # Tier 1
        t1_clean = self.df['tier1_clean'].sum()
        t1_pct = t1_clean / len(self.df) * 100
        print(f"\n✅ TIER 1 - CLINICAL CLEAN (7 Hard Blocks):")
        print(f"   Clean: {t1_clean:,} ({t1_pct:.1f}%)")
        print(f"   Not Clean: {len(self.df) - t1_clean:,} ({100-t1_pct:.1f}%)")
        
        # Top Tier 1 blockers
        print(f"\n   Top Blocking Reasons:")
        t1_blocks = self.df['tier1_primary_block'].value_counts().head(5)
        for block, count in t1_blocks.items():
            if block:
                print(f"   - {block}: {count:,}")
        
        # Tier 2
        t2_clean = self.df['tier2_clean'].sum()
        t2_pct = t2_clean / len(self.df) * 100
        print(f"\n✅ TIER 2 - OPERATIONAL CLEAN (7 Soft Blocks):")
        print(f"   Clean: {t2_clean:,} ({t2_pct:.1f}%)")
        print(f"   Not Clean: {len(self.df) - t2_clean:,} ({100-t2_pct:.1f}%)")
        
        # Quick Wins
        print(f"\n⚡ QUICK WINS:")
        qw_counts = self.df['quick_win_category'].value_counts()
        for category, count in qw_counts.items():
            pct = count / len(self.df) * 100
            icon = "✅" if category == 'Already Clean' else "⚡" if 'Issue' in category else "⏳"
            print(f"   {icon} {category}: {count:,} ({pct:.1f}%)")
        
        # Effort Summary
        total_effort = self.df['total_effort_hours'].sum()
        avg_effort = self.df[~self.df['tier2_clean']]['total_effort_hours'].mean()
        print(f"\n⏱️ EFFORT ESTIMATION:")
        print(f"   Total effort to clean all: {total_effort:,.0f} hours ({total_effort/8:,.0f} days)")
        print(f"   Avg effort per not-clean patient: {avg_effort:.1f} hours")
        
        # Next Actions by Role
        print(f"\n👥 NEXT ACTION BY ROLE:")
        role_counts = self.df['next_responsible'].value_counts()
        for role, count in role_counts.head(5).items():
            if role:
                print(f"   {role}: {count:,} patients")
        
        print(f"\n📁 Output Directory: {self.output_dir}")


def main():
    project_root = Path(__file__).parent.parent.parent
    
    # Use enhanced DQI output if available
    input_path = project_root / 'data' / 'processed' / 'analytics' / 'patient_dqi_enhanced.parquet'
    
    if not input_path.exists():
        input_path = project_root / 'data' / 'processed' / 'metrics' / 'patient_metrics.parquet'
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return
    
    output_dir = project_root / 'data' / 'processed' / 'analytics'
    
    engine = EnhancedCleanPatientEngine(input_path, output_dir)
    engine.load_data()
    engine.run()
    engine.save_outputs()
    engine.print_summary()
    
    return engine


if __name__ == '__main__':
    main()
