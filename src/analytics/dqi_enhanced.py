"""
a6on-i - Enhanced DQI Calculator v3.1 (FIXED)
Phase 2.1: 8-Component DQI with Age, Trend, and Criticality Multipliers

FIXES:
- Primary issue detection only considers patients WITH issues
- Signature penalty calculation fixed
- Better handling of "no issues" cases
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

# PostgreSQL integration
from src.database.pg_data_service import get_data_service
from src.database.pg_writer import get_pg_writer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class DQIBand(Enum):
    """DQI Quality Bands"""
    PRISTINE = ("Pristine", 95, 100, "🟢", "Exceptional quality - ready for any review")
    EXCELLENT = ("Excellent", 85, 95, "🟢", "High quality - minor issues only")
    GOOD = ("Good", 75, 85, "🟡", "Acceptable quality - some attention needed")
    FAIR = ("Fair", 65, 75, "🟡", "Below target - action required")
    POOR = ("Poor", 50, 65, "🟠", "Significant issues - immediate attention")
    CRITICAL = ("Critical", 25, 50, "🔴", "Major problems - escalation required")
    EMERGENCY = ("Emergency", 0, 25, "🔴", "Crisis state - urgent intervention")
    
    @property
    def name_str(self): return self.value[0]
    @property
    def min_score(self): return self.value[1]
    @property
    def max_score(self): return self.value[2]
    @property
    def icon(self): return self.value[3]
    @property
    def description(self): return self.value[4]


@dataclass
class DQIComponentConfig:
    """Configuration for a single DQI component"""
    name: str
    weight: float
    description: str
    criticality: float  # Multiplier (1.0 = normal, 1.5 = safety critical)
    penalty_per_issue: float
    max_penalty: float = 100.0
    columns: List[str] = field(default_factory=list)
    age_column: Optional[str] = None


@dataclass 
class EnhancedDQIConfig:
    """Complete Enhanced DQI Configuration"""
    
    # Component configurations
    components: Dict[str, DQIComponentConfig] = field(default_factory=lambda: {
        'safety': DQIComponentConfig(
            name='Safety',
            weight=25.0,
            description='SAE and safety-related discrepancies',
            criticality=1.5,
            penalty_per_issue=15.0,
            columns=['sae_dm_sae_dm_pending', 'sae_safety_sae_safety_pending']
        ),
        'query': DQIComponentConfig(
            name='Query',
            weight=20.0,
            description='Open queries requiring resolution',
            criticality=1.1,
            penalty_per_issue=3.0,
            columns=['total_queries', 'dm_queries', 'clinical_queries']
        ),
        'completeness': DQIComponentConfig(
            name='Completeness',
            weight=15.0,
            description='Missing visits and CRF pages',
            criticality=1.1,
            penalty_per_issue=8.0,
            columns=['visit_missing_visit_count', 'pages_missing_page_count'],
            age_column='visit_visits_overdue_avg_days'
        ),
        'coding': DQIComponentConfig(
            name='Coding',
            weight=12.0,
            description='Uncoded MedDRA and WHODrug terms',
            criticality=1.0,
            penalty_per_issue=5.0,
            columns=['meddra_coding_meddra_uncoded', 'whodrug_coding_whodrug_uncoded']
        ),
        'lab': DQIComponentConfig(
            name='Lab',
            weight=10.0,
            description='Lab name and range issues',
            criticality=1.1,
            penalty_per_issue=10.0,
            columns=['lab_lab_issue_count', 'open_issues_lnr']
        ),
        'sdv': DQIComponentConfig(
            name='SDV',
            weight=8.0,
            description='Source Data Verification completion',
            criticality=1.0,
            penalty_per_issue=0.5,
            columns=['crfs_require_verification_sdv', 'forms_verified']
        ),
        'signature': DQIComponentConfig(
            name='Signature',
            weight=5.0,
            description='Pending and overdue signatures',
            criticality=1.1,
            penalty_per_issue=3.0,  # REDUCED from 5.0
            columns=['crfs_never_signed', 'broken_signatures']
        ),
        'edrr': DQIComponentConfig(
            name='EDRR',
            weight=5.0,
            description='External Data Reconciliation issues',
            criticality=1.0,
            penalty_per_issue=10.0,
            columns=['edrr_edrr_issue_count', 'open_issues_edrr']
        )
    })
    
    # Age multipliers
    age_multipliers: Dict[str, Tuple[int, int, float]] = field(default_factory=lambda: {
        'fresh': (0, 7, 1.0),
        'aging': (8, 14, 1.1),
        'overdue': (15, 21, 1.2),
        'stale': (22, 30, 1.4),
        'critical': (31, 9999, 1.6)
    })
    
    # Trend multipliers
    trend_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'improving': 0.85,
        'stable': 1.0,
        'worsening': 1.25
    })


# =============================================================================
# ENHANCED DQI CALCULATOR
# =============================================================================

class EnhancedDQICalculator:
    """Enhanced 8-Component DQI Calculator with Multipliers"""
    
    def __init__(self, config: EnhancedDQIConfig = None):
        self.config = config or EnhancedDQIConfig()
        self.component_details = {}
        
    def _get_col(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Safely get column or return zeros"""
        if col in df.columns:
            return df[col].fillna(0)
        return pd.Series(0, index=df.index)
    
    def _get_age_multiplier(self, days: pd.Series) -> pd.Series:
        """Calculate age multiplier based on days outstanding"""
        multiplier = pd.Series(1.0, index=days.index)
        
        for level, (min_days, max_days, mult) in self.config.age_multipliers.items():
            mask = (days >= min_days) & (days <= max_days)
            multiplier[mask] = mult
        
        return multiplier
    
    def calculate_safety_component(self, df: pd.DataFrame) -> Dict:
        """Calculate Safety component (25% weight)"""
        config = self.config.components['safety']
        
        sae_dm_pending = self._get_col(df, 'sae_dm_sae_dm_pending')
        sae_safety_pending = self._get_col(df, 'sae_safety_sae_safety_pending')
        
        total_issues = sae_dm_pending + sae_safety_pending
        
        # Base penalty with criticality
        base_penalty = total_issues * config.penalty_per_issue * config.criticality
        final_penalty = np.minimum(base_penalty, config.max_penalty)
        
        raw_score = (100 - final_penalty).clip(0, 100)
        
        return {
            'name': config.name,
            'weight': config.weight,
            'raw_score': raw_score,
            'weighted_score': raw_score * (config.weight / 100),
            'issue_count': total_issues,
            'criticality': config.criticality,
            'patients_with_issues': (total_issues > 0).sum(),
            'total_issues': int(total_issues.sum())
        }
    
    def calculate_query_component(self, df: pd.DataFrame) -> Dict:
        """Calculate Query component (20% weight)"""
        config = self.config.components['query']
        
        total_queries = self._get_col(df, 'total_queries')
        
        base_penalty = total_queries * config.penalty_per_issue * config.criticality
        final_penalty = np.minimum(base_penalty, config.max_penalty)
        
        raw_score = (100 - final_penalty).clip(0, 100)
        
        return {
            'name': config.name,
            'weight': config.weight,
            'raw_score': raw_score,
            'weighted_score': raw_score * (config.weight / 100),
            'issue_count': total_queries,
            'criticality': config.criticality,
            'patients_with_issues': (total_queries > 0).sum(),
            'total_issues': int(total_queries.sum())
        }
    
    def calculate_completeness_component(self, df: pd.DataFrame) -> Dict:
        """Calculate Completeness component (15% weight) with AGE MULTIPLIER"""
        config = self.config.components['completeness']
        
        missing_visits = self._get_col(df, 'visit_missing_visit_count')
        missing_pages = self._get_col(df, 'pages_missing_page_count')
        
        issues = missing_visits + missing_pages
        
        # Get age multiplier
        age_days = self._get_col(df, 'visit_visits_overdue_avg_days')
        age_mult = self._get_age_multiplier(age_days)
        
        # Apply penalties with age and criticality
        base_penalty = issues * config.penalty_per_issue * age_mult * config.criticality
        final_penalty = np.minimum(base_penalty, config.max_penalty)
        
        raw_score = (100 - final_penalty).clip(0, 100)
        
        return {
            'name': config.name,
            'weight': config.weight,
            'raw_score': raw_score,
            'weighted_score': raw_score * (config.weight / 100),
            'issue_count': issues,
            'criticality': config.criticality,
            'age_multiplier_applied': True,
            'avg_age_days': float(age_days[age_days > 0].mean()) if (age_days > 0).any() else 0,
            'patients_with_issues': (issues > 0).sum(),
            'total_issues': int(issues.sum())
        }
    
    def calculate_coding_component(self, df: pd.DataFrame) -> Dict:
        """Calculate Coding component (12% weight)"""
        config = self.config.components['coding']
        
        # Use coding completion rate if available
        if 'coding_completion_rate' in df.columns:
            raw_score = df['coding_completion_rate'].fillna(100).clip(0, 100)
            issues = self._get_col(df, 'meddra_coding_meddra_uncoded') + \
                     self._get_col(df, 'whodrug_coding_whodrug_uncoded')
        else:
            meddra = self._get_col(df, 'meddra_coding_meddra_uncoded')
            whodrug = self._get_col(df, 'whodrug_coding_whodrug_uncoded')
            issues = meddra + whodrug
            
            base_penalty = issues * config.penalty_per_issue * config.criticality
            final_penalty = np.minimum(base_penalty, config.max_penalty)
            raw_score = (100 - final_penalty).clip(0, 100)
        
        return {
            'name': config.name,
            'weight': config.weight,
            'raw_score': raw_score,
            'weighted_score': raw_score * (config.weight / 100),
            'issue_count': issues,
            'criticality': config.criticality,
            'patients_with_issues': (issues > 0).sum(),
            'total_issues': int(issues.sum())
        }
    
    def calculate_lab_component(self, df: pd.DataFrame) -> Dict:
        """Calculate Lab component (10% weight)"""
        config = self.config.components['lab']
        
        lab_issues = self._get_col(df, 'lab_lab_issue_count')
        lnr_issues = self._get_col(df, 'open_issues_lnr')
        
        issues = lab_issues + lnr_issues
        
        base_penalty = issues * config.penalty_per_issue * config.criticality
        final_penalty = np.minimum(base_penalty, config.max_penalty)
        
        raw_score = (100 - final_penalty).clip(0, 100)
        
        return {
            'name': config.name,
            'weight': config.weight,
            'raw_score': raw_score,
            'weighted_score': raw_score * (config.weight / 100),
            'issue_count': issues,
            'criticality': config.criticality,
            'patients_with_issues': (issues > 0).sum(),
            'total_issues': int(issues.sum())
        }
    
    def calculate_sdv_component(self, df: pd.DataFrame) -> Dict:
        """Calculate SDV component (8% weight)"""
        config = self.config.components['sdv']
        
        forms_required = self._get_col(df, 'crfs_require_verification_sdv')
        forms_verified = self._get_col(df, 'forms_verified')
        
        # Calculate SDV rate
        total_forms = forms_required + forms_verified
        sdv_rate = np.where(
            total_forms > 0,
            (forms_verified / total_forms * 100),
            100
        )
        
        raw_score = pd.Series(sdv_rate, index=df.index).clip(0, 100)
        issues = forms_required  # Forms still needing SDV
        
        return {
            'name': config.name,
            'weight': config.weight,
            'raw_score': raw_score,
            'weighted_score': raw_score * (config.weight / 100),
            'issue_count': issues,
            'criticality': config.criticality,
            'patients_with_issues': (issues > 0).sum(),
            'total_issues': int(issues.sum()),
            'avg_sdv_rate': float(raw_score.mean())
        }
    
    def calculate_signature_component(self, df: pd.DataFrame) -> Dict:
        """Calculate Signature component (5% weight) - FIXED"""
        config = self.config.components['signature']
        
        # Get signature issues
        never_signed = self._get_col(df, 'crfs_never_signed')
        broken_sigs = self._get_col(df, 'broken_signatures')
        
        # Overdue signatures with age weighting
        overdue_45 = self._get_col(df, 'crfs_overdue_for_signs_within_45_days_of_data_entry')
        overdue_90 = self._get_col(df, 'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry')
        overdue_90plus = self._get_col(df, 'crfs_overdue_for_signs_beyond_90_days_of_data_entry')
        
        # Binary flag: has ANY signature issue (not count-based)
        has_sig_issue = (
            (never_signed > 0) | 
            (broken_sigs > 0) | 
            (overdue_45 > 0) | 
            (overdue_90 > 0) | 
            (overdue_90plus > 0)
        ).astype(int)
        
        # Severity-based penalty (not count-based)
        # Having broken signatures is worse than just unsigned
        severity_score = (
            (broken_sigs > 0).astype(int) * 30 +  # Broken = -30
            (overdue_90plus > 0).astype(int) * 25 +  # Very old = -25
            (overdue_90 > 0).astype(int) * 15 +  # Aging = -15
            (overdue_45 > 0).astype(int) * 10 +  # Fresh overdue = -10
            (never_signed > 0).astype(int) * 10  # Never signed = -10
        )
        
        # Apply criticality
        final_penalty = np.minimum(severity_score * config.criticality, config.max_penalty)
        raw_score = (100 - final_penalty).clip(0, 100)
        
        # Total issues for reporting
        total_issues = never_signed + broken_sigs + overdue_45 + overdue_90 + overdue_90plus
        
        return {
            'name': config.name,
            'weight': config.weight,
            'raw_score': raw_score,
            'weighted_score': raw_score * (config.weight / 100),
            'issue_count': has_sig_issue,  # Binary for primary issue detection
            'criticality': config.criticality,
            'patients_with_issues': has_sig_issue.sum(),
            'total_issues': int(total_issues.sum()),
            'breakdown': {
                'never_signed': int(never_signed.sum()),
                'broken': int(broken_sigs.sum()),
                'overdue_45': int(overdue_45.sum()),
                'overdue_90': int(overdue_90.sum()),
                'overdue_90plus': int(overdue_90plus.sum())
            }
        }
    
    def calculate_edrr_component(self, df: pd.DataFrame) -> Dict:
        """Calculate EDRR component (5% weight)"""
        config = self.config.components['edrr']
        
        edrr_issues = self._get_col(df, 'edrr_edrr_issue_count')
        open_edrr = self._get_col(df, 'open_issues_edrr')
        
        issues = np.maximum(edrr_issues, open_edrr)
        
        base_penalty = issues * config.penalty_per_issue * config.criticality
        final_penalty = np.minimum(base_penalty, config.max_penalty)
        
        raw_score = (100 - final_penalty).clip(0, 100)
        
        return {
            'name': config.name,
            'weight': config.weight,
            'raw_score': raw_score,
            'weighted_score': raw_score * (config.weight / 100),
            'issue_count': issues,
            'criticality': config.criticality,
            'patients_with_issues': (issues > 0).sum(),
            'total_issues': int(issues.sum())
        }
    
    def calculate_all_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all 8 DQI components"""
        
        logger.info("=" * 70)
        logger.info("CALCULATING 8-COMPONENT DQI WITH MULTIPLIERS")
        logger.info("=" * 70)
        
        result = df.copy()
        
        # Calculate each component
        components = {
            'safety': self.calculate_safety_component(df),
            'query': self.calculate_query_component(df),
            'completeness': self.calculate_completeness_component(df),
            'coding': self.calculate_coding_component(df),
            'lab': self.calculate_lab_component(df),
            'sdv': self.calculate_sdv_component(df),
            'signature': self.calculate_signature_component(df),
            'edrr': self.calculate_edrr_component(df)
        }
        
        self.component_details = components
        
        # Add component scores to result
        total_weighted = pd.Series(0.0, index=df.index)
        
        for comp_name, comp_data in components.items():
            result[f'dqi_{comp_name}_raw'] = comp_data['raw_score']
            result[f'dqi_{comp_name}_weighted'] = comp_data['weighted_score']
            result[f'dqi_{comp_name}_issues'] = comp_data['issue_count']
            
            total_weighted += comp_data['weighted_score']
            
            # Log
            patients_affected = comp_data['patients_with_issues']
            total_issues = comp_data['total_issues']
            avg_score = comp_data['raw_score'].mean()
            crit = comp_data['criticality']
            
            logger.info(f"\n  {comp_data['name'].upper()} (Weight: {comp_data['weight']}%, Criticality: {crit}x)")
            logger.info(f"    Patients with issues: {patients_affected:,}")
            logger.info(f"    Total issues: {total_issues:,}")
            logger.info(f"    Average raw score: {avg_score:.1f}")
        
        # Final DQI
        result['dqi_score'] = total_weighted.clip(0, 100)
        
        return result
    
    def assign_dqi_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign DQI bands"""
        result = df.copy()
        
        def get_band(score):
            if score >= 95: return 'Pristine'
            elif score >= 85: return 'Excellent'
            elif score >= 75: return 'Good'
            elif score >= 65: return 'Fair'
            elif score >= 50: return 'Poor'
            elif score >= 25: return 'Critical'
            else: return 'Emergency'
        
        def get_icon(score):
            if score >= 85: return '🟢'
            elif score >= 65: return '🟡'
            elif score >= 50: return '🟠'
            else: return '🔴'
        
        result['dqi_band'] = result['dqi_score'].apply(get_band)
        result['dqi_band_icon'] = result['dqi_score'].apply(get_icon)
        
        # Log distribution
        logger.info(f"\n  DQI BAND DISTRIBUTION:")
        for band in ['Pristine', 'Excellent', 'Good', 'Fair', 'Poor', 'Critical', 'Emergency']:
            count = (result['dqi_band'] == band).sum()
            if count > 0:
                pct = count / len(result) * 100
                logger.info(f"    {band}: {count:,} ({pct:.1f}%)")
        
        return result
    
    def identify_primary_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify primary issue for each patient - FIXED"""
        result = df.copy()
        
        # Get issue count columns
        issue_cols = {
            'safety': 'dqi_safety_issues',
            'query': 'dqi_query_issues', 
            'completeness': 'dqi_completeness_issues',
            'coding': 'dqi_coding_issues',
            'lab': 'dqi_lab_issues',
            'sdv': 'dqi_sdv_issues',
            'signature': 'dqi_signature_issues',
            'edrr': 'dqi_edrr_issues'
        }
        
        available_cols = {k: v for k, v in issue_cols.items() if v in result.columns}
        
        def get_primary_issue(row):
            """Find the component with the most issues (not lowest score)"""
            issues = {}
            for comp_name, col in available_cols.items():
                issue_count = row[col]
                if issue_count > 0:
                    issues[comp_name] = issue_count
            
            if not issues:
                return 'none'  # No issues!
            
            # Return component with most issues
            return max(issues, key=issues.get)
        
        def get_worst_components(row):
            """Get top 3 components with issues"""
            issues = {}
            for comp_name, col in available_cols.items():
                issue_count = row[col]
                if issue_count > 0:
                    issues[comp_name] = issue_count
            
            if not issues:
                return 'none'
            
            # Sort by issue count descending
            sorted_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)[:3]
            return '; '.join([f"{name}:{int(count)}" for name, count in sorted_issues])
        
        result['dqi_primary_issue'] = result.apply(get_primary_issue, axis=1)
        result['dqi_worst_components'] = result.apply(get_worst_components, axis=1)
        
        # Log primary issue distribution
        logger.info(f"\n  PRIMARY ISSUE DISTRIBUTION:")
        issue_counts = result['dqi_primary_issue'].value_counts()
        for issue, count in issue_counts.items():
            pct = count / len(result) * 100
            logger.info(f"    {issue}: {count:,} ({pct:.1f}%)")
        
        return result
    
    def calculate_component_grades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign letter grades to each component"""
        result = df.copy()
        
        def score_to_grade(score):
            if score >= 95: return 'A+'
            elif score >= 90: return 'A'
            elif score >= 85: return 'A-'
            elif score >= 80: return 'B+'
            elif score >= 75: return 'B'
            elif score >= 70: return 'B-'
            elif score >= 65: return 'C+'
            elif score >= 60: return 'C'
            elif score >= 55: return 'C-'
            elif score >= 50: return 'D'
            else: return 'F'
        
        for comp_name in self.config.components.keys():
            col = f'dqi_{comp_name}_raw'
            if col in result.columns:
                result[f'dqi_{comp_name}_grade'] = result[col].apply(score_to_grade)
        
        return result
    
    def calculate_full_dqi(self, df: pd.DataFrame, historical_df: pd.DataFrame = None) -> pd.DataFrame:
        """Run complete enhanced DQI calculation"""
        
        logger.info("\n" + "=" * 70)
        logger.info("a6on-i - ENHANCED DQI CALCULATOR v3.1")
        logger.info("=" * 70)
        logger.info(f"Patients: {len(df):,}")
        logger.info(f"Formula: DQI = 100 - Σ(Penalty × Weight × Age × Criticality)")
        
        start_time = datetime.now()
        
        # Step 1: Calculate components
        result = self.calculate_all_components(df)
        
        # Step 2: Assign bands
        result = self.assign_dqi_bands(result)
        
        # Step 3: Identify primary issues (FIXED)
        result = self.identify_primary_issues(result)
        
        # Step 4: Component grades
        result = self.calculate_component_grades(result)
        
        # Step 5: Trend (if historical data)
        if historical_df is not None:
            result = self._apply_trend(result, historical_df)
        else:
            result['dqi_trend'] = 'stable'
            result['dqi_trend_multiplier'] = 1.0
            logger.info("\n  No historical data - trend = stable")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 70)
        logger.info("DQI CALCULATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Mean DQI: {result['dqi_score'].mean():.2f}")
        logger.info(f"Median DQI: {result['dqi_score'].median():.2f}")
        logger.info(f"Range: {result['dqi_score'].min():.2f} - {result['dqi_score'].max():.2f}")
        
        return result
    
    def _apply_trend(self, df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
        """Apply trend multiplier"""
        # Implementation for trend analysis
        return df
    
    def generate_component_summary(self) -> pd.DataFrame:
        """Generate component summary"""
        summary_data = []
        
        for comp_name, comp_data in self.component_details.items():
            summary_data.append({
                'component': comp_data['name'],
                'weight': comp_data['weight'],
                'criticality': comp_data['criticality'],
                'patients_with_issues': comp_data['patients_with_issues'],
                'total_issues': comp_data['total_issues'],
                'avg_raw_score': round(comp_data['raw_score'].mean(), 2),
                'min_raw_score': round(comp_data['raw_score'].min(), 2),
                'max_raw_score': round(comp_data['raw_score'].max(), 2),
                'contribution': round(comp_data['weighted_score'].mean(), 2)
            })
        
        return pd.DataFrame(summary_data)


# =============================================================================
# MAIN ENGINE
# =============================================================================

class EnhancedDQIEngine:
    """Main engine for Enhanced DQI"""
    
    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        self.calculator = EnhancedDQICalculator()
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(self.df):,} patients with {len(self.df.columns)} columns")
        return self.df
    
    def run(self, historical_path: Path = None) -> pd.DataFrame:
        historical_df = None
        if historical_path and historical_path.exists():
            historical_df = pd.read_parquet(historical_path)
        
        self.df = self.calculator.calculate_full_dqi(self.df, historical_df)
        return self.df
    
    def save_outputs(self) -> Dict[str, Path]:
        logger.info("\n" + "=" * 60)
        logger.info("SAVING ENHANCED DQI OUTPUTS")
        logger.info("=" * 60)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # 1. Patient DQI - Save to PostgreSQL
        from src.database.pg_writer import safe_to_postgres
        success, msg = safe_to_postgres(self.df, 'patient_dqi_enhanced', if_exists='replace')
        if success:
            logger.info(f"✅ Saved patient_dqi_enhanced to PostgreSQL: {len(self.df):,} rows")
            saved_files['patient_dqi'] = 'PostgreSQL:patient_dqi_enhanced'
        else:
            logger.warning(f"PostgreSQL write failed: {msg}")
            patient_path = self.output_dir / 'patient_dqi_enhanced.parquet'
            self.df.to_parquet(patient_path, index=False)
            saved_files['patient_dqi'] = patient_path

        
        # 2. Component summary
        comp_summary = self.calculator.generate_component_summary()
        comp_path = self.output_dir / 'dqi_component_summary.csv'
        comp_summary.to_csv(comp_path, index=False)
        saved_files['component_summary'] = comp_path
        logger.info(f"✅ Saved component summary: {comp_path}")
        
        # 3. Band distribution
        band_dist = self.df.groupby('dqi_band').agg({
            'patient_key': 'count',
            'dqi_score': ['mean', 'min', 'max']
        }).reset_index()
        band_dist.columns = ['band', 'count', 'mean', 'min', 'max']
        band_path = self.output_dir / 'dqi_band_distribution.csv'
        band_dist.to_csv(band_path, index=False)
        saved_files['band_distribution'] = band_path
        logger.info(f"✅ Saved band distribution: {band_path}")
        
        # 4. Primary issues
        issue_dist = self.df['dqi_primary_issue'].value_counts().reset_index()
        issue_dist.columns = ['primary_issue', 'count']
        issue_dist['pct'] = (issue_dist['count'] / len(self.df) * 100).round(2)
        issue_path = self.output_dir / 'dqi_primary_issues.csv'
        issue_dist.to_csv(issue_path, index=False)
        saved_files['primary_issues'] = issue_path
        logger.info(f"✅ Saved primary issues: {issue_path}")
        
        # 5. Low DQI patients
        low_dqi_cols = ['patient_key', 'study_id', 'site_id', 'subject_id',
                        'dqi_score', 'dqi_band', 'dqi_primary_issue', 'dqi_worst_components']
        available_cols = [c for c in low_dqi_cols if c in self.df.columns]
        
        low_dqi = self.df[self.df['dqi_score'] < 85][available_cols].sort_values('dqi_score')
        low_path = self.output_dir / 'dqi_low_score_patients.csv'
        low_dqi.to_csv(low_path, index=False)
        saved_files['low_score_patients'] = low_path
        logger.info(f"✅ Saved low DQI patients: {len(low_dqi):,}")
        
        # 6. Summary JSON
        summary = {
            'generated_at': datetime.now().isoformat(),
            'version': '3.1.0',
            'patient_count': len(self.df),
            'dqi_statistics': {
                'mean': round(self.df['dqi_score'].mean(), 2),
                'median': round(self.df['dqi_score'].median(), 2),
                'std': round(self.df['dqi_score'].std(), 2),
                'min': round(self.df['dqi_score'].min(), 2),
                'max': round(self.df['dqi_score'].max(), 2)
            },
            'band_distribution': self.df['dqi_band'].value_counts().to_dict(),
            'primary_issue_distribution': self.df['dqi_primary_issue'].value_counts().to_dict(),
            'patients_with_no_issues': int((self.df['dqi_primary_issue'] == 'none').sum())
        }
        
        summary_path = self.output_dir / 'dqi_enhanced_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary'] = summary_path
        logger.info(f"✅ Saved summary: {summary_path}")
        
        return saved_files
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 PHASE 2.1 COMPLETE - ENHANCED DQI CALCULATOR v3.1")
        print("=" * 70)
        
        print(f"\n🔢 PATIENTS ANALYZED: {len(self.df):,}")
        
        # Patients with no issues
        no_issues = (self.df['dqi_primary_issue'] == 'none').sum()
        no_issues_pct = no_issues / len(self.df) * 100
        
        print(f"\n📈 DQI STATISTICS:")
        print(f"   Mean:   {self.df['dqi_score'].mean():.2f}")
        print(f"   Median: {self.df['dqi_score'].median():.2f}")
        print(f"   Std:    {self.df['dqi_score'].std():.2f}")
        print(f"   Range:  {self.df['dqi_score'].min():.2f} - {self.df['dqi_score'].max():.2f}")
        
        print(f"\n🎯 DQI BAND DISTRIBUTION:")
        for band in ['Pristine', 'Excellent', 'Good', 'Fair', 'Poor', 'Critical', 'Emergency']:
            count = (self.df['dqi_band'] == band).sum()
            if count > 0:
                pct = count / len(self.df) * 100
                icon = '🟢' if band in ['Pristine', 'Excellent'] else '🟡' if band in ['Good', 'Fair'] else '🟠' if band == 'Poor' else '🔴'
                bar = "█" * max(1, int(pct / 2))
                print(f"   {icon} {band:12} {count:>6,} ({pct:>5.1f}%) {bar}")
        
        print(f"\n🔍 PRIMARY ISSUE BREAKDOWN:")
        print(f"   {'none (clean)':<15} {no_issues:>6,} ({no_issues_pct:>5.1f}%) ← No issues!")
        
        issue_counts = self.df[self.df['dqi_primary_issue'] != 'none']['dqi_primary_issue'].value_counts()
        for issue, count in issue_counts.items():
            pct = count / len(self.df) * 100
            print(f"   {issue:<15} {count:>6,} ({pct:>5.1f}%)")
        
        print(f"\n⚖️ COMPONENT WEIGHTS & CRITICALITY:")
        for comp_name, comp_config in self.calculator.config.components.items():
            print(f"   {comp_config.name:12} {comp_config.weight:>5.1f}%  (×{comp_config.criticality} criticality)")
        
        print(f"\n📁 Output Directory: {self.output_dir}")


def main():
    project_root = Path(__file__).parent.parent.parent
    
    input_path = project_root / 'data' / 'processed' / 'metrics' / 'patient_metrics.parquet'
    output_dir = project_root / 'data' / 'processed' / 'analytics'
    
    if not input_path.exists():
        input_path = project_root / 'data' / 'processed' / 'segments' / 'unified_patient_record_segmented.parquet'
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return
    
    engine = EnhancedDQIEngine(input_path, output_dir)
    engine.load_data()
    engine.run()
    engine.save_outputs()
    engine.print_summary()
    
    return engine


if __name__ == '__main__':
    main()
