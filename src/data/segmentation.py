"""
SANCHALAK AI - Population Segmentation Engine v1.0
Phase 1.4: Patient Segmentation & Cohort Creation

Features:
- Segment by subject_status
- Identify DB Lock eligible population
- Tag screen failures (exclude from analysis)
- Create analysis cohorts
- Generate segmentation statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional

# PostgreSQL data access
try:
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class PopulationSegmenter:
    """
    Population Segmentation Engine for SANCHALAK AI
    
    Segments:
    1. By Subject Status (Ongoing, Completed, Discontinued, etc.)
    2. DB Lock Eligible (excludes screen failures, screening)
    3. Analysis Cohorts (active monitoring, closeout, etc.)
    4. Risk-based segments
    """
    
    # Subject status categories
    STATUS_CATEGORIES = {
        'active': ['Ongoing', 'Screening'],
        'completed': ['Completed'],
        'terminated': ['Discontinued', 'Screen Failure'],
        'unknown': ['Unknown']
    }
    
    # DB Lock eligibility rules
    DB_LOCK_ELIGIBLE_STATUSES = ['Ongoing', 'Completed', 'Discontinued']
    DB_LOCK_EXCLUDE_STATUSES = ['Screen Failure', 'Screening', 'Unknown']
    
    # Analysis cohort definitions
    COHORT_DEFINITIONS = {
        'active_monitoring': {
            'description': 'Patients requiring active CRA monitoring',
            'statuses': ['Ongoing', 'Screening'],
            'include_screen_failure': False
        },
        'closeout_ready': {
            'description': 'Patients ready for closeout activities',
            'statuses': ['Completed'],
            'include_screen_failure': False
        },
        'db_lock_eligible': {
            'description': 'Patients eligible for database lock',
            'statuses': ['Ongoing', 'Completed', 'Discontinued'],
            'include_screen_failure': False
        },
        'data_quality_focus': {
            'description': 'Patients with data quality issues requiring attention',
            'statuses': ['Ongoing', 'Completed', 'Discontinued'],
            'include_screen_failure': False,
            'additional_filter': 'has_issues'
        },
        'safety_monitoring': {
            'description': 'Patients with SAE requiring safety monitoring',
            'statuses': ['Ongoing', 'Completed', 'Discontinued'],
            'include_screen_failure': False,
            'additional_filter': 'has_sae'
        },
        'excluded_from_analysis': {
            'description': 'Screen failures and other excluded patients',
            'statuses': ['Screen Failure'],
            'include_screen_failure': True
        }
    }
    
    def __init__(self, upr_path: Path):
        """Initialize with UPR path"""
        self.upr_path = upr_path
        self.upr = None
        self.segments = {}
        self.cohorts = {}
        self.stats = {}
        
    def load_upr(self) -> pd.DataFrame:
        """Load Unified Patient Record"""
        logger.info(f"Loading UPR from {self.upr_path}")
        self.upr = pd.read_parquet(self.upr_path)
        logger.info(f"Loaded {len(self.upr):,} patients with {len(self.upr.columns)} columns")
        return self.upr
    
    def segment_by_status(self) -> Dict[str, pd.DataFrame]:
        """
        Segment patients by subject status
        
        Returns dict of DataFrames by status
        """
        logger.info("=" * 60)
        logger.info("SEGMENTING BY SUBJECT STATUS")
        logger.info("=" * 60)
        
        # Get status column
        status_col = 'subject_status_clean' if 'subject_status_clean' in self.upr.columns else 'subject_status'
        
        # Get unique statuses
        statuses = self.upr[status_col].unique()
        logger.info(f"Found {len(statuses)} unique statuses: {list(statuses)}")
        
        # Create segments
        self.segments['by_status'] = {}
        
        for status in statuses:
            mask = self.upr[status_col] == status
            segment_df = self.upr[mask].copy()
            self.segments['by_status'][status] = segment_df
            
            pct = len(segment_df) / len(self.upr) * 100
            logger.info(f"  {status}: {len(segment_df):,} patients ({pct:.1f}%)")
        
        # Store statistics
        self.stats['status_distribution'] = {
            status: len(df) for status, df in self.segments['by_status'].items()
        }
        
        return self.segments['by_status']
    
    def segment_by_status_category(self) -> Dict[str, pd.DataFrame]:
        """
        Segment by higher-level status categories
        (active, completed, terminated, unknown)
        """
        logger.info("\nSEGMENTING BY STATUS CATEGORY")
        logger.info("-" * 40)
        
        status_col = 'subject_status_clean' if 'subject_status_clean' in self.upr.columns else 'subject_status'
        
        self.segments['by_category'] = {}
        
        for category, statuses in self.STATUS_CATEGORIES.items():
            mask = self.upr[status_col].isin(statuses)
            segment_df = self.upr[mask].copy()
            self.segments['by_category'][category] = segment_df
            
            pct = len(segment_df) / len(self.upr) * 100 if len(self.upr) > 0 else 0
            logger.info(f"  {category.upper()}: {len(segment_df):,} patients ({pct:.1f}%)")
        
        return self.segments['by_category']
    
    def identify_db_lock_eligible(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify DB Lock eligible population
        
        Returns:
            eligible_df: Patients eligible for DB Lock
            excluded_df: Patients excluded from DB Lock
        """
        logger.info("\n" + "=" * 60)
        logger.info("IDENTIFYING DB LOCK ELIGIBLE POPULATION")
        logger.info("=" * 60)
        
        status_col = 'subject_status_clean' if 'subject_status_clean' in self.upr.columns else 'subject_status'
        
        # Create eligibility mask
        eligible_mask = self.upr[status_col].isin(self.DB_LOCK_ELIGIBLE_STATUSES)
        excluded_mask = ~eligible_mask
        
        # Create DataFrames
        eligible_df = self.upr[eligible_mask].copy()
        excluded_df = self.upr[excluded_mask].copy()
        
        # Add eligibility flag to UPR
        self.upr['db_lock_eligible'] = eligible_mask
        self.upr['db_lock_exclusion_reason'] = None
        self.upr.loc[excluded_mask, 'db_lock_exclusion_reason'] = self.upr.loc[excluded_mask, status_col]
        
        # Store segments
        self.segments['db_lock_eligible'] = eligible_df
        self.segments['db_lock_excluded'] = excluded_df
        
        # Log results
        total = len(self.upr)
        eligible_pct = len(eligible_df) / total * 100 if total > 0 else 0
        excluded_pct = len(excluded_df) / total * 100 if total > 0 else 0
        
        logger.info(f"\nDB LOCK ELIGIBILITY SUMMARY:")
        logger.info(f"  Total Patients: {total:,}")
        logger.info(f"  ✅ Eligible: {len(eligible_df):,} ({eligible_pct:.1f}%)")
        logger.info(f"  ❌ Excluded: {len(excluded_df):,} ({excluded_pct:.1f}%)")
        
        # Breakdown of excluded
        logger.info(f"\nEXCLUSION BREAKDOWN:")
        for status in self.DB_LOCK_EXCLUDE_STATUSES:
            count = (self.upr[status_col] == status).sum()
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"  {status}: {count:,} ({pct:.1f}%)")
        
        # Store stats
        self.stats['db_lock'] = {
            'total': total,
            'eligible': len(eligible_df),
            'excluded': len(excluded_df),
            'eligible_pct': eligible_pct,
            'excluded_pct': excluded_pct
        }
        
        return eligible_df, excluded_df
    
    def tag_screen_failures(self) -> pd.DataFrame:
        """
        Tag screen failures for exclusion from primary analysis
        
        Adds columns:
        - is_screen_failure: Boolean
        - include_in_analysis: Boolean
        - exclusion_reason: String
        """
        logger.info("\n" + "=" * 60)
        logger.info("TAGGING SCREEN FAILURES")
        logger.info("=" * 60)
        
        status_col = 'subject_status_clean' if 'subject_status_clean' in self.upr.columns else 'subject_status'
        
        # Tag screen failures
        self.upr['is_screen_failure'] = self.upr[status_col] == 'Screen Failure'
        
        # Tag for analysis inclusion (exclude screen failures by default)
        self.upr['include_in_analysis'] = ~self.upr['is_screen_failure']
        
        # Set exclusion reason
        self.upr['analysis_exclusion_reason'] = None
        self.upr.loc[self.upr['is_screen_failure'], 'analysis_exclusion_reason'] = 'Screen Failure'
        
        # Also exclude Screening (not yet enrolled)
        screening_mask = self.upr[status_col] == 'Screening'
        self.upr.loc[screening_mask, 'include_in_analysis'] = False
        self.upr.loc[screening_mask, 'analysis_exclusion_reason'] = 'Still Screening'
        
        # Also exclude Unknown (data quality issue)
        unknown_mask = self.upr[status_col] == 'Unknown'
        self.upr.loc[unknown_mask, 'include_in_analysis'] = False
        self.upr.loc[unknown_mask, 'analysis_exclusion_reason'] = 'Unknown Status'
        
        # Statistics
        total = len(self.upr)
        sf_count = self.upr['is_screen_failure'].sum()
        include_count = self.upr['include_in_analysis'].sum()
        exclude_count = total - include_count
        
        logger.info(f"\nSCREEN FAILURE TAGGING:")
        logger.info(f"  Total Patients: {total:,}")
        logger.info(f"  Screen Failures: {sf_count:,} ({sf_count/total*100:.1f}%)")
        
        logger.info(f"\nANALYSIS INCLUSION:")
        logger.info(f"  ✅ Include: {include_count:,} ({include_count/total*100:.1f}%)")
        logger.info(f"  ❌ Exclude: {exclude_count:,} ({exclude_count/total*100:.1f}%)")
        
        # Breakdown by exclusion reason
        logger.info(f"\nEXCLUSION REASONS:")
        for reason in self.upr['analysis_exclusion_reason'].dropna().unique():
            count = (self.upr['analysis_exclusion_reason'] == reason).sum()
            logger.info(f"  {reason}: {count:,}")
        
        # Store stats
        self.stats['screen_failures'] = {
            'count': int(sf_count),
            'pct': float(sf_count / total * 100) if total > 0 else 0
        }
        self.stats['analysis_inclusion'] = {
            'include': int(include_count),
            'exclude': int(exclude_count),
            'include_pct': float(include_count / total * 100) if total > 0 else 0
        }
        
        return self.upr
    
    def create_analysis_cohorts(self) -> Dict[str, pd.DataFrame]:
        """
        Create analysis cohorts based on predefined definitions
        """
        logger.info("\n" + "=" * 60)
        logger.info("CREATING ANALYSIS COHORTS")
        logger.info("=" * 60)
        
        status_col = 'subject_status_clean' if 'subject_status_clean' in self.upr.columns else 'subject_status'
        
        self.cohorts = {}
        
        for cohort_name, definition in self.COHORT_DEFINITIONS.items():
            logger.info(f"\n{cohort_name.upper()}")
            logger.info(f"  Description: {definition['description']}")
            
            # Start with status filter
            mask = self.upr[status_col].isin(definition['statuses'])
            
            # Apply additional filters if defined
            if 'additional_filter' in definition:
                filter_type = definition['additional_filter']
                
                if filter_type == 'has_issues':
                    # Patients with any data quality issues
                    if 'total_issues_all_sources' in self.upr.columns:
                        mask = mask & (self.upr['total_issues_all_sources'] > 0)
                    else:
                        # Fallback: check individual issue columns
                        issue_cols = [c for c in self.upr.columns if 'issue' in c.lower() or 'missing' in c.lower()]
                        if issue_cols:
                            mask = mask & (self.upr[issue_cols].sum(axis=1) > 0)
                
                elif filter_type == 'has_sae':
                    # Patients with SAE
                    sae_cols = [c for c in self.upr.columns if 'sae' in c.lower()]
                    if sae_cols:
                        # Check if any SAE column has values > 0
                        sae_mask = self.upr[sae_cols].fillna(0).sum(axis=1) > 0
                        mask = mask & sae_mask
            
            # Create cohort DataFrame
            cohort_df = self.upr[mask].copy()
            cohort_df['cohort'] = cohort_name
            self.cohorts[cohort_name] = cohort_df
            
            pct = len(cohort_df) / len(self.upr) * 100 if len(self.upr) > 0 else 0
            logger.info(f"  Patients: {len(cohort_df):,} ({pct:.1f}%)")
        
        # Add cohort assignment to UPR (primary cohort)
        self.upr['primary_cohort'] = 'unassigned'
        
        # Assign in priority order
        cohort_priority = [
            'safety_monitoring',      # Highest priority
            'data_quality_focus',
            'active_monitoring',
            'closeout_ready',
            'db_lock_eligible',
            'excluded_from_analysis'  # Lowest priority
        ]
        
        for cohort_name in reversed(cohort_priority):  # Reverse so higher priority overwrites
            if cohort_name in self.cohorts:
                mask = self.upr.index.isin(self.cohorts[cohort_name].index)
                self.upr.loc[mask, 'primary_cohort'] = cohort_name
        
        # Store stats
        self.stats['cohorts'] = {
            name: len(df) for name, df in self.cohorts.items()
        }
        
        return self.cohorts
    
    def segment_by_risk(self) -> Dict[str, pd.DataFrame]:
        """
        Segment patients by risk level
        Uses 'risk_level' column if available, otherwise calculates
        """
        logger.info("\n" + "=" * 60)
        logger.info("SEGMENTING BY RISK LEVEL")
        logger.info("=" * 60)
        
        # Check if risk_level already exists
        if 'risk_level' not in self.upr.columns:
            logger.info("Calculating risk levels...")
            self.upr['risk_level'] = self._calculate_risk_level()
        
        # Create segments
        self.segments['by_risk'] = {}
        
        risk_levels = ['Critical', 'High', 'Medium', 'Low']
        for level in risk_levels:
            mask = self.upr['risk_level'] == level
            segment_df = self.upr[mask].copy()
            self.segments['by_risk'][level] = segment_df
            
            pct = len(segment_df) / len(self.upr) * 100 if len(self.upr) > 0 else 0
            
            # Risk indicator
            indicator = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}.get(level, '⚪')
            logger.info(f"  {indicator} {level}: {len(segment_df):,} ({pct:.1f}%)")
        
        # Store stats
        self.stats['risk_distribution'] = {
            level: len(df) for level, df in self.segments['by_risk'].items()
        }
        
        return self.segments['by_risk']
    
    def _calculate_risk_level(self) -> pd.Series:
        """Calculate risk level based on issue counts"""
        
        # Get issue count column
        if 'total_issues_all_sources' in self.upr.columns:
            issues = self.upr['total_issues_all_sources'].fillna(0)
        else:
            # Sum all numeric columns that might indicate issues
            issue_cols = [c for c in self.upr.columns if any(x in c.lower() for x in ['issue', 'missing', 'open', 'pending'])]
            if issue_cols:
                issues = self.upr[issue_cols].fillna(0).sum(axis=1)
            else:
                issues = pd.Series(0, index=self.upr.index)
        
        # Calculate risk level based on issue count
        def assign_risk(count):
            if count >= 10:
                return 'Critical'
            elif count >= 5:
                return 'High'
            elif count >= 1:
                return 'Medium'
            else:
                return 'Low'
        
        return issues.apply(assign_risk)
    
    def segment_by_study(self) -> Dict[str, pd.DataFrame]:
        """Segment patients by study"""
        logger.info("\n" + "=" * 60)
        logger.info("SEGMENTING BY STUDY")
        logger.info("=" * 60)
        
        self.segments['by_study'] = {}
        
        studies = sorted(self.upr['study_id'].unique())
        logger.info(f"Found {len(studies)} studies")
        
        for study in studies:
            mask = self.upr['study_id'] == study
            segment_df = self.upr[mask].copy()
            self.segments['by_study'][study] = segment_df
            
            pct = len(segment_df) / len(self.upr) * 100 if len(self.upr) > 0 else 0
            logger.info(f"  {study}: {len(segment_df):,} patients ({pct:.1f}%)")
        
        return self.segments['by_study']
    
    def create_segment_summary(self) -> pd.DataFrame:
        """Create summary DataFrame of all segments"""
        logger.info("\n" + "=" * 60)
        logger.info("CREATING SEGMENT SUMMARY")
        logger.info("=" * 60)
        
        summary_data = []
        
        # Status segments
        if 'by_status' in self.segments:
            for status, df in self.segments['by_status'].items():
                summary_data.append({
                    'segment_type': 'Status',
                    'segment_name': status,
                    'patient_count': len(df),
                    'pct_of_total': len(df) / len(self.upr) * 100 if len(self.upr) > 0 else 0
                })
        
        # Category segments
        if 'by_category' in self.segments:
            for category, df in self.segments['by_category'].items():
                summary_data.append({
                    'segment_type': 'Category',
                    'segment_name': category,
                    'patient_count': len(df),
                    'pct_of_total': len(df) / len(self.upr) * 100 if len(self.upr) > 0 else 0
                })
        
        # Risk segments
        if 'by_risk' in self.segments:
            for level, df in self.segments['by_risk'].items():
                summary_data.append({
                    'segment_type': 'Risk',
                    'segment_name': level,
                    'patient_count': len(df),
                    'pct_of_total': len(df) / len(self.upr) * 100 if len(self.upr) > 0 else 0
                })
        
        # Cohorts
        for cohort_name, df in self.cohorts.items():
            summary_data.append({
                'segment_type': 'Cohort',
                'segment_name': cohort_name,
                'patient_count': len(df),
                'pct_of_total': len(df) / len(self.upr) * 100 if len(self.upr) > 0 else 0
            })
        
        # DB Lock eligibility
        if 'db_lock' in self.stats:
            summary_data.append({
                'segment_type': 'DB Lock',
                'segment_name': 'Eligible',
                'patient_count': self.stats['db_lock']['eligible'],
                'pct_of_total': self.stats['db_lock']['eligible_pct']
            })
            summary_data.append({
                'segment_type': 'DB Lock',
                'segment_name': 'Excluded',
                'patient_count': self.stats['db_lock']['excluded'],
                'pct_of_total': self.stats['db_lock']['excluded_pct']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        logger.info(f"Summary contains {len(summary_df)} segment entries")
        
        return summary_df
    
    def save_outputs(self, output_dir: Path) -> Dict[str, Path]:
        """
        Save all segmentation outputs
        """
        logger.info("\n" + "=" * 60)
        logger.info("SAVING SEGMENTATION OUTPUTS")
        logger.info("=" * 60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 1. Save updated UPR with segment flags
        upr_path = output_dir / 'unified_patient_record_segmented.parquet'
        self.upr.to_parquet(upr_path, index=False)
        saved_files['upr_segmented'] = upr_path
        logger.info(f"✅ Saved segmented UPR: {upr_path}")
        
        # 2. Save segment summary
        summary_df = self.create_segment_summary()
        summary_path = output_dir / 'segment_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        saved_files['segment_summary'] = summary_path
        logger.info(f"✅ Saved segment summary: {summary_path}")
        
        # 3. Save DB Lock eligible population
        if 'db_lock_eligible' in self.segments:
            dbl_path = output_dir / 'db_lock_eligible_population.parquet'
            self.segments['db_lock_eligible'].to_parquet(dbl_path, index=False)
            saved_files['db_lock_eligible'] = dbl_path
            logger.info(f"✅ Saved DB Lock eligible: {dbl_path}")
        
        # 4. Save analysis cohorts
        cohorts_dir = output_dir / 'cohorts'
        cohorts_dir.mkdir(exist_ok=True)
        for cohort_name, cohort_df in self.cohorts.items():
            cohort_path = cohorts_dir / f'{cohort_name}.parquet'
            cohort_df.to_parquet(cohort_path, index=False)
            saved_files[f'cohort_{cohort_name}'] = cohort_path
        logger.info(f"✅ Saved {len(self.cohorts)} cohorts to {cohorts_dir}")
        
        # 5. Save segmentation manifest
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'total_patients': len(self.upr),
            'statistics': self.stats,
            'segments': {
                seg_type: list(segs.keys()) 
                for seg_type, segs in self.segments.items()
            },
            'cohorts': list(self.cohorts.keys()),
            'columns_added': [
                'db_lock_eligible',
                'db_lock_exclusion_reason',
                'is_screen_failure',
                'include_in_analysis',
                'analysis_exclusion_reason',
                'primary_cohort',
                'risk_level'
            ],
            'files_saved': {k: str(v) for k, v in saved_files.items()}
        }
        
        manifest_path = output_dir / 'segmentation_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        saved_files['manifest'] = manifest_path
        logger.info(f"✅ Saved manifest: {manifest_path}")
        
        return saved_files
    
    def run_full_segmentation(self) -> Dict:
        """
        Run complete segmentation pipeline
        """
        logger.info("\n" + "=" * 70)
        logger.info("SANCHALAK AI - POPULATION SEGMENTATION ENGINE v1.0")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # Load data
        self.load_upr()
        
        # Run all segmentation steps
        self.segment_by_status()
        self.segment_by_status_category()
        self.identify_db_lock_eligible()
        self.tag_screen_failures()
        self.create_analysis_cohorts()
        self.segment_by_risk()
        self.segment_by_study()
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 70)
        logger.info("SEGMENTATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Total Patients: {len(self.upr):,}")
        logger.info(f"Segments Created: {sum(len(s) for s in self.segments.values())}")
        logger.info(f"Cohorts Created: {len(self.cohorts)}")
        
        return {
            'upr': self.upr,
            'segments': self.segments,
            'cohorts': self.cohorts,
            'stats': self.stats,
            'duration': duration
        }


def main():
    """Main entry point for segmentation"""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    upr_path = project_root / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
    output_dir = project_root / 'data' / 'processed' / 'segments'
    
    # Check UPR exists
    if not upr_path.exists():
        logger.error(f"UPR not found at {upr_path}")
        logger.error("Please run Phase 1.3 (UPR Builder) first")
        return
    
    # Run segmentation
    segmenter = PopulationSegmenter(upr_path)
    results = segmenter.run_full_segmentation()
    
    # Save outputs
    saved_files = segmenter.save_outputs(output_dir)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("📊 PHASE 1.4 COMPLETE - POPULATION SEGMENTATION")
    print("=" * 70)
    print(f"\n✅ Total Patients: {len(results['upr']):,}")
    print(f"✅ DB Lock Eligible: {results['stats']['db_lock']['eligible']:,} ({results['stats']['db_lock']['eligible_pct']:.1f}%)")
    print(f"✅ Screen Failures: {results['stats']['screen_failures']['count']:,} ({results['stats']['screen_failures']['pct']:.1f}%)")
    print(f"✅ Analysis Population: {results['stats']['analysis_inclusion']['include']:,} ({results['stats']['analysis_inclusion']['include_pct']:.1f}%)")
    print(f"\n📁 Output Directory: {output_dir}")
    print(f"📁 Files Saved: {len(saved_files)}")
    
    return results


if __name__ == '__main__':
    main()